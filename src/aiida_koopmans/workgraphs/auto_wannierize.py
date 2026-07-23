"""Automated block splitting during Wannierisation (Wannier.jl parallel transport).

A projection block whose bands separate into energy-isolated groups — or
straddle the occupied/empty boundary — is Wannierised once as a whole, split
into per-group manifolds with
`aiida-wannierjl <https://github.com/elinscott/aiida-wannierjl>`_
(``Wannier.Tools.mrwf`` parallel transport, including the cubic b-vector
stencil fallback), re-Wannierised group by group without disentanglement, and
the per-group products (``_u.mat`` / ``_hr.dat`` / ``_centres.xyz``) merged
back into one block-diagonal file set.

The group detection is data-dependent (it reads the eigenvalues of a pw.x
``bands`` run), so the split-vs-plain decision and the per-group fan-out
cannot be drawn at graph-construction time. The standard nested-deferred-graph
pattern applies: :func:`WannierizeAndSplitBlocks` (the top-level builder) wires
the runtime ``detect_band_groups`` result into one nested
:func:`WannierizeAndSplitBlock` graph per block; when that nested body runs the
groups are concrete values and ordinary ``if`` / ``for`` build the branch.

Scope (mirroring the PR that introduces this module): explicitly-projected
blocks and a single spin channel. Implicit/automatic projections and the
``_u_dis.mat`` merge of a disentangled parent block are follow-ups.
"""

from __future__ import annotations

import io
from typing import Annotated, Any, TypedDict

import numpy as np
from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90.calculations import Wannier90Calculation
from aiida_wannierjl.workflows import split_wannierization
from aiida_workgraph import dynamic, task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.projections import (
    detect_band_blocks,
    groups_to_wannier_indices,
    restrict_groups_to_block,
)
from aiida_koopmans.types import ParallelizationDict, ProjectionBlock
from aiida_koopmans.wannier_merge import (
    merge_wannier_centres_file_contents,
    merge_wannier_hr_file_contents,
    merge_wannier_u_file_contents,
)
from aiida_koopmans.workgraphs import (
    Codes,
    merge_parallelization_into_inputs,
    validate_parallelization,
)
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlock, WannierizeOverrides
from aiida_koopmans.workgraphs.pw import PwBaseStep, PwOutputs, RunScfNscf

Wannier90CalcStep = task(Wannier90Calculation)

#: Seedname shared by every wannier90-family calculation in the chain (the
#: aiida-wannier90 / aiida-wannierjl default).
SEEDNAME = "aiida"

#: Disentanglement keywords that must never reach a split sub-block: the
#: parallel-transport manifolds have ``num_bands == num_wann`` by
#: construction, so there is nothing to disentangle from.
_DIS_KEYS = (
    "dis_win_min",
    "dis_win_max",
    "dis_froz_min",
    "dis_froz_max",
    "dis_num_iter",
    "dis_mix_ratio",
    "dis_conv_tol",
    "dis_conv_window",
)

#: Fallback ``metadata.options`` for the raw CalcJobs this module creates
#: directly (the protocol-built steps carry their own defaults). A CalcJob
#: cannot run without ``resources``; MPI behaviour follows the code node.
_DEFAULT_CALCJOB_OPTIONS: dict[str, Any] = {"resources": {"num_machines": 1}}


def _plain_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """Return CalcJob options as freshly-built plain dicts.

    Inside a graph body, dict-valued graph inputs arrive as provenance-tagged
    ``TaggedValue`` proxies, which node-graph refuses to assign into a
    namespace socket (``metadata.options``); rebuilding the mapping tree
    strips the proxies while leaving leaf scalars alone.
    """
    from collections.abc import Mapping

    def rebuild(mapping: Mapping) -> dict[str, Any]:
        return {
            str(key): rebuild(val) if isinstance(val, Mapping) else val
            for key, val in mapping.items()
        }

    return rebuild(options) if options else _DEFAULT_CALCJOB_OPTIONS


# ----------------------------------------------------------------------
# Leaf tasks
# ----------------------------------------------------------------------


@task.calcfunction
def detect_band_groups(
    bands: orm.BandsData,
    num_occ_bands: int | None = None,
    threshold: float | None = None,
    num_bands_total: int | None = None,
    spin_channel_index: int = 0,
) -> orm.List:
    """Detect the energy-separated band groups of a bands calculation.

    Thin runtime wrapper around
    :func:`aiida_koopmans.projections.detect_band_blocks`: reads the
    eigenvalues out of ``bands`` (the ``output_band`` of a pw.x ``bands``
    run along the k-path), restricts them to the first ``num_bands_total``
    bands (the Wannierised manifold — the disentanglement pool above it must
    not influence the grouping), and returns the 1-indexed groups. A
    calcfunction (not a plain ``@task``): it takes AiiDA data nodes, which
    the PyFunction deserializer refuses.
    """
    energies = np.asarray(bands.get_bands(), dtype=float)
    if energies.ndim == 3:
        energies = energies[int(spin_channel_index)]
    if num_bands_total is not None:
        energies = energies[:, : int(num_bands_total)]
    return orm.List(
        detect_band_blocks(
            energies,
            num_occ_bands=None if num_occ_bands is None else int(num_occ_bands),
            threshold=None if threshold is None else float(threshold),
        )
    )


@task.calcfunction
def extract_win_file(retrieved: orm.FolderData) -> orm.SinglefileData:
    """Recover the ``.win`` input of the wannier90 run that created ``retrieved``.

    The Wannier.jl CalcJobs need the ``.win`` as an explicit
    :class:`~aiida.orm.SinglefileData`, but a wannier90 run keeps its input
    file only in the calculation node's repository — so read it back off
    ``retrieved``'s creator. A calcfunction (not a plain ``@task``): it
    takes an AiiDA data node, which the PyFunction deserializer refuses.
    """
    creator = retrieved.creator
    if creator is None:
        raise ValueError("`retrieved` has no creating calculation to read the .win from.")
    filename = creator.get_option("input_filename") or f"{SEEDNAME}.win"
    content = creator.base.repository.get_object_content(filename, mode="rb")
    return orm.SinglefileData(io.BytesIO(content), filename=filename)


@task.calcfunction(outputs=["u_file", "hr_file", "centres_file"])
def merge_split_block_products(**retrieved: orm.FolderData) -> dict:
    """Merge per-sub-block wannier90 products back into one block-wide set.

    ``retrieved`` holds the sub-block wannier90 ``retrieved`` folders, keyed
    so lexicographic order matches the band order of the groups (``b00``,
    ``b01``, ...). The ``_u.mat`` / ``_hr.dat`` merges are block-diagonal and
    the ``_centres.xyz`` centres are concatenated — see
    :mod:`aiida_koopmans.wannier_merge` for the invariants.
    """
    folders = [retrieved[key] for key in sorted(retrieved)]

    def _contents(suffix: str) -> list[str]:
        return [
            folder.base.repository.get_object_content(f"{SEEDNAME}{suffix}", mode="r")
            for folder in folders
        ]

    def _single(content: str, suffix: str) -> orm.SinglefileData:
        return orm.SinglefileData(io.BytesIO(content.encode()), filename=f"{SEEDNAME}{suffix}")

    return {
        "u_file": _single(merge_wannier_u_file_contents(_contents("_u.mat")), "_u.mat"),
        "hr_file": _single(merge_wannier_hr_file_contents(_contents("_hr.dat")), "_hr.dat"),
        "centres_file": _single(
            merge_wannier_centres_file_contents(_contents("_centres.xyz")), "_centres.xyz"
        ),
    }


def _subblock_w90_parameters(
    num_wann: int, mp_grid: list[int], wannier90_overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Wannier90 parameters for re-Wannierising one split sub-block.

    The sub-block reads the split ``.amn`` / ``.mmn`` / ``.eig`` directly
    (``num_bands == num_wann``, no preprocessing, no disentanglement, no
    band exclusion — the split files already cover exactly the group's
    bands). User ``.win`` keywords propagate, minus the per-block counts
    and the disentanglement set.
    """
    dropped = (*_DIS_KEYS, "num_wann", "num_bands", "exclude_bands", "projections")
    params = {
        key: value for key, value in (wannier90_overrides or {}).items() if key not in dropped
    }
    params.update(
        num_wann=int(num_wann),
        num_bands=int(num_wann),
        mp_grid=[int(x) for x in mp_grid],
        write_hr=True,
        write_u_matrices=True,
        write_xyz=True,
    )
    return params


# ----------------------------------------------------------------------
# Sub-block re-Wannierisation (nested: receives the resolved split folders)
# ----------------------------------------------------------------------


class RewannierizeSplitOutputs(TypedDict):
    """Outputs of :func:`RewannierizeSplitBlocks`.

    The merged block-wide product files plus the per-sub-block wannier90
    ``retrieved`` folders (keyed ``b00``, ``b01``, ... in band order).
    """

    u_file: orm.SinglefileData
    hr_file: orm.SinglefileData
    centres_file: orm.SinglefileData
    subblock_retrieved: Annotated[dict, dynamic(orm.FolderData)]


@task.graph
def RewannierizeSplitBlocks(
    codes: Codes,
    structure: orm.StructureData,
    split_blocks: Annotated[dict, dynamic(orm.FolderData)],
    group_sizes: list[int],
    kpoints: orm.KpointsData,
    mp_grid: list[int],
    wannier90_overrides: dict[str, Any] | None = None,
    wannier90_options: dict[str, Any] | None = None,
) -> RewannierizeSplitOutputs:
    """Re-Wannierise each split sub-block and merge the products.

    The keys of the ``SplitCalculation``'s dynamic ``blocks`` namespace only
    exist once it has run, so the whole namespace is passed into this nested
    graph; when this body executes ``split_blocks`` is a resolved
    ``{"block_0": FolderData, ...}`` dict and the per-group fan-out is a
    native ``for`` loop. Each sub-block runs a preprocessing-free
    ``Wannier90Calculation`` on the split ``.amn``/``.mmn``/``.eig``
    (``local_input_folder``), then the ``_u.mat`` / ``_hr.dat`` /
    ``_centres.xyz`` products are merged block-diagonally.
    """
    subblock_retrieved: dict[str, Any] = {}
    for i, num_wann in enumerate(group_sizes):
        rewannierized = Wannier90CalcStep(
            code=codes["wannier90"],
            structure=structure,
            parameters=_subblock_w90_parameters(int(num_wann), mp_grid, wannier90_overrides),
            kpoints=kpoints,
            local_input_folder=split_blocks[f"block_{i}"],
            metadata={
                "call_link_label": f"wannier90_split_block_{i}",
                "options": _plain_options(wannier90_options),
            },
        )
        subblock_retrieved[f"b{i:02d}"] = rewannierized["retrieved"]

    merged = merge_split_block_products(
        **subblock_retrieved,
        metadata={"call_link_label": "merge_split_block_products"},
    )

    return RewannierizeSplitOutputs(
        u_file=merged["u_file"],
        hr_file=merged["hr_file"],
        centres_file=merged["centres_file"],
        subblock_retrieved=subblock_retrieved,
    )


# ----------------------------------------------------------------------
# Per-block graph (nested, deferred: receives the resolved groups)
# ----------------------------------------------------------------------


class AutoWannierizeBlockRequiredOutputs(TypedDict):
    """Whole-block Wannierisation outputs, present on both branches.

    The same shape as ``WannierizeBlockOutputs`` — the whole-block
    Wannierisation always runs (the split, when it triggers, starts from its
    checkpoint).
    """

    hr_retrieved: orm.FolderData
    remote_folder: orm.RemoteData
    nnkp_file: orm.SinglefileData


class AutoWannierizeBlockOutputs(AutoWannierizeBlockRequiredOutputs, total=False):
    """Outputs of :func:`WannierizeAndSplitBlock`.

    The merged product files and the per-sub-block ``retrieved`` namespace
    exist only when the block was actually split (``total=False``: the
    unsplit branch simply does not populate them).
    """

    u_file: orm.SinglefileData
    hr_file: orm.SinglefileData
    centres_file: orm.SinglefileData
    subblock_retrieved: Annotated[dict, dynamic(orm.FolderData)]


@task.graph
def WannierizeAndSplitBlock(
    codes: Codes,
    structure: orm.StructureData,
    block: ProjectionBlock,
    groups: list[list[int]],
    nscf_remote_folder: orm.RemoteData,
    kpoints: orm.KpointsData,
    mp_grid: list[int],
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
    parallelization: ParallelizationDict | None = None,
    wjl_options: dict[str, Any] | None = None,
    wannier90_options: dict[str, Any] | None = None,
    pw2wannier90_options: dict[str, Any] | None = None,
) -> AutoWannierizeBlockOutputs:
    """Wannierise one block, splitting it into detected groups when needed.

    Called as a nested graph task with ``groups`` wired from
    :func:`detect_band_groups`, so by the time this body runs the groups are
    concrete and the split-vs-plain decision is an ordinary ``if``:

    * one detected group overlapping the block — the block is already
      isolated; only the plain :func:`WannierizeBlock` runs;
    * several groups — the whole-block Wannierisation is followed by the
      aiida-wannierjl ``split_wannierization`` chain (cubic-stencil check,
      optional cubic ``.mmn`` regeneration off the shared nscf scratch, and
      the ``mrwf`` split), one preprocessing-free wannier90 run per group on
      the split ``.amn``/``.mmn``/``.eig``, and the block-diagonal product
      merge.

    The groups arrive as global band indices; they are restricted to the
    block and re-based to the block's 1-based Wannier indices before the
    split (Wannier.jl indexes the model's Wannier functions, not global
    bands, so a block that does not start at band 1 must be re-based —
    handing the split global indices would mis-address its Wannier
    functions).
    """
    overrides = overrides or {}

    whole = WannierizeBlock(
        codes=codes,
        structure=structure,
        block=block,
        projection_type=block["projection_type"],
        nscf_remote_folder=nscf_remote_folder,
        kpoints=kpoints,
        mp_grid=mp_grid,
        pseudo_family=pseudo_family,
        protocol=protocol,
        overrides=overrides,
        electronic_type=electronic_type,
        spin_type=spin_type,
        parallelization=parallelization,
        metadata={"call_link_label": "wannierize_whole_block"},
    )

    outputs = AutoWannierizeBlockOutputs(
        hr_retrieved=whole["hr_retrieved"],
        remote_folder=whole["remote_folder"],
        nnkp_file=whole["nnkp_file"],
    )

    local_groups = restrict_groups_to_block(list(groups), list(block["include_bands"]))
    if len(local_groups) <= 1:
        return outputs

    wann_groups = [
        [int(index) for index in group]
        for group in groups_to_wannier_indices(local_groups, list(block["include_bands"]))
    ]

    win_file = extract_win_file(retrieved=whole["hr_retrieved"]).result

    # The wannier90 scratch holds every file the split needs: ``aiida.chk``
    # plus the ``aiida.{amn,mmn,eig}`` symlinks that aiida-wannier90 staged
    # from the pw2wannier90 scratch — so it serves as both parent folders.
    # The nscf scratch and pw2wannier90 code feed the cubic-stencil branch.
    split = split_wannierization(
        wjl_code=codes["wannierjl"],
        win_file=win_file,
        groups=wann_groups,
        wannier90_parent=whole["remote_folder"],
        pw2wannier90_parent=whole["remote_folder"],
        nscf_parent=nscf_remote_folder,
        pw2wannier90_code=codes["pw2wannier90"],
        wjl_options=wjl_options,
        pw2wannier90_options=pw2wannier90_options,
        metadata={"call_link_label": "split_wannierization"},
    )

    # The split's ``blocks`` namespace keys only exist once it has run, so
    # the re-Wannierisation consumes the whole namespace in a nested graph.
    rewannierized = RewannierizeSplitBlocks(
        codes=codes,
        structure=structure,
        split_blocks=split["blocks"],
        group_sizes=[len(group) for group in wann_groups],
        kpoints=kpoints,
        mp_grid=mp_grid,
        wannier90_overrides=overrides.get("wannier90"),
        wannier90_options=wannier90_options,
        metadata={"call_link_label": "rewannierize_split_blocks"},
    )

    outputs["u_file"] = rewannierized["u_file"]
    outputs["hr_file"] = rewannierized["hr_file"]
    outputs["centres_file"] = rewannierized["centres_file"]
    outputs["subblock_retrieved"] = rewannierized["subblock_retrieved"]
    return outputs


# ----------------------------------------------------------------------
# Top-level graph
# ----------------------------------------------------------------------


class WannierizeAndSplitBlocksOutputs(TypedDict):
    """Outputs of :func:`WannierizeAndSplitBlocks`.

    * ``nscf`` — the shared nscf :class:`PwOutputs` (scratch + eigenvalues).
    * ``bands`` — the pw.x ``bands``-run eigenvalues the grouping was
      detected on.
    * ``groups`` — the detected 1-indexed band groups (global indices).
    * ``blocks`` — a dynamic namespace keyed by block label; each entry is an
      :class:`AutoWannierizeBlockOutputs`.
    """

    nscf: PwOutputs
    bands: orm.BandsData
    groups: list[list[int]]
    blocks: Annotated[dict, dynamic(AutoWannierizeBlockOutputs)]


@task.graph
def WannierizeAndSplitBlocks(
    codes: Codes,
    structure: orm.StructureData,
    blocks: list[ProjectionBlock],
    kpoints: orm.KpointsData,
    bands_kpoints: orm.KpointsData,
    num_occ_bands: int,
    threshold: float | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
    parallelization: ParallelizationDict | None = None,
    wjl_options: dict[str, Any] | None = None,
    wannier90_options: dict[str, Any] | None = None,
    pw2wannier90_options: dict[str, Any] | None = None,
) -> WannierizeAndSplitBlocksOutputs:
    """Wannierise a system block-by-block with automated block splitting.

    The shared scf + nscf pair runs once (as in ``WannierizeBlocks``), plus a
    pw.x ``bands`` step along ``bands_kpoints`` off the scf density. The
    runtime :func:`detect_band_groups` task turns the bands eigenvalues into
    energy-separated groups (always splitting at the occupied/empty boundary
    ``num_occ_bands``, and at every gap wider than ``threshold`` eV); each
    projection block is then handled by a nested
    :func:`WannierizeAndSplitBlock` that receives the resolved groups.

    Args:
        codes: code instances. Required: ``pw``, ``wannier90``,
            ``pw2wannier90`` and ``wannierjl`` (the julia binary registered
            via ``aiida_wannierjl.helpers.get_wannierjl_code``).
        structure: the periodic ``StructureData``.
        blocks: the explicit projection blocks (single spin channel).
        kpoints: the Monkhorst-Pack mesh ``KpointsData``; expanded once into
            the explicit list shared by the nscf and every wannier90 step.
        bands_kpoints: the k-path for the pw.x ``bands`` run the group
            detection reads.
        num_occ_bands: occupied-band count of the channel (the detection
            always opens a new group at this boundary).
        threshold: minimum gap (eV) between consecutive bands for a split;
            ``None`` splits only at the occupied/empty boundary.
        pseudo_family / protocol / overrides / electronic_type / spin_type:
            as in ``WannierizeBlocks`` (the ``nscf`` override entry also
            seeds the bands step, so e.g. ``nbnd`` stays consistent).
        parallelization: per-code parallelization mapping (keyed by code
            name), threaded into the shared scf/nscf pair, the bands step,
            and each block's ``pw2wannier90`` / ``wannier90`` steps. The
            Wannier.jl split CalcJobs and the per-group re-wannierisation run
            outside this mapping (see ``wjl_options`` / ``wannier90_options``).
        wjl_options / pw2wannier90_options: optional ``metadata.options``
            for the Wannier.jl CalcJobs and the cubic pw2wannier90 run. Both
            CalcJobs carry their own ``resources`` defaults, so normally
            leave these unset: a non-None dict currently trips
            aiida-wannierjl's options handling inside a graph body (dict
            graph inputs arrive as TaggedValue proxies, which node-graph
            refuses to assign into ``metadata.options``).
        wannier90_options: optional ``metadata.options`` for the per-group
            re-wannierisation ``Wannier90Calculation`` (defaults to
            single-machine resources; that CalcJob has no resources
            default of its own).

    Returns:
        A :class:`WannierizeAndSplitBlocksOutputs`.
    """
    from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
    from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints

    validate_parallelization(parallelization)

    overrides = overrides or {}
    # ``.build()`` executes this body eagerly, where graph inputs arrive as
    # provenance-tagged proxies; the family label ends up bound as an SQL
    # parameter inside ``get_builder_from_protocol``, which needs a plain str.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None

    mp_grid = [int(x) for x in kpoints.get_kpoints_mesh()[0]]
    explicit_kpoints = get_explicit_kpoints(kpoints)

    # --- shared scf + nscf (run once) ---
    scf_nscf_overrides: dict[str, Any] = {}
    if "scf" in overrides:
        scf_nscf_overrides["scf"] = overrides["scf"]
    if "nscf" in overrides:
        scf_nscf_overrides["nscf"] = overrides["nscf"]
    scf_nscf = RunScfNscf(
        code=codes["pw"],
        structure=structure,
        pseudo_family=pseudo_family,
        protocol=protocol,
        overrides=scf_nscf_overrides or None,
        parallelization=parallelization,
        nscf_kpoints=explicit_kpoints,
        metadata={"call_link_label": "scf_nscf"},
    )
    nscf_remote_folder = scf_nscf["nscf_remote_folder"]

    # --- bands run along the k-path, off the scf density ---
    # Seeded from the nscf override entry (same nbnd / cutoffs), with the
    # calculation type forced on top.
    bands_overrides = recursive_merge(
        dict(overrides.get("nscf", {})),
        {"pw": {"parameters": {"CONTROL": {"calculation": "bands"}}}},
    )
    if pseudo_family is not None:
        bands_overrides.setdefault("pseudo_family", pseudo_family)
    bands_builder = PwBaseWorkChain.get_builder_from_protocol(
        code=codes["pw"],
        structure=structure,
        protocol=protocol,
        overrides=bands_overrides,
        electronic_type=electronic_type,
    )
    bands_data = get_dict_from_builder(bands_builder)
    bands_data.pop("clean_workdir", None)
    # The workchain accepts exactly one of ``kpoints`` / ``kpoints_distance``.
    bands_data.pop("kpoints_distance", None)
    bands_data.pop("kpoints_force_parity", None)
    bands_data["kpoints"] = bands_kpoints
    bands_data["pw"]["parent_folder"] = scf_nscf["scf_remote_folder"]
    merge_parallelization_into_inputs(bands_data["pw"], parallelization, "pw")
    bands_data.setdefault("metadata", {})["call_link_label"] = "bands"
    bands_outputs = PwBaseStep(**bands_data)

    # --- runtime group detection over the Wannierised manifold ---
    num_bands_total = sum(int(block["num_wann"]) for block in blocks)
    detect = detect_band_groups(
        bands=bands_outputs["output_band"],
        num_occ_bands=num_occ_bands,
        threshold=threshold,
        num_bands_total=num_bands_total,
    )

    # --- per-block fan-out: native for-loop over the concrete blocks ---
    block_outputs: dict[str, Any] = {}
    for block in blocks:
        block_outputs[block["label"]] = WannierizeAndSplitBlock(
            codes=codes,
            structure=structure,
            block=block,
            groups=detect.result,
            nscf_remote_folder=nscf_remote_folder,
            kpoints=explicit_kpoints,
            mp_grid=mp_grid,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides=overrides,
            electronic_type=electronic_type,
            spin_type=spin_type,
            parallelization=parallelization,
            wjl_options=wjl_options,
            wannier90_options=wannier90_options,
            pw2wannier90_options=pw2wannier90_options,
            metadata={"call_link_label": f"wannierize_split_{block['label']}"},
        )

    return WannierizeAndSplitBlocksOutputs(
        nscf=PwOutputs(
            remote_folder=nscf_remote_folder,
            output_parameters=scf_nscf["nscf_output_parameters"],
            output_band=scf_nscf["nscf_output_band"],
        ),
        bands=bands_outputs["output_band"],
        groups=detect.result,
        blocks=block_outputs,
    )
