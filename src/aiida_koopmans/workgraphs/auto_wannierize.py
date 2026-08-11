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
pattern applies: the split mode of
:func:`~aiida_koopmans.workgraphs.block_wannierize.WannierizeBlocks` (the
entry point) wires the runtime :func:`detect_band_groups` result into one
nested :func:`WannierizeAndSplitBlock` graph per block; when that nested body
runs the groups are concrete values and ordinary ``if`` / ``for`` build the
branch. This module holds the split-specific pieces only.

Scope: a single spin channel. Blocks may be explicitly projected (ANALYTIC)
or automatic (pseudoatomic projectors — no per-orbital list; the whole-block
run relies on ``projection_type``, plus the external projector inputs for
the external source). The ``_u_dis.mat`` merge of a
disentangled parent block is a follow-up: a block routed through the split
must not require disentanglement (``num_bands == num_wann``), which
:func:`~aiida_koopmans.workgraphs.block_wannierize._resolve_split_mode`
enforces at build time. The per-group re-Wannierisation reads only the
parent's gauge products, so a parent's disentanglement matrix would be
dropped on the floor rather than carried into the sub-blocks.
"""

import copy
import io
from typing import Annotated, Any, NotRequired, TypedDict

import numpy as np
from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_wannier90.calculations import Wannier90Calculation
from aiida_wannierjl.workflows import split_wannierization
from aiida_workgraph import dynamic, task
from aiida_workgraph.socket_spec import SocketMeta

from aiida_koopmans.parallelization import ParallelizationDict
from aiida_koopmans.projections import (
    ProjectionBlock,
    detect_band_blocks,
    get_wannier_indices,
    groups_to_wannier_indices,
    restrict_groups_to_block,
)
from aiida_koopmans.workgraphs.block_wannierize import (
    WannierizeBlock,
    WannierizeBlockOutputs,
    WannierizeOverrides,
)
from aiida_koopmans.workgraphs.pw import PwCode, assemble_pw_base_step
from aiida_koopmans.workgraphs.utils.wannier_merge import (
    merge_wannier_centres_file_contents,
    merge_wannier_hr_file_contents,
    merge_wannier_u_file_contents,
)
from aiida_koopmans.workgraphs.wannier90 import (
    Pw2Wannier90Code,
    Wannier90Code,
    require_path_labels,
)


class SplitBlockCodes(TypedDict):
    """Codes for the wannierize-and-split path (:func:`WannierizeAndSplitBlock`)."""

    pw: PwCode
    pw2wannier90: Pw2Wannier90Code
    wannier90: Wannier90Code
    wannierjl: Annotated[
        orm.AbstractCode,
        SocketMeta(help="Needed to split Wannier function blocks by parallel transport."),
    ]


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

#: Convergence keywords copied from the parent whole-block run's resolved
#: wannier90 parameters into each split sub-block. The upstream protocol
#: always sets them (``num_iter`` / ``num_cg_steps`` / ``conv_window`` per
#: tier, ``conv_tol`` derived from its per-atom meta-parameter); without the
#: copy the re-Wannierizations would run at wannier90's compiled-in defaults
#: (``num_iter = 100``, ``num_cg_steps = 5``), which can leave a sub-block
#: far from its spread minimum. The parent's ``dis_*`` keywords stay behind
#: (``num_bands == num_wann``: nothing to disentangle), and the structural
#: keys are rebuilt per block by :func:`_subblock_w90_parameters`.
_PARENT_CONVERGENCE_KEYS = ("num_iter", "num_cg_steps", "conv_tol", "conv_window")

#: Fallback ``metadata.options`` for the raw CalcJobs this module creates
#: directly (the protocol-built steps carry their own defaults). A CalcJob
#: cannot run without ``resources``; MPI behaviour follows the code node. The
#: rank count is written out rather than left to the scheduler, which would
#: otherwise resolve it against the computer's default at submission time.
_DEFAULT_CALCJOB_OPTIONS: dict[str, Any] = {
    "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1}
}


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


def add_bands_step(
    pw_code: orm.AbstractCode,
    structure: orm.StructureData,
    bands_kpoints: orm.KpointsData,
    scf_remote_folder: orm.RemoteData,
    nscf_overrides: dict[str, Any] | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    parallelization: ParallelizationDict | None = None,
) -> Any:
    """Assemble a pw.x ``bands`` step along ``bands_kpoints`` off an scf density.

    A plain graph-assembly helper, not a task: it must be called inside a
    ``@task.graph`` body, where the ``PwBaseStep`` it creates joins the
    surrounding graph (``call_link_label`` ``bands``). The step is seeded
    from the caller's nscf protocol overrides — so e.g. ``nbnd`` and the
    cutoffs stay consistent with the nscf — with the calculation type forced
    on top, and reads the density from ``scf_remote_folder``. Returns the
    step's outputs (``output_band`` holds the eigenvalues along the path).
    """
    # ``.build()`` executes graph bodies eagerly, where graph inputs arrive as
    # provenance-tagged proxies; the family label ends up bound as an SQL
    # parameter inside ``get_builder_from_protocol``, which needs a plain str.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None

    # Deep-copy the seed: the shared assembly stamps this step's calculation
    # type into the overrides, which must never leak into the caller's nscf
    # override through shared nested dicts.
    bands_overrides = copy.deepcopy(dict(nscf_overrides or {}))
    # A calculation type riding along in the seed is residue, not a conflict.
    bands_overrides.get("pw", {}).get("parameters", {}).get("CONTROL", {}).pop("calculation", None)
    if pseudo_family is not None:
        bands_overrides.setdefault("pseudo_family", pseudo_family)
    return assemble_pw_base_step(
        pw_code,
        structure,
        calculation="bands",
        call_link_label="bands",
        overrides=bands_overrides,
        protocol=protocol,
        electronic_type=electronic_type,
        kpoints=bands_kpoints,
        parent_folder=scf_remote_folder,
        parallelization=parallelization,
    )


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
    bands (the Wannierised manifold — the extra disentanglement bands above
    it must not influence the grouping), and returns the 1-indexed groups. A
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
    :mod:`aiida_koopmans.workgraphs.utils.wannier_merge` for the invariants.
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


@task.calcfunction
def merge_wannier_output_parameters(**output_parameters: orm.Dict) -> orm.Dict:
    """Concatenate per-group parsed wannier90 outputs into one block-wide Dict.

    ``output_parameters`` holds the per-group re-Wannierisation outputs,
    keyed so lexicographic order matches the group (= band) order (``b00``,
    ``b01``, ...). The per-WF ``wannier_functions_output`` tables are
    concatenated in that order, entries sorted by their run-local
    ``wf_ids`` and re-based to a block-wide 1-based numbering, and
    ``number_wfs`` is summed. Only these honestly mergeable keys are
    carried; per-run scalars such as the ``Omega_*`` decomposition are
    dropped rather than fabricated. This threads parsed outputs
    (concatenating parsed dicts) — no file is re-parsed.
    """
    merged_wfs: list[dict] = []
    offset = 0
    for key in sorted(output_parameters):
        params = output_parameters[key].get_dict()
        wfs = params.get("wannier_functions_output") or []
        if len(wfs) != params.get("number_wfs"):
            raise ValueError(
                f"A sub-block's wannier90 ``output_parameters`` lists {len(wfs)} "
                "final-state Wannier functions but the run declares "
                f"number_wfs = {params.get('number_wfs')}."
            )
        for wf in sorted(wfs, key=lambda wf: int(wf["wf_ids"])):
            entry = dict(wf)
            entry["wf_ids"] = offset + int(wf["wf_ids"])
            merged_wfs.append(entry)
        offset += len(wfs)
    return orm.Dict({"number_wfs": offset, "wannier_functions_output": merged_wfs})


@task.calcfunction
def merge_interpolated_bands(**interpolated_bands: orm.BandsData) -> orm.BandsData:
    """Concatenate per-group interpolated bands into one block-wide structure.

    ``interpolated_bands`` holds the per-group re-Wannierisation results,
    keyed so lexicographic order matches the group (= band) order (``b00``,
    ``b01``, ...). Every group interpolates along the same k-path, and the
    merged block Hamiltonian is block-diagonal in the groups, so its band
    structure at every k-point is exactly the union of the groups': the
    per-group bands are concatenated along the band axis in group order.
    This threads parsed outputs (concatenating parsed ``BandsData``) — no
    file is re-parsed.
    """
    ordered = [interpolated_bands[key] for key in sorted(interpolated_bands)]
    reference = ordered[0]
    kpoints = reference.get_kpoints()
    for bands in ordered[1:]:
        other = bands.get_kpoints()
        if other.shape != kpoints.shape or not np.allclose(other, kpoints):
            raise ValueError(
                "The per-group interpolated bands do not share one k-path; "
                "they cannot be merged into a single band structure."
            )
    merged = orm.BandsData()
    merged.set_kpointsdata(reference)
    merged.set_bands(
        np.concatenate([np.asarray(bands.get_bands(), dtype=float) for bands in ordered], axis=-1),
        units=reference.units,
    )
    return merged


def _subblock_w90_parameters(
    num_wann: int,
    mp_grid: list[int],
    wannier90_overrides: dict[str, Any] | None,
    parent_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Wannier90 parameters for re-Wannierising one split sub-block.

    The sub-block reads the split ``.amn`` / ``.mmn`` / ``.eig`` directly
    (``num_bands == num_wann``, no preprocessing, no disentanglement, no
    band exclusion — the split files already cover exactly the group's
    bands). The convergence settings are copied from ``parent_parameters``
    (the parent whole-block run's resolved set, carrying the protocol tier
    and any user overrides); explicit user ``.win`` keywords then
    propagate on top, minus the per-block counts and the disentanglement
    set. A parent carrying none of the convergence keywords raises:
    silently proceeding would hand the run wannier90's compiled-in
    defaults.
    """
    convergence = {
        key: parent_parameters[key] for key in _PARENT_CONVERGENCE_KEYS if key in parent_parameters
    }
    if not convergence:
        raise ValueError(
            "The parent Wannierization's resolved parameters carry none of the "
            f"expected convergence keywords {_PARENT_CONVERGENCE_KEYS}; refusing "
            "to fall back to wannier90's compiled-in defaults."
        )
    dropped = (*_DIS_KEYS, "num_wann", "num_bands", "exclude_bands", "projections")
    params = dict(convergence)
    params.update(
        {key: value for key, value in (wannier90_overrides or {}).items() if key not in dropped}
    )
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

    The merged block-wide product files plus the merged parsed
    ``output_parameters`` — all describing the final (split) gauge. The
    per-sub-block wannier90 runs stay reachable through provenance (the
    merge tasks consume their ``retrieved`` folders and parsed Dicts as
    inputs); they are not re-exported as sockets. ``interpolated_bands``
    (populated only when ``interpolation_kpoints`` was given) is the
    per-group interpolated bands concatenated in group order — the merged
    block-diagonal Hamiltonian's own band structure.
    """

    u_file: orm.SinglefileData
    hr_file: orm.SinglefileData
    centres_file: orm.SinglefileData
    output_parameters: orm.Dict
    interpolated_bands: NotRequired[orm.BandsData]


@task.graph
def RewannierizeSplitBlocks(
    w90_code: orm.AbstractCode,
    structure: orm.StructureData,
    split_blocks: Annotated[dict, dynamic(orm.FolderData)],
    parent_parameters: orm.Dict,
    group_sizes: list[int],
    kpoints: orm.KpointsData,
    mp_grid: list[int],
    wannier90_overrides: dict[str, Any] | None = None,
    wannier90_options: dict[str, Any] | None = None,
    interpolation_kpoints: orm.KpointsData | None = None,
) -> RewannierizeSplitOutputs:
    """Re-Wannierise each split sub-block and merge the products.

    The keys of the ``SplitCalculation``'s dynamic ``blocks`` namespace only
    exist once it has run, so the whole namespace is passed into this nested
    graph; when this body executes ``split_blocks`` is a resolved
    ``{"block_0": FolderData, ...}`` dict and the per-group fan-out is a
    native ``for`` loop. Each sub-block runs a preprocessing-free
    ``Wannier90Calculation`` on the split ``.amn``/``.mmn``/``.eig``
    (``local_input_folder``), its convergence settings copied from
    ``parent_parameters`` (the whole-block run's resolved wannier90
    parameters) with the caller's overrides on top; then the ``_u.mat``
    / ``_hr.dat`` / ``_centres.xyz`` products are merged block-diagonally
    and the parsed per-group ``output_parameters`` are concatenated in
    group order.

    When ``interpolation_kpoints`` (a labelled explicit-path
    ``KpointsData``) is given, every sub-block run also sets
    ``bands_plot = True`` and takes the path as its ``bands_kpoints``, so
    each interpolates its own group's bands along it; the per-group
    results are concatenated into the block-wide ``interpolated_bands``
    output (:func:`merge_interpolated_bands`).
    """
    require_path_labels(interpolation_kpoints, "interpolation_kpoints")
    # Deferred bodies receive the resolved ``orm.Dict`` node; eager builds
    # hand the graph input over as a plain mapping already.
    parent_w90_parameters = (
        parent_parameters.get_dict()
        if hasattr(parent_parameters, "get_dict")
        else dict(parent_parameters)
    )
    subblock_retrieved: dict[str, Any] = {}
    subblock_parameters: dict[str, Any] = {}
    subblock_bands: dict[str, Any] = {}
    for i, num_wann in enumerate(group_sizes):
        parameters = _subblock_w90_parameters(
            int(num_wann), mp_grid, wannier90_overrides, parent_w90_parameters
        )
        # Wannier band interpolation: wannier90 interpolates only under
        # ``bands_plot``, and the calculation validator requires the path
        # alongside it, so the pair travels together.
        path_inputs: dict[str, Any] = {}
        if interpolation_kpoints is not None:
            parameters["bands_plot"] = True
            path_inputs["bands_kpoints"] = interpolation_kpoints
        rewannierized = Wannier90CalcStep(
            code=w90_code,
            structure=structure,
            parameters=parameters,
            kpoints=kpoints,
            local_input_folder=split_blocks[f"block_{i}"],
            **path_inputs,
            metadata={
                "call_link_label": f"wannier90_split_block_{i}",
                "options": _plain_options(wannier90_options),
            },
        )
        subblock_retrieved[f"b{i:02d}"] = rewannierized["retrieved"]
        subblock_parameters[f"b{i:02d}"] = rewannierized["output_parameters"]
        if interpolation_kpoints is not None:
            subblock_bands[f"b{i:02d}"] = rewannierized["interpolated_bands"]

    merged = merge_split_block_products(
        **subblock_retrieved,
        metadata={"call_link_label": "merge_split_block_products"},
    )
    merged_parameters = merge_wannier_output_parameters(
        **subblock_parameters,
        metadata={"call_link_label": "merge_wannier_output_parameters"},
    )

    outputs = RewannierizeSplitOutputs(
        u_file=merged["u_file"],
        hr_file=merged["hr_file"],
        centres_file=merged["centres_file"],
        output_parameters=merged_parameters.result,
    )
    if interpolation_kpoints is not None:
        merged_bands = merge_interpolated_bands(
            **subblock_bands,
            metadata={"call_link_label": "merge_interpolated_bands"},
        )
        outputs["interpolated_bands"] = merged_bands.result
    return outputs


# ----------------------------------------------------------------------
# Per-block graph (nested, deferred: receives the resolved groups)
# ----------------------------------------------------------------------


@task.graph
def WannierizeAndSplitBlock(
    codes: SplitBlockCodes,
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
    interpolation_kpoints: orm.KpointsData | None = None,
    external_projectors_path: str | None = None,
    external_projectors: dict[str, Any] | None = None,
) -> WannierizeBlockOutputs:
    """Wannierise one block, splitting it into detected groups when needed.

    Called as a nested graph task with ``groups`` wired from
    :func:`detect_band_groups`, so by the time this body runs the groups are
    concrete and the split-vs-plain decision is an ordinary ``if``:

    * one detected group overlapping the block — the block is already
      isolated; only the plain :func:`WannierizeBlock` runs and the product
      trio carries its gauge;
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

    The external projector inputs (required together by an
    ``atomic_projectors_external`` block) feed only the whole-block
    Wannierisation: the split chain regenerates ``.mmn`` at most (its cubic
    pw2wannier90 rerun writes no ``.amn``) and the per-group re-runs are
    preprocessing-free, so neither reads the projectors again.

    ``interpolation_kpoints`` (a labelled explicit-path ``KpointsData``)
    reaches both the whole-block Wannierisation and — when the block splits
    — every per-group re-run, so each interpolates its band structure along
    it. The entry's ``interpolated_bands`` output is always the final
    gauge's: the whole-block run's parse when the block stays whole, the
    per-group results concatenated in group order when it splits. A split
    block's pre-split interpolated bands are not re-exported here (the
    entry contract carries final-gauge fields only); they remain the
    ``interpolated_bands`` output of the nested ``wannierize_whole_block``
    graph, reachable through provenance like the other whole-block
    artifacts.
    """
    overrides = overrides or {}

    whole = WannierizeBlock(
        codes={
            "pw": codes["pw"],
            "pw2wannier90": codes["pw2wannier90"],
            "wannier90": codes["wannier90"],
        },
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
        interpolation_kpoints=interpolation_kpoints,
        external_projectors_path=external_projectors_path,
        external_projectors=external_projectors,
        metadata={"call_link_label": "wannierize_whole_block"},
    )

    # Split mode rejects every disentangled block, so no block in this
    # sequence reads extra bands and the Wannier-function indices are band
    # indices — which is what the detected band groups are counted in.
    block_bands = get_wannier_indices(block)
    local_groups = restrict_groups_to_block(list(groups), block_bands)
    if len(local_groups) <= 1:
        # The whole-block gauge is final here, so its parsed
        # ``output_parameters`` is the entry's final-gauge Dict. The
        # folder fields stay unpopulated (consumers read ``None`` at
        # runtime): whether a block splits is a runtime question, and
        # their populated-ness stays uniform across the split route.
        block_outputs = WannierizeBlockOutputs(
            u_file=whole["u_file"],
            hr_file=whole["hr_file"],
            centres_file=whole["centres_file"],
            nnkp_file=whole["nnkp_file"],
            output_parameters=whole["output_parameters"],
        )
        if interpolation_kpoints is not None:
            block_outputs["interpolated_bands"] = whole["interpolated_bands"]
        return block_outputs

    wann_groups = [
        [int(index) for index in group]
        for group in groups_to_wannier_indices(local_groups, block_bands)
    ]

    win_file = extract_win_file(retrieved=whole["retrieved"]).result

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
    # The split also emits per-block ``win_files``; they are deliberately
    # unconsumed — WannierIO.jl substitutes its own convergence values in
    # them, so the parent run's resolved parameters are the trustworthy
    # source for the sub-block settings.
    rewannierized = RewannierizeSplitBlocks(
        w90_code=codes["wannier90"],
        structure=structure,
        split_blocks=split["blocks"],
        parent_parameters=whole["wannier90_parameters"],
        group_sizes=[len(group) for group in wann_groups],
        kpoints=kpoints,
        mp_grid=mp_grid,
        wannier90_overrides=overrides.get("wannier90"),
        wannier90_options=wannier90_options,
        interpolation_kpoints=interpolation_kpoints,
        metadata={"call_link_label": "rewannierize_split_blocks"},
    )

    block_outputs = WannierizeBlockOutputs(
        u_file=rewannierized["u_file"],
        hr_file=rewannierized["hr_file"],
        centres_file=rewannierized["centres_file"],
        nnkp_file=whole["nnkp_file"],
        output_parameters=rewannierized["output_parameters"],
    )
    if interpolation_kpoints is not None:
        block_outputs["interpolated_bands"] = rewannierized["interpolated_bands"]
    return block_outputs
