"""Koopmans DFPT workflow (kcw.x): wann2kc → screen → ham.

The three steps are backed by the CalcJobs in
``aiida_koopmans.calculations.kcw`` (one kcw.x binary, three
``CONTROL.calculation`` modes).

Two graphs are exposed:

* :func:`RunDFPT` -- the kcw.x chain proper. It *consumes*
  wannierization outputs (the shared nscf scratch plus the label-keyed
  per-block outputs namespace, picked apart by the caller's band-ordered
  manifold label lists and merged per manifold) and runs
  wann2kcw → screen → ham. When ``alpha_guess`` is provided the screen
  step is skipped and the guess is fed straight to ham.
* :func:`SinglepointDFPTWorkflow` -- the end-to-end workflow: one shared
  scf + nscf, one
  :func:`~aiida_koopmans.workgraphs.block_wannierize.WannierizeBlocks` per
  spin channel (fed the shared nscf scratch, so it skips its internal
  scf + nscf), then :func:`RunDFPT`.

Multi-block manifolds are supported: each projection block is Wannierised
independently and the per-block products are merged per manifold
(block-diagonal u / hr, concatenated centres, identity-extended u_dis --
see :mod:`aiida_koopmans.workgraphs.utils.wannier_merge`) before kcw.x consumes them.

Spin handling (``SinglepointDFPTWorkflow``'s ``spin`` input, an
``aiida_quantumespresso`` ``SpinType``):

* ``NONE`` — kcw.x requires an nspin=2 parent scratch even for
  closed-shell systems (the DFPT perturbations are spin-dependent), so
  the PW runs are forced to ``nspin = 2`` + ``tot_magnetization = 0`` and
  pw2wannier90 to ``spin_component = 'up'``. One kcw chain on the up
  channel.
* ``COLLINEAR`` — per-channel wannierization (wannier90 ``spin``,
  pw2wannier90 ``spin_component``) and a kcw chain per channel
  (``CONTROL.spin_component`` 1 / 2), with each channel's results under
  its key in the ``channels`` output namespace.
* ``NON_COLLINEAR`` / ``SPIN_ORBIT`` — spinor scratch (``noncolin``, plus
  ``lspinorb`` for SOC), ``spinors = .true.`` wannierization with doubled
  ``num_wann``, one kcw chain. QE reference:
  ``KCW/examples/example05.1`` nspin4 variants.

Screening comes in three mutually exclusive flavours per channel (see
:func:`RunDFPT`): a caller ``alpha_guess`` (no screen step at all),
workflow-level orbital grouping (``group_orbitals_tol`` set: cluster the
Wannier functions by their spreads — the unified band-ordered ``spreads``
output of ``WannierizeBlocks``, not the raw retrieved folders — and run
one ``SCREEN.i_orb`` screen calculation per group representative, in
parallel), or the default single screen calculation solving every orbital.

Current limitations:

* No coarse-grid pre-screening (``dfpt_coarse_grid``) and no
  unfold-and-interpolate postprocessing.
"""

# No ``from __future__ import annotations`` in this module: stringified
# annotations hide ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the dispatcher reads off the Codes
# TypedDicts.
import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import (
    Annotated,
    Any,
    NotRequired,
    TypedDict,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from aiida import orm
from aiida_quantumespresso.common.types import SpinType
from aiida_workgraph import dynamic, task
from aiida_workgraph.socket_spec import SocketMeta
from node_graph import ref

from aiida_koopmans.calculations.kcw import (
    KcwHamCalculation,
    KcwScreenCalculation,
    Wann2kcCalculation,
)
from aiida_koopmans.parallelization import (
    ParallelizationDict,
    merge_parallelization_into_inputs,
    validate_parallelization,
)
from aiida_koopmans.projections import (
    ProjectionBlock,
)
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.variational_orbitals import VariationalOrbital, map_key_for
from aiida_koopmans.workgraphs.block_wannierize import (
    WannierizeBlockOutputs,
    WannierizeBlocks,
    WannierizeBlocksCodes,
    WannierizeOverrides,
)
from aiida_koopmans.workgraphs.ph import DielectricTask
from aiida_koopmans.workgraphs.pw import PwCode, PwOutputs, RunScfNscf
from aiida_koopmans.workgraphs.utils.wannier_merge import (
    extend_wannier_u_dis_file_content,
    merge_wannier_centres_file_contents,
    merge_wannier_hr_file_contents,
    merge_wannier_u_file_contents,
    parse_wannier_u_file_shape,
)
from aiida_koopmans.workgraphs.variational_orbitals import (
    assign_orbital_groups,
    expand_alphas_by_group,
    spreads_metric_row,
)
from aiida_koopmans.workgraphs.wannier90 import (
    ProjwfcCode,
    ProjwfcOutputs,
    Pw2Wannier90Code,
    Wannier90Code,
    projected_dos_supported,
)


class DfptCodes(TypedDict):
    """Codes for :func:`SinglepointDFPTWorkflow`."""

    pw: PwCode
    pw2wannier90: Pw2Wannier90Code
    wannier90: Wannier90Code
    kcw: Annotated[
        orm.AbstractCode,
        SocketMeta(
            help="Needed to compute screening parameters and construct Hamiltonians "
            "in reciprocal space."
        ),
    ]
    ph: NotRequired[
        Annotated[
            orm.AbstractCode,
            SocketMeta(help="Needed if the dielectric constant is to be computed automatically."),
        ]
    ]
    projwfc: NotRequired[ProjwfcCode]


# kcw.x reads ``<seedname>_u.mat`` / ``<seedname>_emp_u.mat`` (etc.) from its
# working directory. The wannier90 CalcJob writes its products with the
# ``aiida`` seedname, so keeping the same seedname means the occupied-manifold
# files stage under their retrieved names unchanged.
SEEDNAME = "aiida"

# Wannier90 products each manifold must provide (suffixes appended to the
# seedname). ``_u_dis.mat`` is optional: it only exists when the manifold was
# disentangled (empty manifold with num_bands > num_wann).
_REQUIRED_SUFFIXES = ("_u.mat", "_hr.dat", "_centres.xyz")
_OPTIONAL_SUFFIXES = ("_u_dis.mat",)


Wann2kcStep = task(Wann2kcCalculation)
KcwScreenStep = task(KcwScreenCalculation)
KcwHamStep = task(KcwHamCalculation)


def normalize_alpha_guess(
    raw_guess: float | list,
    n_orbitals: int,
    spin_channel: SpinChannel = SpinChannel.NONE,
) -> list[float]:
    """Flatten a user ``alpha_guess`` into one alpha per orbital.

    Accepts the three shapes the input file allows: a single float (uniform
    guess), a flat list, or the nested per-spin list (``spin_channel.axis``
    selects the channel: up/none/spinor take the first entry, down the
    second).
    """
    if isinstance(raw_guess, float):
        return [raw_guess] * n_orbitals
    if raw_guess and isinstance(raw_guess[0], list):
        return [float(a) for a in raw_guess[spin_channel.axis]]
    return [float(a) for a in raw_guess]


@task
def single_orbital_alpha(alphas: list) -> float:
    """Extract the one alpha an ``SCREEN.i_orb`` screen run computed.

    A single-orbital kcw.x run prints exactly one ``iwann ... alpha ...``
    line, so its ``alphas`` output is a one-entry list; anything else means
    the run did not honour ``i_orb`` and must not be broadcast to a group.
    """
    if len(alphas) != 1:
        raise ValueError(
            f"An ``i_orb`` screen run must yield exactly one alpha, got {len(alphas)}."
        )
    return float(alphas[0])


@task
def alphas_in_orbital_order(
    *,
    orbitals: list[VariationalOrbital],
    filled_alphas: dict | None = None,
    empty_alphas: dict | None = None,
) -> list:
    """Flatten per-orbital alpha dicts into kcw.x orbital order.

    ``filled_alphas`` / ``empty_alphas`` are the broadcast
    ``{map_key: alpha}`` dicts of :func:`expand_alphas_by_group` — one
    entry per orbital. The ham step's ``alphas`` input (and kcw.x's
    ``i_orb`` numbering) wants a flat list, occupied orbitals first, then
    empty, each in ascending index order.
    """
    filled_alphas = filled_alphas or {}
    empty_alphas = empty_alphas or {}
    ordered: list[float] = []
    for filled, source in ((True, filled_alphas), (False, empty_alphas)):
        subset = sorted((o for o in orbitals if o["filled"] == filled), key=lambda o: o["index"])
        for o in subset:
            key = map_key_for(o)
            if key not in source:
                raise ValueError(
                    f"No alpha for orbital {key} — the group broadcast upstream did not cover it."
                )
            ordered.append(float(source[key]))
    return ordered


class GroupedKcwScreeningOutputs(TypedDict):
    """Outputs of :func:`GroupedKcwScreening`.

    ``alphas`` is the full per-orbital screening-parameter list (occupied
    then empty, group representatives broadcast onto their members), ready
    for the ham step.
    """

    alphas: list


@task.graph
def GroupedKcwScreening(
    *,
    kcw_code: orm.AbstractCode,
    control: dict,
    wannier: dict,
    screen_namelist: dict,
    parent_folder: orm.RemoteData,
    wannier_files: orm.FolderData,
    orbitals: list[VariationalOrbital],
    parallelization: ParallelizationDict | None = None,
) -> GroupedKcwScreeningOutputs:
    """Per-group screening fan-out: one ``SCREEN.i_orb`` run per representative.

    A separate ``@task.graph`` (rather than inline in :func:`RunDFPT`)
    because the fan-out cardinality depends on ``orbitals`` — a *runtime*
    output of :func:`assign_orbital_groups` (it clusters the wannier90
    spreads). Inside this deferred body ``orbitals`` is concrete, so the
    scatter is a native ``for`` loop and the gather a plain dict of
    per-representative alpha sockets (same shape as
    ``ComputeOrbitalScreeningParameters`` on the kcp.x route).

    Each representative runs a screen calculation with ``SCREEN.i_orb``
    set to its (1-based, occupied-then-empty) orbital index off the shared
    wann2kcw ``parent_folder``; the runs are independent and execute in
    parallel. ``check_spread`` is forced off: kcw.x's internal self-Hartree
    grouping is meaningless for a single-orbital solve, and the
    workflow-level grouping has already decided who shares an alpha.

    ``control`` / ``wannier`` / ``screen_namelist`` are the namelist dicts
    :func:`RunDFPT` assembled (``screen_namelist`` without ``i_orb`` /
    ``check_spread``, which this graph owns).
    """
    filled_alphas: dict[str, Any] = {}
    empty_alphas: dict[str, Any] = {}
    for orbital in orbitals:
        if not orbital["representative"]:
            continue
        key = map_key_for(orbital)
        namelist = {
            # Explicitly unwrap the (possibly TaggedValue-proxied) namelist by
            # iterating its ``.items()`` into a plain dict before extending it,
            # rather than relying on ``dict(proxy)`` to coerce the proxy.
            **dict((screen_namelist or {}).items()),
            "i_orb": int(orbital["index"]),
            "check_spread": False,
        }
        screen_inputs: dict[str, Any] = {
            "code": kcw_code,
            "parameters": {"CONTROL": control, "WANNIER": wannier, "SCREEN": namelist},
            "parent_folder": parent_folder,
            "wannier_files": wannier_files,
            "metadata": {"call_link_label": f"screen_{key}"},
        }
        merge_parallelization_into_inputs(screen_inputs, parallelization, "kcw")
        screen = KcwScreenStep(**screen_inputs)
        alpha = single_orbital_alpha(
            alphas=screen["alphas"],
            metadata={"call_link_label": f"alpha_{key}"},
        )
        if orbital["filled"]:
            filled_alphas[key] = alpha.result
        else:
            empty_alphas[key] = alpha.result

    expanded = expand_alphas_by_group(
        filled_rep_alphas=filled_alphas or None,
        empty_rep_alphas=empty_alphas or None,
        orbitals=orbitals,
        metadata={"call_link_label": "expand_alphas_by_group"},
    )
    ordered = alphas_in_orbital_order(
        orbitals=orbitals,
        filled_alphas=expanded["filled_alphas"],
        empty_alphas=expanded["empty_alphas"],
        metadata={"call_link_label": "alphas_in_orbital_order"},
    )
    return GroupedKcwScreeningOutputs(alphas=ordered.result)


@task
def alphas_from_guess(alpha_guess: list) -> list:
    """Materialise a caller-provided screening-parameter guess.

    Runs as a named ``@task`` (rather than passing the raw list around) so
    the guess becomes a provenance node and a socket that both the ham step
    and the graph outputs can consume (raw Python values are not valid graph
    return payloads).
    """
    return list(alpha_guess)


@task
def emit_namespace_dict_field(value: dict) -> dict:
    """Materialise one of a namespace's plain-``dict``-typed fields as a task socket.

    Same fix as :func:`alphas_from_guess`, for a namespace parameter's
    ``dict``-typed field (e.g. ``output_parameters`` or
    ``output_atomic_occupations`` on :class:`~aiida_koopmans.workgraphs.pw.PwOutputs`
    / :class:`~aiida_koopmans.workgraphs.wannier90.ProjwfcOutputs`) rather
    than a top-level ``list``: at this graph's own materialisation time it
    arrives fully deserialized to a plain dict (every ``dict``-typed field
    in this codebase does), so echoing it straight into a graph output
    fails the "raw Python value" check. The namespace's other fields
    (``RemoteData``, ``BandsData``, ...) stay socket-linked through
    materialisation and need no rewrap.
    """
    return dict(value)


class ChannelResults(TypedDict, total=False):
    """Results of one kcw.x chain (one spin channel).

    * ``alphas`` -- the screening parameters fed to the ham step (computed by
      screen, or the caller's guess when screening was skipped).
    * ``screen_parameters`` -- screen-step scalars (:class:`KcwScreenParameters`;
      absent when screening was skipped via ``alpha_guess`` or fanned out
      into per-representative ``i_orb`` runs via ``group_orbitals_tol``).
    * ``ham_parameters`` -- ham-step scalars (:class:`KcwHamParameters`),
      including the KS / KI eigenvalues on the k-grid.
    * ``bands`` -- interpolated Koopmans band structure (present only when a
      band path was supplied).
    * ``wannierize_bands`` -- the pw.x quality-check DFT reference bands
      along the same path, off the shared ground state (present only when a
      band path was supplied; see :func:`RunDFPT`'s ``wannierize_bands``).
    * ``projwfc`` -- the projected DOS computed off that quality-check run's
      scratch (present when it ran and a projwfc code was configured; see
      :func:`RunDFPT`'s ``projwfc``).
    * ``wann2kc_remote_folder`` -- the wann2kcw scratch, for chaining further
      kcw.x runs off the same conversion.

    ``screen_parameters`` / ``ham_parameters`` carry the key sets documented
    by :class:`KcwScreenParameters` / :class:`KcwHamParameters`; they are
    annotated as plain ``dict`` here because a TypedDict annotation on a
    ``@task.graph`` output is read as a nested namespace socket rather than a
    leaf ``orm.Dict``. ``alphas`` is annotated as a plain ``list`` so callers
    receive the deserialized python value at the graph boundary.
    """

    alphas: list
    screen_parameters: dict
    ham_parameters: dict
    bands: orm.BandsData
    wannierize_bands: PwOutputs
    projwfc: ProjwfcOutputs
    wann2kc_remote_folder: orm.RemoteData


class KoopmansDFPTOutputs(TypedDict):
    """Outputs of :func:`SinglepointDFPTWorkflow`.

    ``channels`` is a dynamic namespace keyed by spin channel
    (:class:`SpinChannel` values as strings); each entry is the
    :class:`ChannelResults` of that channel's kcw.x chain. Unpolarized and
    spinor runs populate the single key ``"none"``; collinear runs populate
    ``"up"`` and ``"down"``.
    """

    channels: Annotated[dict, dynamic(ChannelResults)]


class ManifoldBlocks(TypedDict):
    """Per-spin-channel manifold description consumed by :func:`SinglepointDFPTWorkflow`.

    * ``occ`` -- the occupied projection blocks in band order (at least one;
      several when the occupied manifold spans multiple projection blocks).
    * ``emp`` -- the empty projection blocks, when the channel has any.
    * ``alpha_guess`` -- per-orbital screening-parameter guess for this
      channel; when given the channel's screen step is skipped.

    A manifold Wannierised as several blocks has its per-block Wannier
    products merged back into one file set by :func:`prepare_kcw_wannier_files`.
    """

    occ: list[ProjectionBlock]
    emp: NotRequired[list[ProjectionBlock]]
    alpha_guess: NotRequired[list[float] | None]


def _read_block_files(folder: orm.FolderData, manifold: str) -> dict[str, bytes]:
    """Read one block's Wannier90 products out of its ``retrieved`` folder.

    Returns the file contents keyed by suffix (``_u.mat`` etc.).
    ``_u_dis.mat`` is included when present; the required products raise
    when absent.
    """
    names = set(folder.base.repository.list_object_names())
    contents: dict[str, bytes] = {}
    for suffix in _REQUIRED_SUFFIXES + _OPTIONAL_SUFFIXES:
        src_name = f"{SEEDNAME}{suffix}"
        if src_name not in names:
            if suffix in _OPTIONAL_SUFFIXES:
                continue
            raise ValueError(
                f"``{src_name}`` is missing from a {manifold}-manifold wannier90 "
                "retrieved folder. The wannier90 runs feeding a DFPT chain must set "
                "``write_u_matrices = True`` and ``write_xyz = True``."
            )
        contents[suffix] = folder.base.repository.get_object_content(src_name, mode="rb")
    return contents


def _manifold_u_dis(blocks: list[dict[str, bytes]], nbnd: int | None, manifold: str) -> None:
    """Attach the merged manifold's ``_u_dis.mat`` to ``blocks[-1]``, in place.

    Only the last block of a manifold is disentangled (the band layout
    :func:`aiida_koopmans.projections._manifold_projection_blocks` fixes).
    When the manifold has more
    bands than Wannier functions its ``u_dis`` is required: a single-block
    manifold stages the file unchanged, a merged one extends it with an
    identity for the preceding blocks
    (:func:`~aiida_koopmans.workgraphs.utils.wannier_merge.extend_wannier_u_dis_file_content`).
    """
    if nbnd is None:
        return
    num_wann = sum(parse_wannier_u_file_shape(b["_u.mat"].decode())[1] for b in blocks)
    if nbnd <= num_wann:
        return
    if "_u_dis.mat" not in blocks[-1]:
        raise ValueError(
            f"The {manifold} manifold is disentangled ({nbnd} bands for {num_wann} "
            "Wannier functions) but its last block's wannier90 retrieved folder holds "
            f"no ``{SEEDNAME}_u_dis.mat``."
        )
    if len(blocks) > 1:
        blocks[-1]["_u_dis.mat"] = extend_wannier_u_dis_file_content(
            blocks[-1]["_u_dis.mat"].decode(), nbnd=nbnd, nwann=num_wann
        ).encode()


def _merged_manifold_files(
    blocks: list[dict[str, bytes]], nbnd: int | None, manifold: str
) -> dict[str, bytes]:
    """Combine per-block product files into one manifold-wide file set.

    A single block passes through byte-identical (plus its optional
    ``_u_dis.mat``); several blocks are merged block-diagonally (u / hr),
    by concatenation (centres), and by identity extension of the last
    block's ``_u_dis.mat`` when ``nbnd`` exceeds the manifold's Wannier
    count.
    """
    _manifold_u_dis(blocks, nbnd, manifold)
    if len(blocks) == 1:
        return blocks[0]
    merged = {
        "_hr.dat": merge_wannier_hr_file_contents([b["_hr.dat"].decode() for b in blocks]).encode(),
        "_u.mat": merge_wannier_u_file_contents([b["_u.mat"].decode() for b in blocks]).encode(),
        "_centres.xyz": merge_wannier_centres_file_contents(
            [b["_centres.xyz"].decode() for b in blocks]
        ).encode(),
    }
    if "_u_dis.mat" in blocks[-1]:
        merged["_u_dis.mat"] = blocks[-1]["_u_dis.mat"]
    return merged


@task.calcfunction(outputs=["wannier_files"])
def prepare_kcw_wannier_files(nbnd_emp: int | None = None, **retrieved: orm.FolderData) -> dict:
    """Assemble the ``wannier_files`` folder the kcw.x CalcJobs stage.

    Collects the Wannier90 products (``aiida_u.mat`` / ``aiida_hr.dat`` /
    ``aiida_centres.xyz``, requiring the wannier90 runs to have set
    ``write_u_matrices`` and ``write_xyz``) out of the per-block
    ``retrieved`` folders, merges multi-block manifolds into one file set,
    and renames the empty-manifold files to kcw.x's hard-coded
    ``<seedname>_emp_*`` convention.

    Args:
        nbnd_emp: total number of empty bands (``nbnd - nocc``). Required to
            stage a merged ``aiida_emp_u_dis.mat`` when the empty manifold is
            disentangled; ignored otherwise.
        retrieved: the per-block wannier90 ``retrieved`` folders, keyed
            ``occ_*`` / ``emp_*`` with the *lexicographic* key order matching
            the band order within each manifold (e.g. ``occ_b00``,
            ``occ_b01``, ...).
    """
    occ_folders = [retrieved[key] for key in sorted(retrieved) if key.startswith("occ")]
    emp_folders = [retrieved[key] for key in sorted(retrieved) if key.startswith("emp")]
    if not occ_folders:
        raise ValueError(
            "prepare_kcw_wannier_files needs at least one occupied-manifold retrieved "
            "folder (an ``occ_*``-keyed input)."
        )

    merged = orm.FolderData()
    manifolds: list[tuple[str, str, list[orm.FolderData], int | None]] = [
        ("", "occupied", occ_folders, None)
    ]
    if emp_folders:
        nbnd = None if nbnd_emp is None else int(nbnd_emp)
        manifolds.append(("_emp", "empty", emp_folders, nbnd))
    for rename, manifold, folders, nbnd in manifolds:
        blocks = [_read_block_files(folder, manifold) for folder in folders]
        for suffix, content in _merged_manifold_files(blocks, nbnd, manifold).items():
            merged.base.repository.put_object_from_bytes(content, f"{SEEDNAME}{rename}{suffix}")

    return {"wannier_files": merged}


@task.graph
def RunDFPT(
    kcw_code: orm.AbstractCode,
    nscf_remote_folder: orm.RemoteData,
    block_wannier: Annotated[dict, dynamic(WannierizeBlockOutputs)],
    occ_labels: list,
    num_wann_occ: int,
    num_wann_emp: int,
    kgrid: list[int],
    emp_labels: list | None = None,
    nbnd_emp: int | None = None,
    spreads: list | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    wannierize_bands: PwOutputs | None = None,
    projwfc: ProjwfcOutputs | None = None,
    eps_inf: float | None = None,
    alpha_guess: list[float] | None = None,
    group_orbitals_tol: float | None = None,
    has_disentangle: bool = False,
    l_vcut: bool | None = None,
    spin_component: int = 1,
    check_spread: bool = True,
    parallelization: ParallelizationDict | None = None,
) -> ChannelResults:
    """Run the kcw.x chain off provided wannierization outputs.

    Args:
        kcw_code: the kcw.x code; it runs every step.
        nscf_remote_folder: scratch of the pw.x **nscf** run the Wannier
            functions were built on (kcw.x re-reads its wavefunctions). Must
            be an ``nspin = 2`` run even for closed-shell systems -- the DFPT
            perturbations are spin-dependent; the kcw chain reads the up
            channel (``CONTROL.spin_component = 1``).
        block_wannier: the per-block wannierization outputs keyed by block
            label — pass ``WannierizeBlocks``' ``blocks`` output namespace
            *wholesale*. A nested sub-graph's dynamic namespace has no
            per-key sockets until it runs, so callers must not subscript
            into it; this graph picks blocks out by label in its own
            deferred body (the ``FoldToSupercell`` pattern). Each entry's
            ``retrieved`` folder must hold ``aiida_u.mat`` /
            ``aiida_hr.dat`` / ``aiida_centres.xyz``.
        occ_labels: the occupied-manifold block labels, in band order.
            Manifold membership and band order are the caller's structural
            knowledge (its own block lists); the file merge keys each
            manifold's blocks ``b{i:02d}`` by list position, so
            lexicographic order matches band order — the convention
            :func:`prepare_kcw_wannier_files` merges by. Multi-block
            manifolds are merged there into one file set.
        num_wann_occ / num_wann_emp: *total* Wannier function counts per
            manifold (``num_wann_emp = 0`` for an occupied-only run).
        kgrid: the Monkhorst-Pack grid of the nscf, for ``CONTROL.mp1-3``.
        emp_labels: the empty-manifold block labels, in band order (omit
            for an occupied-only run).
        nbnd_emp: total number of empty bands (``nbnd - nocc``); needed to
            extend the ``u_dis`` matrix when a merged empty manifold is
            disentangled.
        spreads: the channel's unified per-orbital Wannier spreads (Å²,
            band-ordered occupied-then-empty — the ``spreads`` output of
            ``WannierizeBlocks``). Consumed only by the
            ``group_orbitals_tol`` path (the spread clustering depends on
            the spreads, not on the raw retrieved files) and required when
            it is active; the count is checked against
            ``num_wann_occ + num_wann_emp`` at runtime.
        bands_kpoints: explicit k-path; when given, the ham step interpolates
            the Koopmans Hamiltonian along it (``HAM.do_bands``).
        wannierize_bands: the pw.x quality-check DFT reference bands along
            the same path, forwarded whole from the caller's
            :func:`~aiida_koopmans.workgraphs.block_wannierize.WannierizeBlocks`
            call (its ``bands`` output) into this graph's own
            ``wannierize_bands`` output; this graph runs no bands step of
            its own.
        projwfc: the projected DOS off that quality-check run's scratch,
            forwarded the same way from ``WannierizeBlocks``' ``projwfc``
            output into this graph's own ``projwfc`` output.
        eps_inf: macroscopic dielectric constant for the screen step's
            long-range corrections.
        alpha_guess: when given, skip the screen step and feed these alphas
            straight to ham (takes precedence over ``group_orbitals_tol`` —
            no screening runs at all).
        group_orbitals_tol: when set, workflow-level orbital grouping:
            cluster the Wannier functions by their wannier90 spread (the
            ``spreads`` input; complete linkage within this tolerance,
            never across the occupied/empty boundary —
            :func:`assign_orbital_groups`), run
            one ``SCREEN.i_orb`` screen calculation per group representative
            in parallel (:func:`GroupedKcwScreening`), and broadcast each
            representative's alpha onto its group before the ham step.
            ``None`` (default) keeps the single all-orbital screen
            calculation.
        has_disentangle: whether the empty manifold was disentangled
            (``num_bands != num_wann``).
        l_vcut: Gygi-Baldereschi long-range cutoff (the ``gb_correction``
            workflow keyword); None means the periodic-system default (on).
        spin_component: which collinear spin channel kcw.x reads (1 = up,
            2 = down). Spin-unpolarized runs use the default 1 (the nspin=2
            scratch's channels are identical); a spin-polarized workflow
            calls this task once per channel. Ignored by kcw.x for
            noncollinear scratches.
        check_spread: kcw.x's ``SCREEN.check_spread`` — despite the name it
            groups orbitals *inside a single kcw.x run* by their self-Hartree
            energy (tolerance hardcoded to 1e-4 in kcw.x) and solves the
            linear-response problem once per group. Distinct from workflow-
            level orbital grouping (``group_orbitals_tol``). Only affects
            the single all-orbital screen step; the grouped
            per-representative ``i_orb`` runs force it off.
    """
    # ``bool()`` unwraps a possible wrapt proxy (a TaggedValue graph input)
    # to a plain bool before it lands in the stored ``control`` Dict.
    l_vcut = True if l_vcut is None else bool(l_vcut)
    control = {
        "kcw_iverbosity": 1,
        "kcw_at_ks": False,
        "read_unitary_matrix": True,
        "lrpa": False,
        "l_vcut": l_vcut,
        "spin_component": spin_component,
        "mp1": kgrid[0],
        "mp2": kgrid[1],
        "mp3": kgrid[2],
    }
    wannier = {
        "seedname": SEEDNAME,
        "check_ks": True,
        "num_wann_occ": num_wann_occ,
        "num_wann_emp": num_wann_emp,
        "have_empty": num_wann_emp > 0,
        "has_disentangle": has_disentangle,
    }

    prep_inputs: dict[str, Any] = {
        f"occ_b{i:02d}": block_wannier[str(label)]["retrieved"]
        for i, label in enumerate(occ_labels)
    }
    if emp_labels is not None:
        for i, label in enumerate(emp_labels):
            prep_inputs[f"emp_b{i:02d}"] = block_wannier[str(label)]["retrieved"]
        if nbnd_emp is not None:
            prep_inputs["nbnd_emp"] = nbnd_emp
    wannier_files = prepare_kcw_wannier_files(
        **prep_inputs,
        metadata={"call_link_label": "prepare_kcw_wannier_files"},
    )["wannier_files"]

    wann2kc_inputs: dict[str, Any] = {
        "code": kcw_code,
        "parameters": {"CONTROL": control, "WANNIER": wannier},
        "parent_folder": nscf_remote_folder,
        "wannier_files": wannier_files,
        "metadata": {"call_link_label": "wann2kc"},
    }
    merge_parallelization_into_inputs(wann2kc_inputs, parallelization, "kcw")
    wann2kc = Wann2kcStep(**wann2kc_inputs)

    outputs = ChannelResults(wann2kc_remote_folder=wann2kc["remote_folder"])

    screen_namelist: dict[str, Any] = {
        "tr2": 1.0e-18,
        "nmix": 4,
        "niter": 33,
    }
    if eps_inf is not None:
        screen_namelist["eps_inf"] = eps_inf

    if alpha_guess is not None:
        alphas = alphas_from_guess(
            alpha_guess=list(alpha_guess),
            metadata={"call_link_label": "alphas_from_guess"},
        ).result
    elif group_orbitals_tol is not None:
        # Workflow-level orbital grouping: cluster the Wannier functions by
        # their wannier90 spread (the unified ``spreads`` input), then screen
        # one representative per group with ``SCREEN.i_orb`` (embarrassingly
        # parallel) and broadcast the alphas. The fan-out cardinality depends
        # on the runtime clustering, hence the nested deferred graph.
        if spreads is None:
            raise ValueError(
                "group_orbitals_tol requires the channel's per-orbital wannier90 "
                "spreads (``spreads``, the unified WannierizeBlocks output): the "
                "spread clustering depends on them."
            )
        metric = spreads_metric_row(
            spreads=spreads,
            expected_count=int(num_wann_occ) + int(num_wann_emp),
            metadata={"call_link_label": "spreads_metric_row"},
        )
        orbitals = assign_orbital_groups(
            metric=metric.result,
            nelup=int(num_wann_occ),
            neldw=0,
            nbnd=int(num_wann_occ) + int(num_wann_emp),
            spin_polarized=False,
            tol=group_orbitals_tol,
            metadata={"call_link_label": "assign_orbital_groups"},
        )
        grouped = GroupedKcwScreening(
            kcw_code=kcw_code,
            control=control,
            wannier=wannier,
            screen_namelist=screen_namelist,
            parent_folder=wann2kc["remote_folder"],
            wannier_files=wannier_files,
            orbitals=orbitals.result,
            parallelization=parallelization,
            metadata={"call_link_label": "grouped_screen"},
        )
        alphas = grouped["alphas"]
    else:
        # ``bool()`` unwraps a possible wrapt proxy, as for ``l_vcut``.
        screen_namelist["check_spread"] = bool(check_spread)
        screen_inputs: dict[str, Any] = {
            "code": kcw_code,
            "parameters": {"CONTROL": control, "WANNIER": wannier, "SCREEN": screen_namelist},
            "parent_folder": wann2kc["remote_folder"],
            "wannier_files": wannier_files,
            "metadata": {"call_link_label": "screen"},
        }
        merge_parallelization_into_inputs(screen_inputs, parallelization, "kcw")
        screen = KcwScreenStep(**screen_inputs)
        alphas = screen["alphas"]
        outputs["screen_parameters"] = screen["output_parameters"]

    do_bands = bands_kpoints is not None
    ham_namelist = {
        "do_bands": do_bands,
        "use_ws_distance": True,
        "write_hr": True,
        "on_site_only": False,
    }
    ham_inputs: dict[str, Any] = {
        "code": kcw_code,
        "parameters": {"CONTROL": control, "WANNIER": wannier, "HAM": ham_namelist},
        "parent_folder": wann2kc["remote_folder"],
        "wannier_files": wannier_files,
        "alphas": alphas,
        "metadata": {"call_link_label": "ham"},
    }
    if do_bands:
        ham_inputs["kpoints"] = bands_kpoints
    # The kcw.x ham step takes no -npool (kcw_readin.f90 rejects pools for
    # calculation='ham'), only -pd; wann2kc and screen above take both.
    merge_parallelization_into_inputs(ham_inputs, parallelization, "kcw", pools=False)
    ham = KcwHamStep(**ham_inputs)

    outputs["alphas"] = alphas
    outputs["ham_parameters"] = ham["output_parameters"]
    _add_optional_band_outputs(outputs, ham, do_bands, wannierize_bands, projwfc)
    return outputs


def _dict_typed_field_names(typeddict_cls: type) -> frozenset[str]:
    """Return ``typeddict_cls``'s fields declared as plain ``dict``.

    Those are exactly the fields that, on a namespace of this type, arrive
    fully deserialized (not as a socket) at this graph's own
    materialisation time — see :func:`emit_namespace_dict_field` — so any of
    them present on a namespace *instance* need routing through it before
    the namespace can be re-exported as a graph output. Read off the
    TypedDict's own declared hints rather than named by hand, so a new
    ``dict``-typed field (this module already missed one once —
    ``output_atomic_occupations``, pw.x DFT+U) is covered automatically.
    """
    hints = get_type_hints(typeddict_cls, include_extras=True)
    names = set()
    for name, hint in hints.items():
        if get_origin(hint) is NotRequired:
            hint = get_args(hint)[0]
        if hint is dict:
            names.add(name)
    return frozenset(names)


def _reexported_namespace[NamespaceT: Mapping[str, Any]](
    namespace: NamespaceT, typeddict_cls: type, label: str
) -> NamespaceT:
    """Return ``namespace`` with its plain-``dict``-typed fields re-exported as sockets.

    ``typeddict_cls`` is the namespace's own declared shape (e.g.
    :class:`~aiida_koopmans.workgraphs.pw.PwOutputs`); see
    :func:`_dict_typed_field_names` / :func:`emit_namespace_dict_field`.
    Every other field of the namespace (``remote_folder``, ``output_band``,
    ...) stays socket-linked through this graph's own materialisation and
    passes through unchanged.
    """
    rebuilt = dict(namespace)
    for field in _dict_typed_field_names(typeddict_cls) & rebuilt.keys():
        rebuilt[field] = emit_namespace_dict_field(
            value=rebuilt[field],
            metadata={"call_link_label": f"emit_{label}_{field}"},
        ).result
    return cast("NamespaceT", rebuilt)


def _add_optional_band_outputs(
    outputs: ChannelResults,
    ham: Any,
    do_bands: bool,
    wannierize_bands: PwOutputs | None,
    projwfc: ProjwfcOutputs | None,
) -> None:
    """Populate ``ChannelResults``' band / quality-check outputs, in place.

    Each is present exactly when its own producing step ran: ``bands`` from
    the ham step's own interpolation (``do_bands``), ``wannierize_bands`` /
    ``projwfc`` forwarded from the caller's
    :func:`~aiida_koopmans.workgraphs.block_wannierize.WannierizeBlocks` call
    (:func:`_reexported_namespace` fixes up the fields that need it).
    """
    if do_bands:
        outputs["bands"] = ham["bands"]
    if wannierize_bands is not None:
        outputs["wannierize_bands"] = _reexported_namespace(
            wannierize_bands, PwOutputs, "wannierize_bands"
        )
    if projwfc is not None:
        outputs["projwfc"] = _reexported_namespace(projwfc, ProjwfcOutputs, "projwfc")


def _pw_spin_system_defaults(spin: SpinType) -> dict[str, Any]:
    """Return the SYSTEM-namelist keys a DFPT chain forces on the PW runs.

    * Unpolarized: kcw.x requires an nspin=2 scratch even for closed-shell
      systems (the DFPT perturbations are spin-dependent), so the PW runs
      carry ``nspin=2 + tot_magnetization=0``.
    * Collinear: nspin=2 without pinning the magnetization — the caller's
      overrides carry the physical ``tot_magnetization`` /
      ``starting_magnetization``.
    * Noncollinear / spin-orbit: spinor wavefunctions (``noncolin``), plus
      ``lspinorb`` for SOC, with a tiny ``starting_magnetization`` so QE runs
      the spin-accounting (``domag = .TRUE.``) branch — kcw.x's screening
      drops the magnetization channels from the xc kernel otherwise and
      diverges from the collinear result (QE reference:
      KCW/examples/example05.1, ``nspin4_noSOC_MAG`` variant).
    """
    if spin == SpinType.COLLINEAR:
        return {"nspin": 2}
    if spin == SpinType.NON_COLLINEAR:
        return {"noncolin": True, "starting_magnetization": [0.001]}
    if spin == SpinType.SPIN_ORBIT:
        return {"noncolin": True, "lspinorb": True, "starting_magnetization": [0.001]}
    return {"nspin": 2, "tot_magnetization": 0}


def _channel_w90_defaults(spin: SpinType, channel: SpinChannel) -> WannierizeOverrides:
    """Return the per-channel wannierization overrides a DFPT chain forces on.

    kcw.x reads the U matrices and Wannier centres from files the wannier90
    runs only write on request (``write_u_matrices`` / ``write_xyz``). With a
    collinear scratch, pw2wannier90 must pick its channel explicitly and the
    wannier90 input selects the same channel via ``spin`` (KCW example05.1
    nspin2); a spinor scratch instead needs ``spinors = .true.`` and no
    channel selection (nspin4 variants).

    Returned as the flat :class:`WannierizeOverrides` shape (``wannier90``
    / ``pw2wannier90``); :func:`WannierizeBlock` wraps these into
    the upstream builder namespace.

    These must be explicit overrides rather than upstream's
    ``spin_type`` machinery: ``Wannier90WorkChain`` injects
    ``spin_component`` at runtime by detecting nspin=2 from its *own*
    scf/nscf inputs, which :func:`WannierizeBlock` deliberately omits
    (shared-nscf pattern), so the upstream path can never fire here.
    """
    wannier90: dict[str, Any] = {"write_u_matrices": True, "write_xyz": True}
    defaults: WannierizeOverrides = {"wannier90": wannier90}
    if spin in (SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT):
        wannier90["spinors"] = True
        return defaults
    if spin == SpinType.COLLINEAR:
        wannier90["spin"] = channel.value
    defaults["pw2wannier90"] = {"spin_component": "down" if channel == SpinChannel.DOWN else "up"}
    return defaults


def _manifold_wannier_overrides(
    spin: SpinType, channel: SpinChannel, overrides: WannierizeOverrides
) -> WannierizeOverrides:
    """Assemble the flat wannier overrides for one channel's manifolds.

    Tight wannier90 convergence defaults (guiding centres keep the
    minimisation near the projection guess so the Wannier functions land in a
    reproducible minimum), the caller's ``overrides`` on top, and the channel
    staging/selection keys (:func:`_channel_w90_defaults`, kcw-chain
    requirements) force-merged last. All flat :class:`WannierizeOverrides` —
    :func:`WannierizeBlock` wraps the keyword dicts into the upstream builder
    namespace.
    """
    from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

    wannier_defaults: dict[str, Any] = {
        "guiding_centres": True,
        "num_iter": 10000,
        # The aiida-wannier90-workflows protocol raises num_cg_steps to 200;
        # on the ZnO live validation that setting left the spread
        # minimisation oscillating without convergence on matrices where the
        # wannier90 default (5) converges in ~400 iterations.
        "num_cg_steps": 5,
        "conv_tol": 1.0e-10,
        "conv_window": 5,
        # The aiida-wannier90-workflows protocol loosens dis_conv_tol to 4e-7;
        # pin wannier90's own default (1e-10) so the disentanglement is
        # tightly converged rather than the protocol's looser 4e-7.
        "dis_conv_tol": 1.0e-10,
    }
    channel_defaults = _channel_w90_defaults(spin, channel)
    wannier90 = recursive_merge(
        recursive_merge(wannier_defaults, dict(overrides.get("wannier90", {}))),
        channel_defaults.get("wannier90", {}),
    )
    wannier_overrides: WannierizeOverrides = {"wannier90": wannier90}
    pw2wannier90 = recursive_merge(
        dict(overrides.get("pw2wannier90", {})),
        channel_defaults.get("pw2wannier90", {}),
    )
    if pw2wannier90:
        wannier_overrides["pw2wannier90"] = pw2wannier90
    return wannier_overrides


def _seed_quality_check_nscf(
    wannier_overrides: WannierizeOverrides,
    bands_kpoints: orm.KpointsData | None,
    scf_nscf_overrides: dict[str, Any],
) -> None:
    """Seed the quality-check bands step's SYSTEM parameters, in place.

    A bands path unlocks the quality-check bands step in
    :func:`WannierizeBlocks` (paired with the shared scf,
    :func:`SinglepointDFPTWorkflow`'s own ``scf_remote_folder``): the same
    nscf overrides (``nbnd``, in particular) that seeded the shared nscf
    seed that step's SYSTEM parameters too, so it reads the full set of
    Wannierised bands rather than pw.x's default occupied-only count.
    """
    if bands_kpoints is not None:
        wannier_overrides["nscf"] = scf_nscf_overrides["nscf"]


def _wannierize_codes_for_channel(codes: DfptCodes) -> WannierizeBlocksCodes:
    """Return one channel's :func:`WannierizeBlocks` codes namespace.

    Wires every code :class:`WannierizeBlocksCodes` requires through
    :class:`DfptCodes`' ``ref()``: a member :class:`WannierizeBlocksCodes`
    requires but :class:`DfptCodes` never declared is a build-time
    ``ValueError`` from ``ref()`` itself, naming the missing member, rather
    than the bare ``KeyError`` a subscript would raise. ``projwfc`` stays a
    membership check on ``codes`` rather than an unconditional ``ref()``:
    this function's own caller (:func:`_add_quality_check_dfpt_inputs`)
    tests ``"projwfc" in wannierize_codes`` in the *same* eager scope — an
    unresolved ``ref()`` is still a present dict value there, so an
    unconditional ``ref()`` would make that membership test always true and
    wire a projected-DOS input WannierizeBlocks never populates.
    ``wannierjl`` (:class:`WannierizeBlocksCodes`' other ``NotRequired``
    member, for split-mode) is out of scope here: the DFPT route never
    triggers a split.
    """
    wannierize_codes: dict[str, Any] = {
        name: ref(codes, name) for name in ("pw", "pw2wannier90", "wannier90")
    }
    if "projwfc" in codes:
        wannierize_codes["projwfc"] = ref(codes, "projwfc")
    return cast("WannierizeBlocksCodes", wannierize_codes)


def _projwfc_step_will_run(pseudo_family: str | None, structure: orm.StructureData) -> bool:
    """Whether :func:`WannierizeBlocks`' own projwfc step actually runs.

    Mirrors its gate (:func:`~aiida_koopmans.workgraphs.wannier90.projected_dos_supported`)
    exactly — the wiring decision below and the inner step's decision must
    share one predicate, or a caller with a configured ``projwfc`` code but
    unsupported pseudos (no ``PP_PSWFC``, unreadable headers, or no
    ``pseudo_family``) gets a ``projwfc`` socket wired to a step that never
    ran. Suppresses the warning here: :func:`WannierizeBlocks` already emits
    it once, for the user, when it makes the same call for real.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return projected_dos_supported(pseudo_family, structure)


def _add_quality_check_dfpt_inputs(
    dfpt_inputs: dict[str, Any],
    bands_kpoints: orm.KpointsData | None,
    wannierized: Any,
    wannierize_codes: WannierizeBlocksCodes,
    pseudo_family: str | None,
    structure: orm.StructureData,
) -> None:
    """Wire the quality-check ``bands`` / ``projwfc`` sockets into ``dfpt_inputs``, in place.

    ``bands`` runs whenever a bands path was given (:func:`WannierizeBlocks`'
    own gate); subscripting its socket only then avoids wiring one the run
    structurally never populates. ``projwfc`` additionally needs a
    configured code *and* pseudos :func:`WannierizeBlocks` accepts for the
    projected DOS (:func:`_projwfc_step_will_run`) — wiring it on the code's
    presence alone would hand ``RunDFPT`` a socket with nothing behind it
    whenever the pseudos are unsupported.
    """
    if bands_kpoints is not None:
        dfpt_inputs["wannierize_bands"] = wannierized["bands"]
        if "projwfc" in wannierize_codes and _projwfc_step_will_run(pseudo_family, structure):
            dfpt_inputs["projwfc"] = wannierized["projwfc"]


@task.graph
def SinglepointDFPTWorkflow(
    codes: DfptCodes,
    structure: orm.StructureData,
    manifolds: dict[str, ManifoldBlocks],
    kpoints: orm.KpointsData,
    scf_kpoints: orm.KpointsData | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    eps_inf: float | str | None = None,
    l_vcut: bool | None = None,
    spin: SpinType = SpinType.NONE,
    check_spread: bool = True,
    group_orbitals_tol: float | None = None,
    parallelization: ParallelizationDict | None = None,
) -> KoopmansDFPTOutputs:
    """End-to-end singlepoint Koopmans DFPT: wannierize, then the kcw.x chain.

    ``eps_inf`` may be ``"auto"``: a scf + ph.x dielectric chain
    (:func:`~aiida_koopmans.workgraphs.ph.DielectricTask`, needs
    ``codes["ph"]``) runs first and the isotropic average of its dielectric
    tensor feeds the screen step.

    ``kpoints`` is the nscf mesh, and must be a Monkhorst-Pack mesh rather
    than an explicit list: the Wannier functions and kcw.x's
    ``CONTROL.mp1-3`` both count in its dimensions. The scf shares it unless
    ``scf_kpoints`` gives it a mesh of its own or ``overrides["scf"]`` a
    ``kpoints_distance``. Whichever it samples, the ``eps_inf = "auto"``
    dielectric chain's ground state samples the same.

    The workflow has three stages: compute the ground state (one shared
    scf + nscf, with ``nosym`` / ``noinv`` on the nscf so kcw.x sees the
    full k-point set), Wannierize each spin channel's occupied and empty
    blocks (:func:`WannierizeBlocks`), and run the kcw.x chain per spin
    channel (:func:`RunDFPT`). ``manifolds`` — a dict keyed by spin channel
    (:class:`SpinChannel` values as strings) with :class:`ManifoldBlocks`
    values — sets the channels:

    ``bands_kpoints`` also reaches each channel's :func:`WannierizeBlocks`
    call as its ``interpolation_kpoints`` (paired with the shared scf's
    scratch, so the quality-check step runs off it rather than a fresh
    scf): each channel's Wannierization then interpolates its own band
    structure along the path and runs a pw.x quality-check ``bands`` run
    (the explicit reference the interpolation is judged against), forwarded
    into that channel's ``ChannelResults`` as ``wannierize_bands``. With a
    ``projwfc`` code in ``codes``, a projected DOS off that run's scratch
    rides along too (``ChannelResults.projwfc``) — unless a pseudo carries
    no ``PP_PSWFC`` atomic wavefunctions, which skips it with a warning
    (:func:`~aiida_koopmans.workgraphs.wannier90.projected_dos_supported`).

    * ``spin = NONE`` — ``manifolds = {"none": ...}``: one chain on the up
      channel of the closed-shell nspin=2 scratch.
    * ``spin = COLLINEAR`` — ``manifolds = {"up": ..., "down": ...}``: one
      wannierization + kcw chain per channel (``CONTROL.spin_component``
      1 / 2). The caller's ``overrides`` must supply the magnetization
      (``tot_magnetization`` or ``starting_magnetization``).
    * ``spin = NON_COLLINEAR`` / ``SPIN_ORBIT`` — ``manifolds =
      {"none": ...}``: one chain on the spinor scratch; the blocks must be
      spinor manifolds (``num_wann`` doubled, from
      ``derive_dfpt_manifolds(..., spin_channel=SPINOR)``).

    Each channel's results land under its key in the ``channels`` output
    namespace.

    ``overrides`` is the flat :class:`WannierizeOverrides`: ``"scf"`` /
    ``"nscf"`` feed the shared PW steps, and ``"wannier90"`` /
    ``"pw2wannier90"`` feed every per-manifold wannier builder (the
    channel staging keys are force-merged on top per channel).

    ``check_spread`` reaches every channel's screen step unchanged (kcw.x's
    internal self-Hartree grouping — see :func:`RunDFPT`).

    ``group_orbitals_tol`` reaches every channel's :func:`RunDFPT`:
    workflow-level orbital grouping by wannier90 spread with one
    ``SCREEN.i_orb`` screen calculation per group representative. The
    spreads are the channel's unified band-ordered ``spreads`` output of
    :func:`WannierizeBlocks`, threaded to :func:`RunDFPT` alongside the
    retrieved folders. Each channel clusters its own Wannier functions
    independently (a channel running from its ``alpha_guess`` skips
    screening entirely, grouping included).
    """
    validate_parallelization(parallelization)

    from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

    overrides = overrides or {}
    collinear = spin == SpinType.COLLINEAR

    # Dynamic-namespace output keys must be plain strings, and the channel
    # bookkeeping below rests on the keys naming real spin channels.
    channel_keys = {str(key) for key in manifolds}
    expected_keys = (
        {SpinChannel.UP.value, SpinChannel.DOWN.value} if collinear else {SpinChannel.NONE.value}
    )
    if channel_keys != expected_keys:
        raise ValueError(
            f"spin={spin.value!r} requires manifolds keyed by "
            f"{sorted(expected_keys)}, got {sorted(channel_keys)}."
        )

    # The scf shares the nscf mesh unless the caller states otherwise, either
    # as a mesh of its own or as a ``kpoints_distance`` in its overrides. Both
    # scf runs the graph may contain resolve it here, so the two ground states
    # in one graph cannot end up on different meshes.
    if scf_kpoints is None and "kpoints_distance" not in overrides.get("scf", {}):
        scf_kpoints = kpoints

    if eps_inf == "auto":
        # Run a scf + ph.x dielectric chain first and feed tr(eps)/3 into the
        # screen step. The dielectric scf drops ``nbnd`` (no empty bands are
        # needed for a ground-state response) and none of the kcw spin
        # forcing — it is an independent ground state, but on the same mesh as
        # the chain's own.
        eps_scf_overrides = deepcopy(dict(overrides.get("scf", {})))
        eps_scf_overrides.get("pw", {}).get("parameters", {}).get("SYSTEM", {}).pop("nbnd", None)
        dielectric = DielectricTask(
            codes={"pw": ref(codes, "pw"), "ph": ref(codes, "ph")},
            structure=structure,
            pseudo_family=pseudo_family,
            protocol=protocol,
            scf_kpoints=scf_kpoints,
            overrides={"scf": eps_scf_overrides},
            parallelization=parallelization,
            metadata={"call_link_label": "dielectric"},
        )
        eps_inf = dielectric["eps_inf"]

    forced_system = _pw_spin_system_defaults(spin)
    # The domag nudge is a *default*, not a requirement: a genuinely magnetic
    # system supplies its own starting_magnetization, which must win.
    seed_system = {}
    if "starting_magnetization" in forced_system:
        seed_system = {"starting_magnetization": forced_system.pop("starting_magnetization")}

    def _with_spin(user: dict[str, Any], extra_forced: dict[str, Any]) -> dict[str, Any]:
        # seed (under) <- user <- forced (on top): the forced nspin/noncolin
        # keys overwrite user values, since e.g. a user nspin=1 would
        # silently break kcw.x.
        forced = {
            "pw": {"parameters": {"SYSTEM": {**forced_system, **extra_forced}}},
        }
        seeded = recursive_merge({"pw": {"parameters": {"SYSTEM": dict(seed_system)}}}, user)
        return recursive_merge(seeded, forced)

    scf_nscf_overrides: dict[str, Any] = {
        "scf": _with_spin(overrides.get("scf", {}), {}),
        "nscf": _with_spin(overrides.get("nscf", {}), {"nosym": True, "noinv": True}),
    }

    # wannier90 / pw2wannier90 need the nscf eigenstates on the full
    # (symmetry-unreduced) user grid, listed in wannier90's own k-point
    # order — expand the mesh once and share the explicit list between the
    # nscf and every per-block wannierisation. ``mp_grid`` keeps the mesh
    # dimensions, which wannier90 cannot re-derive from an explicit list.
    # The scf takes the mesh itself and may reduce it by symmetry.
    from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints

    # wannier90's ``mp_grid`` and kcw.x's ``CONTROL.mp1-3`` are the same three
    # numbers: the dimensions of the mesh the Wannier functions were built on.
    try:
        mp_grid = [int(size) for size in kpoints.get_kpoints_mesh()[0]]
    except AttributeError:
        raise ValueError(
            "`kpoints` must be a Monkhorst-Pack mesh (`set_kpoints_mesh`), not an "
            "explicit list of k-points: kcw.x counts in the mesh dimensions "
            "(`CONTROL.mp1-3`)."
        ) from None
    explicit_kpoints = get_explicit_kpoints(kpoints)

    scf_nscf = RunScfNscf(
        pw_code=ref(codes, "pw"),
        structure=structure,
        pseudo_family=pseudo_family,
        protocol=protocol,
        overrides=scf_nscf_overrides,
        nscf_kpoints=explicit_kpoints,
        scf_kpoints=scf_kpoints,
        parallelization=parallelization,
        metadata={"call_link_label": "scf_nscf"},
    )
    nscf_remote_folder = scf_nscf["nscf_remote_folder"]

    channel_results: dict[str, ChannelResults] = {}
    for channel_key, manifold in manifolds.items():
        channel_key = str(channel_key)
        channel = SpinChannel(channel_key)
        suffix = f"_{channel_key}" if collinear else ""
        wannier_overrides = _manifold_wannier_overrides(spin, channel, overrides)
        _seed_quality_check_nscf(wannier_overrides, bands_kpoints, scf_nscf_overrides)

        occ_blocks = list(manifold["occ"])
        emp_blocks = list(manifold.get("emp") or [])
        alpha_guess = manifold.get("alpha_guess")

        wannierize_codes = _wannierize_codes_for_channel(codes)

        # One WannierizeBlocks per channel, over the channel's blocks in band
        # order (occupied then empty). Fed the shared nscf scratch so its
        # internal scf + nscf is skipped — the ground state runs once across
        # channels. The unified ``spreads`` output is band-ordered by the
        # same list, exactly the order kcw.x counts ``SCREEN.i_orb`` in.
        # ``scf_remote_folder`` alongside the shared ``nscf_remote_folder``
        # still unlocks the quality-check bands / projected-DOS run: it
        # reads the shared scf's density rather than running its own.
        wannierized = WannierizeBlocks(
            codes=wannierize_codes,
            structure=structure,
            blocks=occ_blocks + emp_blocks,
            kpoints=explicit_kpoints,
            mp_grid=mp_grid,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides=wannier_overrides,
            nscf_remote_folder=nscf_remote_folder,
            scf_remote_folder=scf_nscf["scf_remote_folder"],
            nscf_bands=scf_nscf["nscf_output_band"],
            interpolation_kpoints=bands_kpoints,
            parallelization=parallelization,
            metadata={"call_link_label": f"wannierize{suffix}"},
        )
        # Hand RunDFPT the whole ``blocks`` namespace: a nested sub-graph's
        # dynamic namespace has no per-key sockets at build time, so it must
        # flow wholesale; RunDFPT picks blocks out by label in its deferred
        # body. Manifold membership and band order travel as the caller's
        # own label lists (structural knowledge, not label parsing).
        dfpt_inputs: dict[str, Any] = {
            "kcw_code": ref(codes, "kcw"),
            "nscf_remote_folder": nscf_remote_folder,
            "block_wannier": wannierized["blocks"],
            "occ_labels": [str(block["label"]) for block in occ_blocks],
            "num_wann_occ": sum(block["num_wann"] for block in occ_blocks),
            "num_wann_emp": 0,
            "kgrid": mp_grid,
            "spreads": wannierized["spreads"],
            "bands_kpoints": bands_kpoints,
            "eps_inf": eps_inf,
            "alpha_guess": alpha_guess,
            "group_orbitals_tol": group_orbitals_tol,
            "l_vcut": l_vcut,
            "spin_component": 2 if channel == SpinChannel.DOWN else 1,
            "check_spread": check_spread,
            "parallelization": parallelization,
            "metadata": {"call_link_label": f"dfpt{suffix}"},
        }
        _add_quality_check_dfpt_inputs(
            dfpt_inputs, bands_kpoints, wannierized, wannierize_codes, pseudo_family, structure
        )

        if emp_blocks:
            num_wann_emp = sum(block["num_wann"] for block in emp_blocks)
            # Every block has num_bands == num_wann except the last, which
            # absorbs the manifold's disentanglement bands, so the sum is the
            # total empty-band count (nbnd - nocc).
            nbnd_emp = sum(block["num_bands"] for block in emp_blocks)
            dfpt_inputs["emp_labels"] = [str(block["label"]) for block in emp_blocks]
            dfpt_inputs["num_wann_emp"] = num_wann_emp
            dfpt_inputs["nbnd_emp"] = nbnd_emp
            # Disentanglement is a property of the empty manifold, not caller
            # state: extra bands beyond the Wannier count mean it disentangles.
            dfpt_inputs["has_disentangle"] = nbnd_emp != num_wann_emp

        dfpt = RunDFPT(**dfpt_inputs)

        # Assign the whole RunDFPT output namespace as this key's value (the
        # engine maps one socket per dynamic key; re-packing individual
        # sockets into a fresh dict is not resolvable at execution time).
        channel_results[channel_key] = dfpt

    return KoopmansDFPTOutputs(channels=channel_results)
