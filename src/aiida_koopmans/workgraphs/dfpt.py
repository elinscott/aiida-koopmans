"""Koopmans DFPT workflow (kcw.x): wann2kc → screen → ham.

The three steps are backed by the CalcJobs in
``aiida_koopmans.calculations.kcw`` (one kcw.x binary, three
``CONTROL.calculation`` modes).

Two graphs are exposed:

* :func:`RunDFPT` -- the kcw.x chain proper. It *consumes*
  wannierization outputs (the shared nscf scratch plus the per-block
  wannier90 ``retrieved`` folders, merged per manifold) and runs
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
see :mod:`aiida_koopmans.wannier_merge`) before kcw.x consumes them.

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

from __future__ import annotations

from copy import deepcopy
from typing import Annotated, Any, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import SpinType
from aiida_workgraph import dynamic, task

from aiida_koopmans.calculations.kcw import (
    KcwHamCalculation,
    KcwScreenCalculation,
    Wann2kcCalculation,
)
from aiida_koopmans.occupations import default_channel_nocc
from aiida_koopmans.projections import (
    band_range_complement,
    projection_num_wann,
    projection_win_string,
)
from aiida_koopmans.types import (
    ExplicitProjectionBlock,
    ProjectionBlock,
    SpinChannel,
    VariationalOrbital,
    map_key_for,
)
from aiida_koopmans.wannier_merge import (
    extend_wannier_u_dis_file_content,
    merge_wannier_centres_file_contents,
    merge_wannier_hr_file_contents,
    merge_wannier_u_file_contents,
    parse_wannier_u_file_shape,
)
from aiida_koopmans.workgraphs import Codes
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks, WannierizeOverrides
from aiida_koopmans.workgraphs.ph import DielectricTask
from aiida_koopmans.workgraphs.pw import RunScfNscf
from aiida_koopmans.workgraphs.variational_orbitals import (
    assign_orbital_groups,
    expand_alphas_by_group,
    spreads_metric_row,
)

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


def _split_manifolds(
    blocks_with_counts: list[tuple[Any, int]], nocc: int
) -> tuple[list[tuple[Any, int]], list[tuple[Any, int]]]:
    """Split (block, num_wann) pairs at the occupied/empty boundary."""
    occupied: list[tuple[Any, int]] = []
    empty: list[tuple[Any, int]] = []
    cursor = 0
    for block, num_wann in blocks_with_counts:
        if cursor + num_wann <= nocc:
            occupied.append((block, num_wann))
        elif cursor >= nocc:
            empty.append((block, num_wann))
        else:
            raise ValueError(
                f"A projection block (bands {cursor + 1}-{cursor + num_wann}) straddles "
                f"the occupied/empty boundary at band {nocc}."
            )
        cursor += num_wann
    return occupied, empty


def _manifold_projection_blocks(
    manifold: list[tuple[Any, int]],
    name: str,
    label_suffix: str,
    spin_channel: SpinChannel,
    first_band: int,
    nbnd: int,
    extra_bands: int,
) -> list[ExplicitProjectionBlock]:
    """Materialise one manifold's per-block :class:`ExplicitProjectionBlock` list.

    Blocks cover consecutive band windows starting at ``first_band``. Only
    the *last* block absorbs the manifold's ``extra_bands`` disentanglement
    bands (``num_bands > num_wann``), the band layout the u_dis merge in
    :func:`prepare_kcw_wannier_files` relies on. A single-block manifold
    keeps the bare ``occ`` / ``emp`` label; multi-block manifolds are
    numbered (``occ_1``, ``occ_up_1``, ...).
    """
    from aiida_wannier90_workflows.common.types import WannierProjectionType

    blocks: list[ExplicitProjectionBlock] = []
    cursor = first_band - 1
    for i, (projections, num_wann) in enumerate(manifold):
        is_last = i == len(manifold) - 1
        num_bands = num_wann + (extra_bands if is_last else 0)
        start = cursor + 1
        end = start + num_bands - 1
        label = f"{name}{label_suffix}" if len(manifold) == 1 else f"{name}{label_suffix}_{i + 1}"
        blocks.append(
            ExplicitProjectionBlock(
                label=label,
                spin=spin_channel,
                num_wann=num_wann,
                num_bands=num_bands,
                include_bands=list(range(start, end + 1)),
                exclude_bands=band_range_complement(start, end, nbnd),
                projection_type=WannierProjectionType.ANALYTIC,
                projections=[projection_win_string(p) for p in projections],
            )
        )
        cursor += num_wann
    return blocks


def derive_dfpt_manifolds(
    structure: orm.StructureData,
    projection_blocks: list,
    nelec: int,
    nbnd: int | None,
    spin_channel: SpinChannel = SpinChannel.NONE,
    nocc: int | None = None,
) -> tuple[list[ExplicitProjectionBlock], list[ExplicitProjectionBlock], bool, int]:
    """Turn user projection blocks into the occupied/empty DFPT manifolds.

    Handles the manifold bookkeeping (nocc from the electron count, per-block
    consecutive band windows, disentanglement bands attached to the last
    block of the empty manifold) for one spin channel. Any number of blocks
    per manifold is allowed; a manifold Wannierised as several blocks is
    merged again before kcw.x by :func:`prepare_kcw_wannier_files`.

    Args:
        structure: the periodic structure (for per-site projection counting).
        projection_blocks: list of projection blocks *for this channel*, each
            a list of ``wannier90_input`` ``Projection``-like objects, in
            band order.
        nelec: total electron count (from the pseudopotential valences).
        nbnd: number of bands of the nscf, or None to default to nocc.
        spin_channel: which channel these blocks describe. ``NONE`` (default)
            is spin-unpolarized (``nocc = nelec / 2``); ``UP`` / ``DOWN`` are
            the collinear channels (caller must supply the per-channel
            ``nocc`` from the magnetization); ``SPINOR`` is the noncollinear
            case — every band is singly occupied (``nocc = nelec``) and each
            projection yields two spinor Wannier functions.
        nocc: per-channel occupied-band count, overriding the electron-count
            default. Required for ``UP`` / ``DOWN``.

    Returns:
        ``(occ_blocks, emp_blocks, has_disentangle, n_orbitals)`` where the
        block lists hold :class:`ExplicitProjectionBlock` entries in band
        order (``emp_blocks`` may be empty), ``has_disentangle`` says whether
        the empty manifold has more bands than Wannier functions, and
        ``n_orbitals = num_wann_occ + num_wann_emp``.
    """
    spinor = spin_channel == SpinChannel.SPINOR
    if nocc is None:
        nocc = default_channel_nocc(spin_channel, nelec)
    nbnd = nocc if nbnd is None else int(nbnd)

    if not projection_blocks:
        raise NotImplementedError(
            "DFPT screening requires explicit Wannier90 projections in "
            "``calculator_parameters.w90.projections``."
        )

    # With spinors (nspin=4) each projection orbital carries two spin
    # components, so a projection block spans twice as many Wannier
    # functions as its orbital count (KCW example05.1: sp3 -> num_wann 8).
    wann_per_orbital = 2 if spinor else 1
    blocks_with_counts = [
        (block, wann_per_orbital * sum(projection_num_wann(structure, p) for p in block))
        for block in projection_blocks
    ]
    occupied, empty = _split_manifolds(blocks_with_counts, nocc)

    num_wann_occ = sum(num_wann for _, num_wann in occupied)
    if num_wann_occ != nocc:
        raise ValueError(
            f"The occupied projection blocks span {num_wann_occ} Wannier functions but "
            f"the system has {nocc} occupied bands."
        )

    label_suffix = (
        f"_{spin_channel.value}" if spin_channel in (SpinChannel.UP, SpinChannel.DOWN) else ""
    )
    occ_blocks = _manifold_projection_blocks(
        occupied, "occ", label_suffix, spin_channel, 1, nbnd, 0
    )

    emp_blocks: list[ExplicitProjectionBlock] = []
    has_disentangle = False
    num_wann_emp = sum(num_wann for _, num_wann in empty)
    if empty:
        num_bands_emp = nbnd - nocc
        if num_bands_emp < num_wann_emp:
            raise ValueError(
                f"nbnd = {nbnd} leaves only {num_bands_emp} empty bands but the empty "
                f"projection blocks require {num_wann_emp} Wannier functions."
            )
        has_disentangle = num_bands_emp != num_wann_emp
        emp_blocks = _manifold_projection_blocks(
            empty,
            "emp",
            label_suffix,
            spin_channel,
            nocc + 1,
            nbnd,
            num_bands_emp - num_wann_emp,
        )

    return occ_blocks, emp_blocks, has_disentangle, num_wann_occ + num_wann_emp


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
    code: orm.AbstractCode,
    control: dict,
    wannier: dict,
    screen_namelist: dict,
    parent_folder: orm.RemoteData,
    wannier_files: orm.FolderData,
    orbitals: list[VariationalOrbital],
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
        screen = KcwScreenStep(
            code=code,
            parameters={"CONTROL": control, "WANNIER": wannier, "SCREEN": namelist},
            parent_folder=parent_folder,
            wannier_files=wannier_files,
            metadata={"call_link_label": f"screen_{key}"},
        )
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


class _ManifoldBlocksRequired(TypedDict):
    """Required part of :class:`ManifoldBlocks` (split so the rest can be optional)."""

    occ: list[ProjectionBlock]


class ManifoldBlocks(_ManifoldBlocksRequired, total=False):
    """Per-spin-channel manifold description consumed by :func:`SinglepointDFPTWorkflow`.

    * ``occ`` -- the occupied projection blocks in band order (at least one;
      several when the occupied manifold spans multiple projection blocks).
    * ``emp`` -- the empty projection blocks, when the channel has any.
    * ``alpha_guess`` -- per-orbital screening-parameter guess for this
      channel; when given the channel's screen step is skipped.

    A manifold Wannierised as several blocks has its per-block Wannier
    products merged back into one file set by :func:`prepare_kcw_wannier_files`.
    """

    emp: list[ProjectionBlock]
    alpha_guess: list[float] | None


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
    :func:`_manifold_projection_blocks` fixes). When the manifold has more
    bands than Wannier functions its ``u_dis`` is required: a single-block
    manifold stages the file unchanged, a merged one extends it with an
    identity for the preceding blocks
    (:func:`~aiida_koopmans.wannier_merge.extend_wannier_u_dis_file_content`).
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
    codes: Codes,
    nscf_remote_folder: orm.RemoteData,
    occ_retrieved: Annotated[dict, dynamic(orm.FolderData)],
    num_wann_occ: int,
    num_wann_emp: int,
    kgrid: list[int],
    emp_retrieved: Annotated[dict | None, dynamic(orm.FolderData)] = None,
    nbnd_emp: int | None = None,
    spreads: list | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    eps_inf: float | None = None,
    alpha_guess: list[float] | None = None,
    group_orbitals_tol: float | None = None,
    has_disentangle: bool = False,
    l_vcut: bool | None = None,
    spin_component: int = 1,
    check_spread: bool = True,
) -> ChannelResults:
    """Run the kcw.x chain off provided wannierization outputs.

    Args:
        codes: code instances; only ``codes["kcw"]`` is used.
        nscf_remote_folder: scratch of the pw.x **nscf** run the Wannier
            functions were built on (kcw.x re-reads its wavefunctions). Must
            be an ``nspin = 2`` run even for closed-shell systems -- the DFPT
            perturbations are spin-dependent; the kcw chain reads the up
            channel (``CONTROL.spin_component = 1``).
        occ_retrieved: the occupied-manifold wannier90 ``retrieved`` folders
            (each must hold ``aiida_u.mat`` / ``aiida_hr.dat`` /
            ``aiida_centres.xyz``), keyed so lexicographic key order matches
            the band order of the manifold's blocks; multi-block manifolds
            are merged by :func:`prepare_kcw_wannier_files`.
        num_wann_occ / num_wann_emp: *total* Wannier function counts per
            manifold (``num_wann_emp = 0`` for an occupied-only run).
        kgrid: the Monkhorst-Pack grid of the nscf, for ``CONTROL.mp1-3``.
        emp_retrieved: the empty-manifold wannier90 ``retrieved`` folders
            (same keying convention as ``occ_retrieved``).
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

    prep_inputs: dict[str, Any] = {f"occ_{key}": folder for key, folder in occ_retrieved.items()}
    if emp_retrieved is not None:
        for key, folder in emp_retrieved.items():
            prep_inputs[f"emp_{key}"] = folder
        if nbnd_emp is not None:
            prep_inputs["nbnd_emp"] = nbnd_emp
    wannier_files = prepare_kcw_wannier_files(
        **prep_inputs,
        metadata={"call_link_label": "prepare_kcw_wannier_files"},
    )["wannier_files"]

    wann2kc = Wann2kcStep(
        code=codes["kcw"],
        parameters={"CONTROL": control, "WANNIER": wannier},
        parent_folder=nscf_remote_folder,
        wannier_files=wannier_files,
        metadata={"call_link_label": "wann2kc"},
    )

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
            code=codes["kcw"],
            control=control,
            wannier=wannier,
            screen_namelist=screen_namelist,
            parent_folder=wann2kc["remote_folder"],
            wannier_files=wannier_files,
            orbitals=orbitals.result,
            metadata={"call_link_label": "grouped_screen"},
        )
        alphas = grouped["alphas"]
    else:
        # ``bool()`` unwraps a possible wrapt proxy, as for ``l_vcut``.
        screen_namelist["check_spread"] = bool(check_spread)
        screen = KcwScreenStep(
            code=codes["kcw"],
            parameters={"CONTROL": control, "WANNIER": wannier, "SCREEN": screen_namelist},
            parent_folder=wann2kc["remote_folder"],
            wannier_files=wannier_files,
            metadata={"call_link_label": "screen"},
        )
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
        "code": codes["kcw"],
        "parameters": {"CONTROL": control, "WANNIER": wannier, "HAM": ham_namelist},
        "parent_folder": wann2kc["remote_folder"],
        "wannier_files": wannier_files,
        "alphas": alphas,
        "metadata": {"call_link_label": "ham"},
    }
    if do_bands:
        ham_inputs["kpoints"] = bands_kpoints
    ham = KcwHamStep(**ham_inputs)

    outputs["alphas"] = alphas
    outputs["ham_parameters"] = ham["output_parameters"]
    if do_bands:
        outputs["bands"] = ham["bands"]
    return outputs


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
        # pin wannier90's own default (1e-10, what legacy runs with) so the
        # disentanglement is converged as tightly as the legacy reference.
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


@task.graph
def SinglepointDFPTWorkflow(
    codes: Codes,
    structure: orm.StructureData,
    manifolds: dict[str, ManifoldBlocks],
    kpoints: orm.KpointsData,
    kgrid: list[int],
    bands_kpoints: orm.KpointsData | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    eps_inf: float | str | None = None,
    l_vcut: bool | None = None,
    spin: SpinType = SpinType.NONE,
    check_spread: bool = True,
    group_orbitals_tol: float | None = None,
) -> KoopmansDFPTOutputs:
    """End-to-end singlepoint Koopmans DFPT: wannierize, then the kcw.x chain.

    ``eps_inf`` may be ``"auto"``: a scf + ph.x dielectric chain
    (:func:`~aiida_koopmans.workgraphs.ph.DielectricTask`, needs
    ``codes["ph"]``) runs first and the isotropic average of its dielectric
    tensor feeds the screen step.

    One shared scf + nscf (:func:`RunScfNscf`, with the spin-regime SYSTEM
    keys of :func:`_pw_spin_system_defaults` forced on and ``nosym`` /
    ``noinv`` on the nscf so kcw.x sees the full k-point set), then one
    :func:`WannierizeBlocks` per spin channel — fed the shared nscf scratch
    via its ``nscf_remote_folder`` input, so its internal scf + nscf is
    skipped and the ground state runs exactly once across channels — over
    that channel's occupied + empty blocks in band order (a manifold may
    span several blocks, whose Wannier products :func:`RunDFPT` merges back
    into one file set), and one :func:`RunDFPT` per entry of ``manifolds``
    — a dict keyed by spin channel (:class:`SpinChannel` values as strings)
    whose values are :class:`ManifoldBlocks`:

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

    if eps_inf == "auto":
        # Run a scf + ph.x dielectric chain first and feed tr(eps)/3 into the
        # screen step. The dielectric scf drops ``nbnd`` (no empty bands are
        # needed for a ground-state response) and none of the kcw spin
        # forcing — it is an independent ground state.
        if "ph" not in codes:
            raise ValueError("eps_inf='auto' requires a ph.x code under codes['ph'].")
        eps_scf_overrides = deepcopy(dict(overrides.get("scf", {})))
        eps_scf_overrides.get("pw", {}).get("parameters", {}).get("SYSTEM", {}).pop("nbnd", None)
        dielectric = DielectricTask(
            pw_code=codes["pw"],
            ph_code=codes["ph"],
            structure=structure,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides={"scf": eps_scf_overrides},
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
    from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints

    mp_grid = kpoints.get_kpoints_mesh()[0]
    explicit_kpoints = get_explicit_kpoints(kpoints)

    scf_nscf = RunScfNscf(
        code=codes["pw"],
        structure=structure,
        pseudo_family=pseudo_family,
        protocol=protocol,
        overrides=scf_nscf_overrides,
        nscf_kpoints=explicit_kpoints,
        metadata={"call_link_label": "scf_nscf"},
    )
    nscf_remote_folder = scf_nscf["nscf_remote_folder"]

    def _manifold_retrieved(blocks_ns: Any, manifold_blocks: list) -> dict[str, Any]:
        """Key one manifold's per-block ``retrieved`` sockets for the file merge.

        The manifold membership and order come from the caller's own block
        lists (structural knowledge, not label parsing); the ``b{i:02d}``
        keying is the lexicographic-equals-band-order convention
        :func:`prepare_kcw_wannier_files` merges by.
        """
        return {
            f"b{i:02d}": blocks_ns[block["label"]]["hr_retrieved"]
            for i, block in enumerate(manifold_blocks)
        }

    channel_results: dict[str, ChannelResults] = {}
    for channel_key, manifold in manifolds.items():
        channel_key = str(channel_key)
        channel = SpinChannel(channel_key)
        suffix = f"_{channel_key}" if collinear else ""
        wannier_overrides = _manifold_wannier_overrides(spin, channel, overrides)

        occ_blocks = list(manifold["occ"])
        emp_blocks = list(manifold.get("emp") or [])
        alpha_guess = manifold.get("alpha_guess")

        # One WannierizeBlocks per channel, over the channel's blocks in band
        # order (occupied then empty). Fed the shared nscf scratch so its
        # internal scf + nscf is skipped — the ground state runs once across
        # channels. The unified ``spreads`` output is band-ordered by the
        # same list, exactly the order kcw.x counts ``SCREEN.i_orb`` in.
        wannierized = WannierizeBlocks(
            codes=codes,
            structure=structure,
            blocks=occ_blocks + emp_blocks,
            kpoints=explicit_kpoints,
            mp_grid=mp_grid,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides=wannier_overrides,
            nscf_remote_folder=nscf_remote_folder,
            metadata={"call_link_label": f"wannierize{suffix}"},
        )
        blocks_ns = wannierized["blocks"]

        dfpt_inputs: dict[str, Any] = {
            "codes": codes,
            "nscf_remote_folder": nscf_remote_folder,
            "occ_retrieved": _manifold_retrieved(blocks_ns, occ_blocks),
            "num_wann_occ": sum(block["num_wann"] for block in occ_blocks),
            "num_wann_emp": 0,
            "kgrid": kgrid,
            "spreads": wannierized["spreads"],
            "bands_kpoints": bands_kpoints,
            "eps_inf": eps_inf,
            "alpha_guess": alpha_guess,
            "group_orbitals_tol": group_orbitals_tol,
            "l_vcut": l_vcut,
            "spin_component": 2 if channel == SpinChannel.DOWN else 1,
            "check_spread": check_spread,
            "metadata": {"call_link_label": f"dfpt{suffix}"},
        }

        if emp_blocks:
            num_wann_emp = sum(block["num_wann"] for block in emp_blocks)
            # Every block has num_bands == num_wann except the last, which
            # absorbs the manifold's disentanglement bands, so the sum is the
            # total empty-band count (nbnd - nocc).
            nbnd_emp = sum(block["num_bands"] for block in emp_blocks)
            dfpt_inputs["emp_retrieved"] = _manifold_retrieved(blocks_ns, emp_blocks)
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
