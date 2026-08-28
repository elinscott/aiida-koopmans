"""Block-by-block Wannierisation of a periodic system.

A single shared scf + nscf is run once (via :func:`RunWannierGroundState`, or skipped
entirely when the caller supplies an existing ``nscf_remote_folder``), then
each projection block (occupied / empty manifold, per spin) is Wannierised
in its own ``Wannier90WorkChain`` that *skips* scf and nscf and reads the
shared nscf scratch directly. The per-block fan-out is a native ``for``
loop over ``blocks`` inside the ``@task.graph`` body -- do not convert it
to a ``Map`` zone. Results are collected into a dict keyed by each block's
stable ``label`` (e.g. ``"block_1"`` / ``"block_1_spin_up"``) and returned
as a dynamic output namespace.

Per-block file staging that the supercell fold consumes:

* ``retrieved`` -- the wannier90 ``retrieved`` :class:`~aiida.orm.FolderData`,
  which holds ``aiida_hr.dat`` (the real-space Hamiltonian, written because
  ``write_hr=True``) plus ``aiida.chk``, ``aiida_u.mat``, ``aiida_centres.xyz``
  and, for disentangling blocks, ``aiida_u_dis.mat``. All but ``aiida.chk``
  are retrieved by upstream's default suffix list once written (the ``write_*``
  pins are what guarantee they exist); ``aiida.chk`` is force-retrieved.
  Downstream consumers such as pw2wannier90 ``wan_mode='decompose'`` and the
  wannierjl split read them.
* ``remote_folder`` -- the wannier90 ``RemoteData`` scratch.
* ``nnkp_file`` -- the ``aiida.nnkp`` :class:`~aiida.orm.SinglefileData`
  emitted by the wannier90 post-processing (``-pp``) run.

Alongside the file staging, each block also exposes the parsed wannier90
``output_parameters`` :class:`~aiida.orm.Dict` (per-WF spreads / centres,
Omega decomposition), so downstream consumers that depend on parsed
quantities — e.g. the DFPT spread-based orbital grouping — read them from
the parser output rather than re-parsing the raw ``.wout``.

Because every downstream code consumes a *unified* view of the
Wannierisation (kcw.x reads one occupied + one empty file set, the fold
route merges per manifold), :func:`WannierizeBlocks` also emits unified,
band-ordered ``centres`` and ``spreads`` arrays concatenated across all
blocks by :func:`collect_wannier_functions`. Band order is taken from the
input ``blocks`` list order — the single authority — never reconstructed
from block labels or output-namespace keys.

:func:`WannierizeBlocks` also carries an optional split mode (triggered by a
``bands_kpoints`` input): a pw.x ``bands`` run feeds a runtime band-group
detection, and each block is handled by the automated block-splitting chain
of :mod:`aiida_koopmans.workgraphs.auto_wannierize` instead of a plain
:func:`WannierizeBlock`. The split machinery is lazy-imported inside the
graph body: it depends on ``aiida-wannierjl``, which the plain mode must not
require.
"""

# No ``from __future__ import annotations`` in this module: stringified
# annotations hide ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the dispatcher reads off the Codes
# TypedDicts.
import warnings
from typing import Annotated, Any, NotRequired, TypedDict, cast

import numpy as np
from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_wannier90_workflows.common.types import WannierFrozenType, WannierProjectionType
from aiida_wannier90_workflows.utils.workflows.builder.projections import (
    guess_wannier_projection_types,
)
from aiida_wannier90_workflows.workflows import Wannier90WorkChain
from aiida_workgraph import dynamic, task
from aiida_workgraph.socket_spec import SocketMeta
from aiida_workgraph.utils import get_dict_from_builder
from node_graph import reference

from aiida_koopmans.owned_keywords import owned
from aiida_koopmans.parallelization import (
    ParallelizationDict,
    merge_parallelization_into_inputs,
    validate_parallelization,
)
from aiida_koopmans.projections import (
    ProjectionBlock,
    ProjectionBlockId,
    block_display_name,
    block_occupancy,
    block_w90_kwargs,
    validate_projection_block,
    validate_projection_block_sequence,
)
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.variational_orbitals import VariationalOrbital
from aiida_koopmans.workgraphs import name_step, stamp_wannier90_pp_namespace, unwrap_enum
from aiida_koopmans.workgraphs.pw import (
    PwCode,
    PwOutputs,
    run_bands_step,
    run_nscf_step,
)
from aiida_koopmans.workgraphs.variational_orbitals import (
    initial_orbital_partition,
    ordered_block_specs,
)
from aiida_koopmans.workgraphs.wannier90 import (
    ProjwfcCode,
    ProjwfcOutputs,
    Pw2Wannier90Code,
    RunProjwfc,
    Wannier90Code,
    Wannier90Step,
    projected_dos_supported,
    require_path_labels,
)
from aiida_koopmans.workgraphs.wannier_ground_state import RunWannierGroundState, ScfNscfOutputs


class WannierizeBlockCodes(TypedDict):
    """Codes for one block's wannierization (:func:`WannierizeBlock`)."""

    # The upstream ``Wannier90WorkChain`` builder cannot assemble its inputs
    # without a pw code, even though this graph discards the scf / nscf
    # namespaces and runs no pw.x itself. This member can be dropped once the
    # upstream builder no longer demands a pw code for discarded namespaces.
    pw: Annotated[
        orm.AbstractCode,
        SocketMeta(help="Needed to set up the block's Wannierization; no pw.x calculation runs."),
    ]
    pw2wannier90: Pw2Wannier90Code
    wannier90: Wannier90Code


class WannierizeBlocksCodes(TypedDict):
    """Codes for :func:`WannierizeBlocks`."""

    pw: PwCode
    pw2wannier90: Pw2Wannier90Code
    wannier90: Wannier90Code
    wannierjl: NotRequired[
        Annotated[
            orm.AbstractCode,
            SocketMeta(help="Needed for threshold-based splitting of bands into blocks."),
        ]
    ]
    projwfc: NotRequired[ProjwfcCode]


# ``aiida.chk`` is the only wannier90 product upstream excludes from its
# retrieve-everything default: ``_DEFAULT_RETRIEVE_SUFFIXES`` in
# aiida-wannier90's ``Wannier90Calculation`` already covers ``_u.mat`` /
# ``_u_dis.mat`` / ``_centres.xyz`` / ``_hr.dat``, so once those files are
# written they land in ``retrieved`` automatically. What guarantees the
# product set is therefore the ``write_hr`` / ``write_u_matrices`` /
# ``write_xyz`` pins below, not this list. The supercell fold needs
# ``aiida.chk`` to unitarily rotate the per-block manifolds, so force it.
_W90_RETRIEVE_SETTINGS: dict[str, list[str]] = {"additional_retrieve_list": ["aiida.chk"]}

#: Projection sources the block wannierization supports: explicit orbital
#: lists, and — for automated wannierization — pseudoatomic orbitals fetched
#: from the pseudopotentials or from an external projector directory.
SUPPORTED_PROJECTION_TYPES = (
    WannierProjectionType.ANALYTIC,
    WannierProjectionType.ATOMIC_PROJECTORS_QE,
    WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
)


def validate_projection_type(projection_type: WannierProjectionType) -> None:
    """Reject projection types the block wannierization does not support.

    Every type outside :data:`SUPPORTED_PROJECTION_TYPES` (SCDM, random,
    ...) is out of scope. Compares by ``.value``: in a graph body the enum
    arrives as a provenance-tagged proxy, which fails the ``Enum``
    constructor's by-value lookup, while attribute access and ``==``
    delegate cleanly.
    """
    value = getattr(projection_type, "value", projection_type)
    if any(value == member.value for member in SUPPORTED_PROJECTION_TYPES):
        return
    raise ValueError(
        f"Projection type '{value}' is not supported: blocks are "
        "wannierized from explicit projections ('analytic') or pseudoatomic "
        "projectors, fetched from the pseudopotentials "
        "('atomic_projectors_qe') or from an external projector directory "
        "('atomic_projectors_external')."
    )


def _is_external(projection_type: WannierProjectionType) -> bool:
    """Report whether a (possibly proxy-wrapped) type is the external source."""
    value = getattr(projection_type, "value", projection_type)
    return bool(value == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL.value)


def validate_external_projector_inputs(
    projection_types: list[WannierProjectionType],
    external_projectors_path: str | None,
    external_projectors: dict[str, Any] | None,
) -> None:
    """Pair the external projector inputs with the external projection type.

    ``atomic_projectors_external`` needs both the projector directory (the
    pw2wannier90 step remote-copies each element's ``<symbol>.dat`` from it)
    and the per-element orbital tables (the builder derives the projector
    count from them); any other type consumes neither, so a
    mismatch in either direction raises rather than silently ignoring the
    given inputs. At most one external entry is allowed: the tables are not
    split per block, so every external block would wannierize the full
    projector manifold.
    """
    n_external = sum(1 for projection_type in projection_types if _is_external(projection_type))
    external = n_external > 0
    if n_external > 1:
        raise ValueError(
            f"{n_external} 'atomic_projectors_external' blocks were given, but only one "
            "is supported per call: the orbital tables are not split per block, so "
            "every external block would wannierize the full projector manifold."
        )
    if external_projectors is not None and not external_projectors:
        raise ValueError(
            "`external_projectors` is empty; the per-element orbital tables must "
            "contain at least one element entry."
        )
    if external_projectors_path is not None and not str(external_projectors_path).strip():
        raise ValueError(
            "`external_projectors_path` is blank; it must point at the projector directory."
        )
    given = [
        name
        for name, value in (
            ("external_projectors_path", external_projectors_path),
            ("external_projectors", external_projectors),
        )
        if value is not None
    ]
    if external and len(given) < 2:
        missing = sorted({"external_projectors_path", "external_projectors"} - set(given))
        raise ValueError(
            "Projection type 'atomic_projectors_external' requires both "
            "`external_projectors_path` (the directory holding one "
            "`<element>.dat` per element) and `external_projectors` (the "
            f"per-element orbital tables); missing: {missing}."
        )
    if not external and given:
        raise ValueError(
            "External projector inputs were given without any "
            f"'atomic_projectors_external' block: {given}; they would be "
            "silently ignored."
        )


# The wannier90 keywords that constrain which bands the disentangled subspace
# is built from: the outer/frozen energy windows, the projectability-based
# frozen-manifold selection, and the k-point spheres that localize where
# disentanglement acts. ``dis_num_iter`` and the other ``dis_*`` convergence
# knobs deliberately do not count -- they tune the minimization without
# constraining the manifold.
_DISENTANGLEMENT_CONSTRAINT_KEYS = frozenset(
    {
        "dis_win_min",
        "dis_win_max",
        "dis_froz_min",
        "dis_froz_max",
        "dis_froz_proj",
        "dis_proj_min",
        "dis_proj_max",
        "dis_spheres",
        "dis_spheres_num",
        "dis_spheres_first_wann",
    }
)

# The resolved frozen types whose protocol handling writes a constraint key
# into the wannier90 parameters (``dis_froz_max`` and/or the
# ``dis_froz_proj`` / ``dis_proj_*`` trio). ``ENERGY_AUTO`` is absent on
# purpose: it computes its window at workchain runtime without setting any
# parameter key, so the pre-dispatch check below cannot see it.
_WINDOW_SETTING_FROZEN_TYPES = frozenset(
    {
        WannierFrozenType.ENERGY_FIXED,
        WannierFrozenType.PROJECTABILITY,
        WannierFrozenType.FIXED_PLUS_PROJECTABILITY,
    }
)


class UnconstrainedDisentanglementWarning(UserWarning):
    """A disentangling block carries no window or frozen-manifold constraint.

    With ``num_bands > num_wann`` and none of the
    ``dis_win_* / dis_froz_* / dis_proj_* / dis_spheres_*`` keywords set,
    wannier90 free-minimizes over every included band and may silently swap
    the extra bands into the Wannier manifold.
    """


def _warn_unconstrained_disentanglement(label: str, num_bands: int, num_wann: int) -> None:
    """Emit the :class:`UnconstrainedDisentanglementWarning` for one block.

    The advice names only the keywords the bundled wannier90 accepts
    (``dis_win_*`` / ``dis_froz_*`` / ``dis_spheres_*``); the
    projectability keys silence the check when a protocol sets them but are
    not suggested.
    """
    warnings.warn(
        f"Block '{label}' includes num_bands = {num_bands} bands for "
        f"num_wann = {num_wann} Wannier functions but sets no disentanglement "
        "constraint (dis_win_min/dis_win_max, dis_froz_min/dis_froz_max or "
        "dis_spheres_*): the Wannierized manifold will be chosen by spread "
        "minimization over all included bands.",
        UnconstrainedDisentanglementWarning,
        stacklevel=3,
    )


def _disentanglement_unconstrained(
    block: ProjectionBlock,
    wannier90_overrides: dict[str, Any] | None,
    electronic_type: ElectronicType,
) -> bool:
    """Decide pre-dispatch whether a block will free-minimize its disentanglement.

    Mirror what the merged wannier90 parameters will contain without
    building them (the nested per-block builder only runs after dispatch):
    the flat ``wannier90`` overrides contribute their keys verbatim, and
    the protocol contributes a window exactly when upstream's resolved
    frozen type is one of :data:`_WINDOW_SETTING_FROZEN_TYPES`.
    """
    if int(block["num_bands"]) <= int(block["num_wann"]):
        return False
    override_keys = {str(key).lower() for key in (wannier90_overrides or {})}
    if _DISENTANGLEMENT_CONSTRAINT_KEYS & override_keys:
        return False
    try:
        _, _, frozen_type = guess_wannier_projection_types(
            electronic_type, block["projection_type"], None, None
        )
    except ValueError:
        # The per-block builder will raise the same complaint with full
        # context; don't warn on top of an imminent failure.
        return False
    return frozen_type not in _WINDOW_SETTING_FROZEN_TYPES


#: Largest ``max_kn |E_up - E_down|``, in eV, at which a two-channel nscf still
#: counts as one channel for a block that names none. A closed-shell run forced
#: to nspin=2 leaves the channels apart by its scf convergence noise (4e-8 eV on
#: the ZnO DFPT chain), while the smallest exchange splitting worth reading off
#: a band structure is two orders above this value.
_DEGENERATE_CHANNEL_TOLERANCE = 1.0e-3


def _block_eigenvalues(label: str, spin: SpinChannel, nscf_bands: orm.BandsData) -> np.ndarray:
    """Return one block's own eigenvalues as a ``(nkpoints, nbands)`` array.

    A collinear nscf emits one array per spin channel, and the block's own
    channel selects between them. A block that names no channel keeps the
    up channel when the pair satisfies ``max_kn |E_up - E_down| <=
    _DEGENERATE_CHANNEL_TOLERANCE`` — a closed-shell scratch that carries
    two channels only because kcw.x demands nspin=2, where up is also the
    channel pw2wannier90 and wannier90 read. Channels further apart than
    that make the block's manifold ambiguous, so that pairing raises
    instead of picking a half.
    """
    eigenvalues = np.asarray(nscf_bands.get_bands(), dtype=float)
    if eigenvalues.ndim < 3:
        return eigenvalues
    # A graph body receives the block's fields as provenance-tagged proxies,
    # and a stored block round-trips its channel back as a plain string; both
    # answer the constructor by value, so normalise once and read the member.
    spin = SpinChannel(spin)
    index = {SpinChannel.UP: 0, SpinChannel.DOWN: 1}.get(spin)
    if index is not None:
        return eigenvalues[index]
    if spin == SpinChannel.NONE and eigenvalues.shape[0] == 2:
        split = float(np.abs(eigenvalues[0] - eigenvalues[1]).max())
        if split <= _DEGENERATE_CHANNEL_TOLERANCE:
            return eigenvalues[SpinChannel.NONE.axis]
        raise ValueError(
            f"Block '{label}' names no spin channel (spin = '{spin.value}'), but the two "
            f"channels of the eigenvalues it was given differ by up to {split:.6f} eV, "
            f"above the {_DEGENERATE_CHANNEL_TOLERANCE} eV at which they count as one. "
            "Set the block's spin to 'up' or 'down', or pass that channel's own "
            "eigenvalues."
        )
    raise ValueError(
        f"Block '{label}' names no spin channel (spin = '{spin.value}'), but the "
        f"eigenvalues it was given are spin-resolved ({eigenvalues.shape[0]} "
        "channels). Give the block the channel it belongs to, or pass that "
        "channel's own eigenvalues."
    )


class FrozenWindowError(ValueError):
    """A block's frozen window freezes more bands than it Wannierises.

    One piece of user advice, one class: this one exists so the koopmans
    package can advise adjusting the ``dis_froz_*`` thresholds.
    Subclassing ``ValueError`` keeps every existing handler catching;
    ``label`` names the offending block.
    """

    def __init__(self, message: str, *, label: str | None = None) -> None:
        """Store ``message`` and, when known, the offending block's ``label``."""
        super().__init__(message)
        self.label = label


def validate_frozen_window(
    label: str,
    parameters: dict[str, Any],
    spin: SpinChannel,
    nscf_bands: orm.BandsData | None,
) -> None:
    """Reject a frozen window that freezes more bands than the block Wannierises.

    wannier90 stops unless at most ``num_wann`` bands are frozen at every
    k-point. Numbering the bands the block reads (those left after
    ``exclude_bands``) 1, 2, 3, ... and counting a band as frozen when
    ``dis_froz_min <= E <= dis_froz_max`` (an unset minimum being -inf),
    the condition on the window is::

        dis_froz_max < min_k E(num_wann + 1, k)

    where the numbering skips whatever sits below ``dis_froz_min``. Only a
    disentangling block keeps a window — the others have theirs stripped —
    so a hand-written value reaches exactly one block, per spin channel if
    written that way, and is either right for that block or wrong. Wrong is
    rejected here, naming the block and the largest value that would work,
    rather than left to wannier90, which stops mid-run blaming the window
    without saying which block it was reading or by how much it was over.
    The window is never adjusted: it is the user's choice, made against the
    band structure they can see.

    ``parameters`` is the block's fully merged ``.win`` set, so ``num_wann``
    and ``exclude_bands`` are read from the same place wannier90 reads them.
    Without eigenvalues (``nscf_bands`` is None, which only happens when the
    caller brought its own nscf scratch and no bands) there is nothing to
    check against and the window passes unexamined.
    """
    froz_max = parameters.get("dis_froz_max")
    if froz_max is None or nscf_bands is None:
        return
    froz_min = parameters.get("dis_froz_min")
    num_wann = int(parameters["num_wann"])

    eigenvalues = _block_eigenvalues(label, spin, nscf_bands)
    excluded = {int(band) for band in parameters.get("exclude_bands") or ()}
    if excluded:
        kept = [index for index in range(eigenvalues.shape[1]) if index + 1 not in excluded]
        eigenvalues = eigenvalues[:, kept]

    inside = eigenvalues <= float(froz_max)
    if froz_min is not None:
        inside &= eigenvalues >= float(froz_min)
    frozen = inside.sum(axis=1)
    worst = int(np.argmax(frozen))
    if int(frozen[worst]) <= num_wann:
        return

    # The largest window top that would work everywhere: at each k-point
    # the window must stop below the band that would be its
    # (num_wann + 1)-th frozen one, counting up from ``dis_froz_min``.
    limits = []
    for row in eigenvalues:
        row = np.sort(row)
        if froz_min is not None:
            row = row[row >= float(froz_min)]
        if row.size > num_wann:
            limits.append(float(row[num_wann]))
    window = f"dis_froz_max = {froz_max}"
    if froz_min is not None:
        window = f"dis_froz_min = {froz_min}, {window}"
    raise FrozenWindowError(
        f"Block '{label}' Wannierises num_wann = {num_wann} bands, but its frozen "
        f"window ({window}) freezes {int(frozen[worst])} of the bands it reads at "
        f"k-point {worst + 1} of {eigenvalues.shape[0]}. wannier90 accepts at most "
        f"num_wann frozen bands at every k-point: lower dis_froz_max below "
        f"{min(limits):.6f} eV, or give the block more Wannier functions.",
        label=label,
    )


def _warn_unconstrained_blocks(
    blocks: list[ProjectionBlock],
    wannier90_overrides: dict[str, Any] | None,
    electronic_type: ElectronicType,
    split: bool,
) -> None:
    """Warn pre-dispatch for every block that will free-minimize its disentanglement.

    Called from the eager :func:`WannierizeBlocks` body so the warning
    surfaces at build time in the caller's terminal: the nested per-block
    body (which sees the fully merged parameters) only runs daemon-side.
    Split-mode blocks are exempt by construction — their sub-blocks are
    derived at runtime, after the bands step.
    """
    if split:
        return
    for block in blocks:
        if _disentanglement_unconstrained(block, wannier90_overrides, electronic_type):
            _warn_unconstrained_disentanglement(
                block["label"], block["num_bands"], block["num_wann"]
            )


class WannierizeOverrides(TypedDict, total=False):
    """Flat, semantic overrides for :func:`WannierizeBlocks` / :func:`WannierizeBlock`.

    Deliberately NOT the upstream namespace-mirroring override shape
    (``wannier90.wannier90.parameters...``): that nesting stutters, is easy
    to mis-wrap, and a wrong depth is silently ignored by
    ``recursive_merge``. The upstream builder shape is produced in exactly
    one place — the builder call inside :func:`WannierizeBlock`.

    * ``scf`` / ``nscf`` — ``PwBaseWorkChain``-protocol override dicts for
      the shared scf/nscf pair (upstream shape, consumed verbatim by
      :func:`RunWannierGroundState`).
    * ``wannier90`` — a flat ``.win`` keyword dict (e.g.
      ``{"dis_froz_max": 10.6}``) applied to every block's wannier90.
    * ``pw2wannier90`` — a flat ``INPUTPP`` keyword dict (e.g.
      ``{"write_unk": True}``) applied to every block's pw2wannier90.
    """

    scf: dict[str, Any]
    nscf: dict[str, Any]
    wannier90: dict[str, Any]
    pw2wannier90: dict[str, Any]


class WannierizeBlockOutputs(TypedDict):
    """The flat per-block contract, and the entry shape of ``blocks``.

    Every :func:`WannierizeBlocks` mode declares this socket set per block,
    and a field never changes meaning per route: anything that would (the
    whole-block run's folders, whose product files describe the pre-split
    gauge on the split route) is populated on the plain route only. If a
    split consumer someday needs the whole-block artifacts, they get
    explicitly named new optional fields — never overloaded ones.

    Always populated, always the entry's *final* gauge:

    * ``u_file`` / ``hr_file`` / ``centres_file`` -- the gauge-product trio
      (``aiida_u.mat`` / ``aiida_hr.dat`` / ``aiida_centres.xyz``):
      extracted from the wannier90 ``retrieved`` folder for a
      plainly-Wannierised block, merged block-diagonally from the per-group
      runs for a split one.
    * ``nnkp_file`` -- the ``aiida.nnkp`` SinglefileData from the ``-pp``
      run (gauge-independent, hence shared by both routes).
    * ``output_parameters`` -- the parsed wannier90 output Dict, holding at
      least the per-WF ``wannier_functions_output`` table (spreads /
      centres, 1-based block-wide ``wf_ids``) and ``number_wfs``: the
      producing run's Dict for a plainly-Wannierised block, the per-group
      parsed outputs concatenated in band order for a split one (whose
      merged Dict carries only the honestly mergeable keys).
    * ``interpolated_bands`` -- the block's Wannier-interpolated band
      structure along the caller's ``interpolation_kpoints``, threaded from
      the wannier90 parser: the producing run's parse for a
      plainly-Wannierised block, the per-group interpolated bands
      concatenated in group order (the merged block-diagonal Hamiltonian's
      own band structure) for a split one. Populated only when
      ``interpolation_kpoints`` was given (wannier90 interpolates only
      under ``bands_plot``).

    Populated on the plain route only, because on the split route no folder
    of the final gauge exists (the whole-block run's folders describe the
    pre-split gauge, and a field never changes meaning). Split entries
    leave them unpopulated (consumers read ``None`` at runtime, not a
    ``KeyError``), so e.g. a decompose-style consumer reading ``retrieved``
    off a split entry fails loudly instead of silently working with the
    pre-split gauge. All current readers of these fields (``RunDFPT``,
    ``FoldToSupercell``, the decompose dataset route) are fed by plain-mode
    ``WannierizeBlocks`` calls, so none is affected:

    * ``retrieved`` -- the wannier90 ``retrieved`` FolderData (holds
      ``aiida_hr.dat``, ``aiida.chk``, ``aiida_u.mat``, ``aiida_centres.xyz``
      and, when the block disentangles, ``aiida_u_dis.mat``).
    * ``remote_folder`` -- the wannier90 ``RemoteData`` scratch.
    * ``wannier90_parameters`` -- the resolved ``.win`` parameter Dict fed
      to the wannier90 run (protocol defaults, user overrides and the
      per-block structural keys merged). The split route consumes the
      whole-block run's copy internally — it seeds the sub-block
      convergence settings — and leaves the entry field unpopulated (the
      set describes the pre-split run).
    """

    u_file: orm.SinglefileData
    hr_file: orm.SinglefileData
    centres_file: orm.SinglefileData
    nnkp_file: orm.SinglefileData
    output_parameters: orm.Dict
    retrieved: NotRequired[orm.FolderData]
    remote_folder: NotRequired[orm.RemoteData]
    wannier90_parameters: NotRequired[orm.Dict]
    interpolated_bands: NotRequired[orm.BandsData]


class CollectedWannierFunctions(TypedDict):
    """Outputs of :func:`collect_wannier_functions`.

    * ``centres`` -- per-WF centres as ``[x, y, z]`` lists (Å), band-ordered
      across all blocks. Coordinates the upstream parser could not read are
      ``None`` (it None-pads individually), so consumers that need numbers
      must check.
    * ``spreads`` -- per-WF final-state spreads (Å²), same ordering. Strict:
      a block without a parsed final-state spread table is rejected.
    """

    centres: list
    spreads: list


class WannierizeBlocksOutputs(TypedDict):
    """Outputs of :func:`WannierizeBlocks`.

    * ``blocks`` -- a dynamic namespace keyed by block label; every entry is
      the uniform :class:`WannierizeBlockOutputs` set, identical across
      modes, consumable downstream as a namespace.
    * ``centres`` / ``spreads`` -- the unified, band-ordered per-WF arrays of
      :class:`CollectedWannierFunctions`, concatenated across all blocks in
      input-list order (every downstream code wants the unified view);
      final-gauge in both modes.
    * ``nscf`` -- the shared nscf :class:`PwOutputs` so the supercell fold
      can read ``nscf["remote_folder"]`` (the nscf scratch every block was
      built on). Populated whenever an nscf actually ran here: the internal
      scf + nscf pair, or the nscf-only run off a caller's own
      ``scf_remote_folder`` (see that argument). Absent only when the
      caller supplied its own ``nscf_remote_folder`` and no nscf ran here
      either.
    * ``scf_remote_folder`` -- the internal scf's own ``RemoteData``,
      populated only when the internal scf + nscf pair ran here (never
      alongside a caller-supplied ``scf_remote_folder`` or
      ``nscf_remote_folder``, which already own that node). Feeds a
      second :func:`WannierizeBlocks` call's own ``scf_remote_folder``
      input.
    * ``bands`` -- the pw.x ``bands`` run along the input k-path, off
      whichever scf density this call had -- the internal scf, or a
      caller's ``scf_remote_folder`` (with or without ``nscf_remote_folder``
      alongside it) -- as a :class:`PwOutputs` namespace (``output_band``
      holds the explicit eigenvalues the Wannier interpolation is judged
      against). One run serves both modes: split mode always runs it along
      ``bands_kpoints`` (the group detection reads its eigenvalues), plain
      mode runs it along ``interpolation_kpoints`` when given. Absent with
      an external ``nscf_remote_folder`` and no ``scf_remote_folder``,
      which carries no scf density to run it from.
    * ``projwfc`` -- the projected DOS computed off the ``bands`` run's
      scratch (:class:`~aiida_koopmans.workgraphs.wannier90.ProjwfcOutputs`,
      projections resolved along the path). Present exactly when the
      ``bands`` run exists, ``codes`` carries a ``projwfc`` code, and
      every pseudo carries ``PP_PSWFC`` atomic wavefunctions
      (:func:`~aiida_koopmans.workgraphs.wannier90.projected_dos_supported`,
      which skips with a warning otherwise).
    * ``groups`` -- split mode only: the detected 1-indexed band groups
      (global indices).
    * ``orbitals`` -- one
      :class:`~aiida_koopmans.variational_orbitals.VariationalOrbital`
      per Wannier function in per-channel iwann order (occupied ascending
      then empty ascending — the kcw.x / kcp.x orbital order), with
      ``manifold`` = the block label, ``filled`` / ``spin`` from the
      block, and ``group_id`` encoding the coarsest screening partition
      consistent with the blocks' exact splits (filling, spin and
      manifold — :func:`~aiida_koopmans.workgraphs.variational_orbitals.initial_orbital_partition`).
      Emitted only when every input block carries a ``filled`` occupancy
      stamp; identical across plain and split modes (a split block's
      orbitals keep the parent block's label as their ``manifold``).
      Engine storage returns the list with each orbital's ``spin``
      degraded to a plain ``str`` — consumers compare with ``==``, never
      ``is`` (see :class:`~aiida_koopmans.variational_orbitals.VariationalOrbital`).
    """

    blocks: Annotated[dict, dynamic(WannierizeBlockOutputs)]
    centres: list
    spreads: list
    nscf: NotRequired[PwOutputs]
    scf_remote_folder: NotRequired[orm.RemoteData]
    bands: NotRequired[PwOutputs]
    projwfc: NotRequired[ProjwfcOutputs]
    groups: NotRequired[list[list[int]]]
    orbitals: NotRequired[list[VariationalOrbital]]


def _builder_overrides(overrides: WannierizeOverrides) -> dict[str, Any] | None:
    """Wrap the flat keyword dicts into the upstream builder override shape.

    The ONLY place the upstream override nesting is produced. The protocol
    overrides mirror the workchain's input namespace tree: base-workchain
    namespace -> calculation namespace -> ``parameters`` — hence
    ``wannier90.wannier90.parameters`` for ``.win`` keywords and
    ``pw2wannier90.pw2wannier90.parameters.INPUTPP`` for the pw2wannier90
    namelist. Callers supply the flat :class:`WannierizeOverrides` and never
    touch this shape.

    ``scf`` / ``nscf`` are already in upstream shape (they feed
    :func:`~aiida_koopmans.workgraphs.wannier_ground_state.RunWannierGroundState` verbatim) and pass
    through unwrapped: :meth:`Wannier90WorkChain.get_builder_from_protocol`
    reads them at this same top-level key and forwards them into its own
    nested ``PwBaseWorkChain.get_builder_from_protocol`` calls. Without this,
    a pseudo family that recommends no cutoffs crashes that nested call even
    when the caller's ``ecutwfc``/``ecutrho`` are sitting right there in
    ``scf``/``nscf`` — :func:`WannierizeBlock` discards both namespaces
    afterwards (it reuses the shared nscf scratch), so this is the only use
    the cutoffs get.
    """
    scf = overrides.get("scf")
    nscf = overrides.get("nscf")
    wannier90 = overrides.get("wannier90")
    pw2wannier90 = overrides.get("pw2wannier90")
    builder_overrides: dict[str, Any] = {}
    if scf:
        builder_overrides["scf"] = scf
    if nscf:
        builder_overrides["nscf"] = nscf
    if wannier90:
        builder_overrides["wannier90"] = {"wannier90": {"parameters": dict(wannier90)}}
    if pw2wannier90:
        builder_overrides["pw2wannier90"] = {
            "pw2wannier90": {"parameters": {"INPUTPP": dict(pw2wannier90)}}
        }
    return builder_overrides or None


@task.calcfunction(outputs=["u_file", "hr_file", "centres_file"])
def extract_wannier_output_files(retrieved: orm.FolderData) -> dict:
    """Pull the gauge-product trio out of a wannier90 ``retrieved`` folder.

    Wraps ``aiida_u.mat`` / ``aiida_hr.dat`` / ``aiida_centres.xyz`` as
    individual :class:`~aiida.orm.SinglefileData` nodes so a plainly
    Wannierised block exposes the same file sockets as a split one. The
    files exist because :func:`WannierizeBlock` pins ``write_hr`` /
    ``write_u_matrices`` / ``write_xyz`` (upstream's default retrieve
    suffixes then pick them up). A calcfunction (not a plain ``@task``): it
    takes an AiiDA data node, which the PyFunction deserializer refuses.
    """
    import io

    def _single(filename: str) -> orm.SinglefileData:
        if filename not in retrieved.base.repository.list_object_names():
            raise ValueError(
                f"``{filename}`` is missing from the wannier90 retrieved folder. "
                "The wannier90 run must set ``write_hr = True``, "
                "``write_u_matrices = True`` and ``write_xyz = True``."
            )
        content = retrieved.base.repository.get_object_content(filename, mode="rb")
        return orm.SinglefileData(io.BytesIO(content), filename=filename)

    return {
        "u_file": _single("aiida_u.mat"),
        "hr_file": _single("aiida_hr.dat"),
        "centres_file": _single("aiida_centres.xyz"),
    }


@task.calcfunction
def emit_wannier90_parameters(parameters: orm.Dict) -> orm.Dict:
    """Emit the block's merged wannier90 parameters as an explicit socket.

    The merged set (protocol defaults, flat user overrides and the
    per-block structural keys) both feeds the wannier90 step and rides out
    of the graph as the ``wannier90_parameters`` output — a graph output
    must be a task socket, and this hop makes the resolved Dict one, so
    consumers (the split route's sub-block seeding) read the explicit
    parameters instead of walking provenance.
    """
    return orm.Dict(parameters.get_dict())


def _apply_block_disentanglement(
    block: ProjectionBlock,
    w90_params: dict[str, Any],
    wannier90_overrides: dict[str, Any] | None,
    nscf_bands: orm.BandsData | None,
) -> None:
    """Edit one block's merged ``.win`` set for its disentanglement, in place.

    A block with extra bands (``num_bands > num_wann``) genuinely
    disentangles, so give it wannier90's real default iteration budget (the
    aiida-wannier90-workflows protocol pins ``dis_num_iter: 0``, which
    freezes the initial projection subspace) and check its frozen window
    against ``nscf_bands`` (:func:`validate_frozen_window`); a block with
    ``num_bands == num_wann`` cannot disentangle, so strip the (globally
    supplied) windows outright. ``w90_params`` must already carry the
    block's ``num_wann`` / ``num_bands``.
    """
    num_bands, num_wann = int(w90_params["num_bands"]), int(w90_params["num_wann"])
    if num_bands != num_wann:
        if "dis_num_iter" not in (wannier90_overrides or {}):
            w90_params["dis_num_iter"] = 5000
        validate_frozen_window(str(block["label"]), w90_params, block["spin"], nscf_bands)
        # ``w90_params`` at this point holds the full merge (protocol defaults
        # plus the flat ``wannier90`` overrides), so any window the protocol
        # itself set counts as a constraint too. On the production path this
        # body runs daemon-side, where the warning lands in the worker log
        # only; the user-visible copy is emitted pre-dispatch from
        # :func:`WannierizeBlocks`.
        if not _DISENTANGLEMENT_CONSTRAINT_KEYS.intersection(key.lower() for key in w90_params):
            _warn_unconstrained_disentanglement(block["label"], num_bands, num_wann)
    else:
        for key in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"):
            w90_params.pop(key, None)


@task.graph
def WannierizeBlock(
    codes: WannierizeBlockCodes,
    structure: orm.StructureData,
    block: ProjectionBlock,
    projection_type: WannierProjectionType,
    nscf_remote_folder: orm.RemoteData,
    kpoints: orm.KpointsData,
    mp_grid: list[int] | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
    parallelization: ParallelizationDict | None = None,
    nscf_bands: orm.BandsData | None = None,
    interpolation_kpoints: orm.KpointsData | None = None,
    external_projectors_path: str | None = None,
    external_projectors: dict[str, Any] | None = None,
) -> WannierizeBlockOutputs:
    """Wannierise a single projection block off the shared nscf scratch.

    ``overrides`` is the flat :class:`WannierizeOverrides`; this block-level
    graph consumes its ``wannier90`` / ``pw2wannier90`` entries
    (the ``scf`` / ``nscf`` entries belong to the shared scf+nscf pair and
    are ignored here). This function is the single place the flat keyword
    dicts are wrapped into the upstream builder's namespace-mirroring
    override shape.

    An ``atomic_projectors_external`` block needs both external projector
    inputs (see :func:`validate_external_projector_inputs`): the upstream
    builder turns them into the pw2wannier90 step's projector-directory
    ``RemoteData`` plus the ``atom_proj_ext`` namelist keywords, and sizes
    the projector count from the orbital tables.

    Seeds a ``Wannier90WorkChain`` builder via ``get_builder_from_protocol``
    for this block's ``projection_type``, then:

    * pops the ``scf`` namespace and the ``nscf`` namespace so the workchain
      skips both steps (upstream gates each on ``"scf" in inputs`` /
      ``"nscf" in inputs``), and points the pw2wannier90 step at the shared
      nscf scratch via ``pw2wannier90.pw2wannier90.parent_folder`` -- the only
      parent the validator accepts once both scf and nscf are absent;
    * overrides the per-block ``num_wann`` / ``num_bands`` / ``exclude_bands``
      (and ``projections`` for explicit blocks) from
      :func:`block_w90_kwargs`;
    * forces ``write_hr`` / ``write_u_matrices`` / ``write_xyz`` so
      ``aiida_hr.dat`` / ``aiida_u.mat`` / ``aiida_u_dis.mat`` /
      ``aiida_centres.xyz`` are written (upstream's default retrieve list then
      picks them up), and force-retrieves ``aiida.chk``, which upstream
      excludes by default;
    * checks a disentangling block's frozen window against ``nscf_bands`` and
      rejects one that would freeze more bands than the block Wannierises,
      which wannier90 refuses (:func:`validate_frozen_window`);
    * when ``interpolation_kpoints`` (a labelled explicit-path
      ``KpointsData``) is given, sets ``bands_plot = True`` and feeds the
      path to the wannier90 ``bands_kpoints`` input, so the run interpolates
      its band structure along it and the parsed ``interpolated_bands``
      rides out as an output. The path renders as an ``explicit_kpath``
      block, which needs wannier90 4.0 or newer (releases up to 3.1.0
      reject it).
    """
    validate_projection_type(projection_type)
    validate_external_projector_inputs(
        [projection_type], external_projectors_path, external_projectors
    )
    overrides = overrides or {}
    wannier90 = overrides.get("wannier90")

    # ``.build()`` executes this body eagerly, where graph inputs arrive as
    # provenance-tagged proxies; the family label ends up bound as an SQL
    # parameter inside ``get_builder_from_protocol``, which needs a plain str
    # — as does the projector directory, which becomes a ``RemoteData``
    # remote path. The two enums the builder forwards into ``PwBaseWorkChain``,
    # whose branches test them with ``is``, are coerced to match the other
    # builder calls; this block discards those pw namespaces below, so here
    # the coercion changes nothing on its own.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None
    if external_projectors_path is not None:
        external_projectors_path = str(external_projectors_path)

    builder = Wannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        protocol=protocol,
        overrides=_builder_overrides(overrides),
        pseudo_family=pseudo_family,
        electronic_type=unwrap_enum(electronic_type, ElectronicType),
        spin_type=unwrap_enum(spin_type, SpinType),
        projection_type=projection_type,
        external_projectors_path=external_projectors_path,
        external_projectors=external_projectors,
        # For koopmans we do not exclude semicore states automatically.
        exclude_semicore=False,
        # The hamiltonian-retrieval protocol override sets ``write_hr`` /
        # ``write_tb`` and the hr retrieve handling.
        retrieve_hamiltonian=True,
        print_summary=False,
    )
    # Flatten to a plain dict up front; every edit below is a dict edit.
    data = get_dict_from_builder(builder)
    w90 = data["wannier90"]["wannier90"]

    # --- per-block wannier90 parameters / projections ---
    w90_kwargs = block_w90_kwargs(block)
    w90_params = w90["parameters"].get_dict()
    w90_params["num_wann"] = w90_kwargs["num_wann"]
    w90_params["num_bands"] = w90_kwargs["num_bands"]
    if "exclude_bands" in w90_kwargs:
        w90_params["exclude_bands"] = w90_kwargs["exclude_bands"]
    else:
        # A block that excludes nothing must not inherit an exclusion the
        # protocol machinery may have seeded.
        w90_params.pop("exclude_bands", None)
    _apply_block_disentanglement(block, w90_params, wannier90, nscf_bands)
    # ``write_hr`` is set by the retrieve_hamiltonian override above; pin it
    # explicitly so a stripped-down override dict can't silently drop it.
    # ``write_u_matrices`` / ``write_xyz`` produce the U matrices and Wannier
    # centres that pw2wannier90 ``wan_mode='decompose'`` and the wannierjl
    # split consume.
    w90_params.update(
        owned("wannier90", {"write_hr": True, "write_u_matrices": True, "write_xyz": True})
    )
    # The protocol builder froze ``mp_grid`` from its own distance-derived
    # mesh, which goes stale once the shared k-list is substituted below.
    # Pin the real mesh dimensions when given (wannier90 cannot re-derive
    # them from an explicit list); otherwise drop the key so a mesh
    # ``kpoints`` input lets the calculation re-derive it.
    if mp_grid is not None:
        w90_params["mp_grid"] = mp_grid
    else:
        w90_params.pop("mp_grid", None)
    # Wannier band interpolation: wannier90 writes ``aiida_band.dat`` (which
    # the parser turns into ``interpolated_bands``) only under ``bands_plot``,
    # and the calculation validator requires the path alongside it, so the
    # pair travels together.
    if interpolation_kpoints is not None:
        require_path_labels(interpolation_kpoints, "interpolation_kpoints")
        w90_params["bands_plot"] = True
        w90["bands_kpoints"] = interpolation_kpoints
    # Route the merged parameters through a task so the same node both feeds
    # the wannier90 step and rides out as the ``wannier90_parameters``
    # output (a graph output must be a task socket).
    w90_parameters = emit_wannier90_parameters(
        parameters=w90_params, metadata={"call_link_label": "emit_wannier90_parameters"}
    ).result
    w90["parameters"] = w90_parameters

    # Explicit (ANALYTIC) blocks carry resolved projection orbitals; automatic
    # blocks rely on ``projection_type`` alone (no ``projections`` key).
    if "projections" in w90_kwargs:
        w90["projections"] = orm.List(list=w90_kwargs["projections"])

    # Share the nscf k-mesh so the per-block wannier90 / pw2wannier90 read
    # eigenstates on the exact grid the shared nscf produced.
    w90["kpoints"] = kpoints

    # Force-retrieve ``aiida.chk`` (upstream's only non-default product), merged
    # on top of whatever ``settings`` the protocol set; the workchain only adds
    # its own ``postproc_setup`` key on top of this.
    existing_settings: dict = {}
    if "settings" in w90:
        existing_settings = w90["settings"].get_dict()
    existing_settings.update(_W90_RETRIEVE_SETTINGS)
    w90["settings"] = orm.Dict(existing_settings)

    # Skip scf + nscf and reuse the shared nscf scratch. With both namespaces
    # absent the workchain validator requires the parent on the pw2wannier90
    # step.
    data.pop("scf", None)
    data.pop("nscf", None)
    data.pop("clean_workdir", None)
    data["pw2wannier90"]["pw2wannier90"]["parent_folder"] = nscf_remote_folder

    # Per-code parallelization: wannier90.x takes ntasks only (no pool/pd
    # concept); pw2wannier90.x takes ntasks plus -npool / -pd. QE rejects
    # pw2wannier90 pools under gamma_only, but this block wannierization is a
    # periodic (full-grid nscf) path, so no schema guard is needed here.
    merge_parallelization_into_inputs(data["wannier90"]["wannier90"], parallelization, "wannier90")
    merge_parallelization_into_inputs(
        data["pw2wannier90"]["pw2wannier90"], parallelization, "pw2wannier90"
    )

    # The workchain carries a label through to its pw2wannier90 step; its
    # two wannier90.x steps get their own labels via the ``wannier90`` /
    # ``wannier90_pp`` namespaces (see ``Wannierize``).
    name_step(data["pw2wannier90"], "Overlaps")
    name_step(data["pw2wannier90"]["pw2wannier90"], "Overlaps")
    name_step(data["wannier90"], "Minimization")
    name_step(data["wannier90"]["wannier90"], "Minimization")
    stamp_wannier90_pp_namespace(data)
    data.setdefault("metadata", {})["call_link_label"] = "wannier90"
    outputs = Wannier90Step(**data)

    output_files = extract_wannier_output_files(
        retrieved=outputs["wannier90"]["retrieved"],
        metadata={"call_link_label": "extract_wannier_output_files"},
    )

    block_outputs = WannierizeBlockOutputs(
        u_file=output_files["u_file"],
        hr_file=output_files["hr_file"],
        centres_file=output_files["centres_file"],
        retrieved=outputs["wannier90"]["retrieved"],
        remote_folder=outputs["wannier90"]["remote_folder"],
        nnkp_file=outputs["wannier90_pp"]["nnkp_file"],
        output_parameters=outputs["wannier90"]["output_parameters"],
        wannier90_parameters=w90_parameters,
    )
    if interpolation_kpoints is not None:
        block_outputs["interpolated_bands"] = outputs["wannier90"]["interpolated_bands"]
    return block_outputs


@task
def collect_wannier_functions(
    output_parameters: Annotated[dict, dynamic(orm.Dict)],
) -> CollectedWannierFunctions:
    """Concatenate per-block parsed wannier90 outputs into unified arrays.

    Walks each block's ``output_parameters`` (arriving as plain dicts via
    aiida-pythonjob's built-in ``Dict`` deserializer) and concatenates the
    final-state per-WF centres and spreads from ``wannier_functions_output``
    (a list of ``{wf_ids, wf_centres, wf_spreads}`` dicts with 1-based
    ``wf_ids``; distinct from the manifold-total ``Omega_*`` scalars) into
    one band-ordered array pair. Within a block the entries are ordered by
    ``wf_ids``.

    Callers key the input namespace so that lexicographic key order is the
    band order they want out (``b00``, ``b01``, ...); the concatenation
    follows that order and nothing else. The keys carry no other meaning —
    in particular they are not block labels.
    """
    centres: list[list[float | None]] = []
    spreads: list[float] = []
    for key in sorted(output_parameters):
        parameters = output_parameters[key]
        wfs = parameters.get("wannier_functions_output") or []
        if len(wfs) != parameters.get("number_wfs"):
            raise ValueError(
                f"A block's wannier90 ``output_parameters`` lists {len(wfs)} "
                "final-state Wannier functions but the run declares "
                f"number_wfs = {parameters.get('number_wfs')}."
            )
        if any("wf_spreads" not in wf for wf in wfs):
            # A wannier90 restart-for-plotting run parses only wf_ids +
            # im_re_ratio per WF (no final-state spread table).
            raise ValueError(
                "A ``wannier_functions_output`` entry carries no ``wf_spreads`` — "
                "the run did not minimise to a final state (e.g. a "
                "restart-for-plotting run)."
            )
        for wf in sorted(wfs, key=lambda wf: int(wf["wf_ids"])):
            spreads.append(float(wf["wf_spreads"]))
            coords = wf.get("wf_centres") or (None, None, None)
            centres.append([None if c is None else float(c) for c in coords])
    return CollectedWannierFunctions(centres=centres, spreads=spreads)


def _validate_blocks(blocks: list[ProjectionBlock]) -> None:
    """Reject any block whose bookkeeping, projection type or disentanglement is unsupported."""
    for block in blocks:
        validate_projection_block(block)
        validate_projection_type(block["projection_type"])
    validate_projection_block_sequence(blocks)


def _external_kwargs_for(
    block: ProjectionBlock,
    external_projectors_path: str | None,
    external_projectors: dict[str, Any] | None,
) -> dict[str, Any]:
    """Select the external projector kwargs a block's wannier graph takes.

    The inputs feed only the blocks that consume them; a non-external
    block's builder would reject them.
    """
    if not _is_external(block["projection_type"]):
        return {}
    return {
        "external_projectors_path": external_projectors_path,
        "external_projectors": external_projectors,
    }


def _maybe_emit_orbital_partition(
    outputs: WannierizeBlocksOutputs, blocks: list[ProjectionBlock]
) -> None:
    """Wire the initial orbital partition into ``outputs`` when the blocks carry occupancy.

    A block derived from atomic projectors carries no occupancy until the
    runtime band-group detection settles it, so the partition -- which is
    a split of the Wannier functions into occupied and empty -- cannot be
    built for it here. A partial stamping is a caller bug rather than a
    smaller feature, so it raises. The partition task takes a JSON-pure
    reduced view of the blocks: a full block carries a non-``str`` enum
    (``projection_type``) that the PyFunction input serializer cannot
    store.
    """
    stamped = [("filled" in block) for block in blocks]
    if any(stamped) and not all(stamped):
        unstamped = [
            str(block["label"]) for block, has in zip(blocks, stamped, strict=True) if not has
        ]
        raise ValueError(
            "Some blocks carry a `filled` occupancy stamp and some do not "
            f"({unstamped}); stamp every block to emit the orbital partition, "
            "or none to skip it."
        )
    if not (blocks and all(stamped)):
        return
    specs = [
        ProjectionBlockId(
            label=str(block["label"]),
            spin=SpinChannel(block["spin"]),
            filled=block_occupancy(block),
            num_wann=int(block["num_wann"]),
        )
        for block in blocks
    ]
    # The partition lists orbitals in emitted order while the unified
    # ``spreads`` / ``centres`` concatenate in input-list order; a consumer
    # pairing the two positionally would mis-align silently if the orders
    # diverged, so require the input to already be the emitted order.
    ordered = ordered_block_specs(specs)
    if ordered != specs:
        raise ValueError(
            f"The blocks are not in emitted orbital order: got labels "
            f"{[s['label'] for s in specs]} but the partition walks "
            f"{[s['label'] for s in ordered]} (spin channels in SpinChannel declaration "
            "order, each contiguous, with every occupied block before every empty one). Reorder "
            "the blocks so `orbitals` stays aligned with the input-list-ordered "
            "`spreads` / `centres`."
        )
    partition = initial_orbital_partition(
        blocks=specs,
        metadata={"call_link_label": "initial_orbital_partition"},
    )
    outputs["orbitals"] = partition.result


def _reject_inputs_an_external_scratch_ignores(
    overrides: WannierizeOverrides,
    scf_kpoints: orm.KpointsData | None,
    *,
    source: str,
) -> None:
    """Reject an input to the internal scf that an external scratch skips.

    ``overrides["scf"]`` / ``overrides["nscf"]`` are not rejected here: an
    external scratch skips the *internal* scf (an ``nscf_remote_folder``
    skips the nscf too), but both namespaces still reach every per-block
    :func:`WannierizeBlock` call below (see :func:`_builder_overrides`) —
    the only way a pseudo family that recommends no cutoffs can satisfy the
    nested ``Wannier90WorkChain.get_builder_from_protocol`` call there.
    Only ``scf_kpoints`` has no consumer left once the internal scf is
    skipped. ``source`` names what was given instead, for the error.
    """
    if scf_kpoints is not None:
        raise ValueError(
            f"scf_kpoints was given together with {source}; no scf runs "
            "here, so the mesh would be silently ignored. Set it on "
            "whoever ran the scf."
        )


def _resolve_split_mode(
    blocks: list[ProjectionBlock],
    mp_grid: list[int] | None,
    nscf_remote_folder: orm.RemoteData | None,
    scf_remote_folder: orm.RemoteData | None,
    split_threshold: float | None,
    bands_kpoints: orm.KpointsData | None,
    num_occ_bands: int | None,
    wjl_options: dict[str, Any] | None,
    subblock_wannier90_options: dict[str, Any] | None,
    cubic_pw2wannier90_options: dict[str, Any] | None,
    interpolation_kpoints: orm.KpointsData | None,
) -> bool:
    """Decide split-vs-plain for :func:`WannierizeBlocks` and validate the inputs.

    Split mode triggers on the *need* for splitting: a gap threshold was
    requested (``split_threshold``), or a block's band groups are only
    discovered at runtime (an automatic-projections block — no
    ``projections`` key). Split mode requires ``bands_kpoints`` but does not
    trigger on it: it rides on both routes, setting the quality-check pw.x
    run's k-point list either way (and doubling as the split-detection
    input when split). Plain mode rejects the remaining split-only knobs
    rather than silently ignore them; every violation raises naming the
    gap. ``interpolation_kpoints`` also rides on both routes and only its
    labels are checked here. Does not validate the ``wannierjl`` code: that
    requirement lives on
    :func:`~aiida_koopmans.workgraphs.auto_wannierize.WannierizeAndSplitBlock`'s
    own ``codes`` spec, which the split branch enters unconditionally.
    """
    require_path_labels(interpolation_kpoints, "interpolation_kpoints")
    split = split_threshold is not None or any("projections" not in block for block in blocks)
    if not split:
        # bands_kpoints is not split-only: off the split path it still sets
        # the quality-check pw.x run's own k-point list, independent of
        # interpolation_kpoints (see _run_explicit_bands_and_dos_steps).
        split_only = {
            "num_occ_bands": num_occ_bands,
            "wjl_options": wjl_options,
            "subblock_wannier90_options": subblock_wannier90_options,
            "cubic_pw2wannier90_options": cubic_pw2wannier90_options,
        }
        given = [name for name, value in split_only.items() if value is not None]
        if given:
            raise ValueError(
                "Split-only inputs were given without a split trigger "
                f"(`split_threshold` or an automatic-projections block): {given}; "
                "they would be silently ignored."
            )
        return False
    if bands_kpoints is None:
        raise ValueError(
            "Split mode requires `bands_kpoints`: the band-group detection reads "
            "the eigenvalues of a pw.x bands run along it."
        )
    if num_occ_bands is None:
        raise ValueError(
            "Split mode requires `num_occ_bands`: the group detection always "
            "opens a group at the occupied/empty boundary."
        )
    if nscf_remote_folder is not None:
        raise ValueError(
            "Split mode cannot build on an external `nscf_remote_folder`: the "
            "bands step the detection reads runs off the internal scf's remote "
            "folder, and an external nscf scratch carries no scf density to run "
            "it from."
        )
    if scf_remote_folder is not None:
        raise ValueError(
            "Split mode cannot build on an external `scf_remote_folder`: the "
            "bands step the detection reads runs off the internal scf, which "
            "an external scf density skips here."
        )
    disentangling = [
        str(block["label"]) for block in blocks if int(block["num_bands"]) > int(block["num_wann"])
    ]
    if disentangling:
        raise NotImplementedError(
            f"Block(s) {disentangling} read more bands than they Wannierise, which "
            "would disentangle them; splitting a disentangled block is not "
            "supported. The split re-Wannierises each group from the parent's "
            "gauge alone, so the parent's disentanglement matrix has nowhere to "
            "go. Lower the nscf band count so each block reads exactly its "
            "`num_wann` bands, or drop `split_threshold`."
        )
    if mp_grid is None:
        raise ValueError(
            "Split mode requires `mp_grid`: the per-group re-Wannierisation "
            "writes it into each sub-block `.win`."
        )
    return True


def _run_explicit_bands_and_dos_steps(
    codes: WannierizeBlocksCodes,
    structure: orm.StructureData,
    bands_kpoints: orm.KpointsData | None,
    interpolation_kpoints: orm.KpointsData | None,
    scf_remote_folder: orm.RemoteData,
    nscf_overrides: dict[str, Any] | None,
    pseudo_family: str | None,
    protocol: str | None,
    electronic_type: ElectronicType,
    parallelization: ParallelizationDict | None,
) -> tuple[PwOutputs | None, ProjwfcOutputs | None]:
    """Assemble the pw.x quality-check bands run and its projwfc step.

    A plain graph-assembly helper: it must be called inside a
    ``@task.graph`` body. Samples ``bands_kpoints`` when given (in split
    mode this run doubles as the detection input), and falls back to
    ``interpolation_kpoints`` otherwise; without either path nothing is
    assembled. Returns the bands run as a ready-to-wire :class:`PwOutputs`
    namespace (``output_band`` holds the eigenvalues the detection and the
    quality comparison read) and — whenever every pseudo carries
    ``PP_PSWFC`` atomic wavefunctions
    (:func:`~aiida_koopmans.workgraphs.wannier90.projected_dos_supported`,
    which skips with a warning otherwise) — the chained projected-DOS
    namespace off the run's scratch
    (:func:`~aiida_koopmans.workgraphs.wannier90.RunProjwfc`, entered
    unconditionally on that predicate; a ``projwfc`` code missing from
    ``codes`` then surfaces as that graph's own structural missing-input
    error).
    """
    quality_path = bands_kpoints if bands_kpoints is not None else interpolation_kpoints
    if quality_path is None:
        return None, None
    bands_step = run_bands_step(
        pw_code=codes["pw"],
        structure=structure,
        bands_kpoints=quality_path,
        scf_remote_folder=scf_remote_folder,
        nscf_overrides=nscf_overrides,
        pseudo_family=pseudo_family,
        protocol=protocol,
        electronic_type=electronic_type,
        parallelization=parallelization,
    )
    bands_outputs = PwOutputs(
        remote_folder=bands_step["remote_folder"],
        output_parameters=bands_step["output_parameters"],
        output_band=bands_step["output_band"],
    )
    projwfc_outputs = None
    if projected_dos_supported(pseudo_family, structure):
        projwfc_outputs = RunProjwfc(
            projwfc_code=reference(codes, "projwfc"),
            parent_folder=bands_step["remote_folder"],
            protocol=protocol,
            parallelization=parallelization,
            metadata={"call_link_label": "projwfc", "label": "Atomic projections"},
        )
    return bands_outputs, projwfc_outputs


def _run_nscf_off_external_scf(
    codes: WannierizeBlocksCodes,
    structure: orm.StructureData,
    kpoints: orm.KpointsData,
    scf_remote_folder: orm.RemoteData,
    bands_kpoints: orm.KpointsData | None,
    interpolation_kpoints: orm.KpointsData | None,
    nscf_bands: orm.BandsData | None,
    overrides: WannierizeOverrides,
    pseudo_family: str | None,
    protocol: str | None,
    electronic_type: ElectronicType,
    parallelization: ParallelizationDict | None,
) -> tuple[Any, orm.RemoteData, orm.BandsData | None, PwOutputs | None, ProjwfcOutputs | None]:
    """Run a fresh nscf on ``kpoints`` off ``scf_remote_folder``, plus the quality check.

    A plain graph-assembly helper: it must be called inside a
    ``@task.graph`` body. Runs :func:`run_nscf_step`, then, whenever
    ``bands_kpoints`` or ``interpolation_kpoints`` is given, the
    quality-check bands / projwfc pair off the same density. Returns
    ``(nscf_step, nscf_scratch, block_bands, bands_outputs,
    projwfc_outputs)``.
    """
    nscf_step = run_nscf_step(
        pw_code=codes["pw"],
        structure=structure,
        nscf_kpoints=kpoints,
        scf_remote_folder=scf_remote_folder,
        nscf_overrides=overrides.get("nscf"),
        pseudo_family=pseudo_family,
        protocol=protocol,
        electronic_type=electronic_type,
        parallelization=parallelization,
    )
    nscf_scratch = nscf_step["remote_folder"]
    block_bands = nscf_bands if nscf_bands is not None else nscf_step["output_band"]
    bands_outputs, projwfc_outputs = _run_explicit_bands_and_dos_steps(
        codes=codes,
        structure=structure,
        bands_kpoints=bands_kpoints,
        interpolation_kpoints=interpolation_kpoints,
        scf_remote_folder=scf_remote_folder,
        nscf_overrides=overrides.get("nscf"),
        pseudo_family=pseudo_family,
        protocol=protocol,
        electronic_type=electronic_type,
        parallelization=parallelization,
    )
    return nscf_step, nscf_scratch, block_bands, bands_outputs, projwfc_outputs


def _reuse_external_nscf_scratch(
    codes: WannierizeBlocksCodes,
    structure: orm.StructureData,
    nscf_remote_folder: orm.RemoteData,
    scf_remote_folder: orm.RemoteData | None,
    interpolation_kpoints: orm.KpointsData | None,
    nscf_bands: orm.BandsData | None,
    overrides: WannierizeOverrides,
    pseudo_family: str | None,
    protocol: str | None,
    electronic_type: ElectronicType,
    parallelization: ParallelizationDict | None,
) -> tuple[orm.RemoteData, orm.BandsData | None, PwOutputs | None, ProjwfcOutputs | None]:
    """Wire ``nscf_remote_folder`` through; the quality check reads ``scf_remote_folder``.

    A plain graph-assembly helper: it must be called inside a
    ``@task.graph`` body. Runs no scf or nscf itself. With
    ``scf_remote_folder`` given, also runs the quality-check bands /
    projwfc pair off it. Returns ``(nscf_scratch, block_bands,
    bands_outputs, projwfc_outputs)``.
    """
    bands_outputs = None
    projwfc_outputs = None
    if scf_remote_folder is not None:
        bands_outputs, projwfc_outputs = _run_explicit_bands_and_dos_steps(
            codes=codes,
            structure=structure,
            bands_kpoints=None,
            interpolation_kpoints=interpolation_kpoints,
            scf_remote_folder=scf_remote_folder,
            nscf_overrides=overrides.get("nscf"),
            pseudo_family=pseudo_family,
            protocol=protocol,
            electronic_type=electronic_type,
            parallelization=parallelization,
        )
    return nscf_remote_folder, nscf_bands, bands_outputs, projwfc_outputs


def _detect_split_groups(
    bands_outputs: PwOutputs | None,
    num_occ_bands: int | None,
    split_threshold: float | None,
    num_bands_total: int,
) -> Any:
    """Run the band-group detection off the quality-check run's eigenvalues.

    The split machinery depends on aiida-wannierjl, so its import lives
    here, reached only on the split route; the import direction
    (auto_wannierize imports this module at module level, this helper
    imports auto_wannierize lazily) avoids the cycle.
    """
    from aiida_koopmans.workgraphs.auto_wannierize import detect_band_groups

    if bands_outputs is None:
        raise ValueError(
            "The split-mode group detection reads the quality-check bands run, "
            "which requires `bands_kpoints`; the split-mode validation should "
            "have rejected this build."
        )
    # The detection is restricted to the Wannierised manifold — the extra
    # disentanglement bands above it must not influence the grouping.
    return detect_band_groups(
        bands=bands_outputs["output_band"],
        num_occ_bands=num_occ_bands,
        threshold=split_threshold,
        num_bands_total=num_bands_total,
    )


def _wire_quality_check_outputs(
    outputs: WannierizeBlocksOutputs,
    bands_outputs: PwOutputs | None,
    projwfc_outputs: ProjwfcOutputs | None,
) -> None:
    """Wire the quality-check namespaces into ``outputs`` when their steps ran."""
    if bands_outputs is not None:
        outputs["bands"] = bands_outputs
    if projwfc_outputs is not None:
        outputs["projwfc"] = projwfc_outputs


def _wire_scf_nscf_outputs(
    outputs: WannierizeBlocksOutputs,
    scf_nscf: ScfNscfOutputs | None,
    nscf_step: Any,
    nscf_scratch: orm.RemoteData | None,
) -> None:
    """Wire the ``nscf`` / ``scf_remote_folder`` outputs when either scf mode ran here.

    ``scf_nscf`` is set only when the internal scf + nscf pair ran (also
    populating ``scf_remote_folder``, the internal scf's own density);
    ``nscf_step`` only when the ``scf_remote_folder``-alone nscf-only mode
    ran. Both are ``None`` when the caller supplied its own
    ``nscf_remote_folder`` and neither ran here.
    """
    if scf_nscf is not None:
        outputs["nscf"] = PwOutputs(
            remote_folder=cast("orm.RemoteData", nscf_scratch),
            output_parameters=scf_nscf["nscf_output_parameters"],
            output_band=scf_nscf["nscf_output_band"],
        )
        outputs["scf_remote_folder"] = scf_nscf["scf_remote_folder"]
    elif nscf_step is not None:
        outputs["nscf"] = PwOutputs(
            remote_folder=cast("orm.RemoteData", nscf_scratch),
            output_parameters=nscf_step["output_parameters"],
            output_band=nscf_step["output_band"],
        )


@task.graph
def WannierizeBlocks(
    codes: WannierizeBlocksCodes,
    structure: orm.StructureData,
    blocks: list[ProjectionBlock],
    kpoints: orm.KpointsData,
    mp_grid: list[int] | None = None,
    scf_kpoints: orm.KpointsData | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
    parallelization: ParallelizationDict | None = None,
    nscf_remote_folder: orm.RemoteData | None = None,
    scf_remote_folder: orm.RemoteData | None = None,
    nscf_bands: orm.BandsData | None = None,
    split_threshold: float | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    num_occ_bands: int | None = None,
    wjl_options: dict[str, Any] | None = None,
    subblock_wannier90_options: dict[str, Any] | None = None,
    cubic_pw2wannier90_options: dict[str, Any] | None = None,
    interpolation_kpoints: orm.KpointsData | None = None,
    external_projectors_path: str | None = None,
    external_projectors: dict[str, Any] | None = None,
) -> WannierizeBlocksOutputs:
    """Wannierise a periodic system block-by-block off one shared scf + nscf.

    A single :func:`RunWannierGroundState` runs scf + nscf once; every projection
    block is then Wannierised in its own ``Wannier90WorkChain`` that skips
    scf / nscf and reads the shared nscf scratch (``nscf["remote_folder"]``).
    The per-block fan-out is a native ``for`` loop over ``blocks`` inside this
    ``@task.graph`` body; the per-block outputs are collected into a dict
    keyed by block label and returned as the ``blocks`` dynamic namespace,
    and the per-block parsed outputs are concatenated in input-list order
    into the unified ``centres`` / ``spreads`` outputs
    (:func:`collect_wannier_functions`).

    The automated block-splitting mode triggers when a gap threshold was
    requested (``split_threshold``) or when any block uses automatic
    projections, whose band grouping only exists at runtime.
    In split mode a pw.x ``bands`` step runs along the required
    ``bands_kpoints`` off the internal scf density, the runtime
    ``detect_band_groups`` task turns the eigenvalues into energy-separated
    groups (always splitting at the occupied/empty boundary
    ``num_occ_bands``, and at every gap wider than ``split_threshold`` eV),
    and each block is handled by a nested
    :func:`~aiida_koopmans.workgraphs.auto_wannierize.WannierizeAndSplitBlock`
    that receives the resolved groups and splits the block when they divide
    it. Either way every ``blocks`` entry emits the same
    :class:`WannierizeBlockOutputs` socket set — one final-gauge
    ``output_parameters`` per block included — so downstream consumers
    never branch on the mode and the unified ``centres`` / ``spreads``
    are collected unconditionally.

    Args:
        codes: code instances (:class:`WannierizeBlocksCodes`); ``wannierjl``
            only in split mode. A ``projwfc`` code requests the projected
            DOS: whenever the quality-check ``bands`` run exists, a
            projwfc.x step runs off its scratch and the ``projwfc`` output
            namespace is populated — unless a pseudo carries no
            ``PP_PSWFC`` atomic wavefunctions, which skips the step with
            a warning.
        structure: the periodic ``StructureData``.
        blocks: the resolved projection blocks, in band order (the unified
            outputs concatenate in this order); occupied and empty manifolds
            appear as separate blocks. Each is Wannierised independently.
            When every block carries a ``filled`` occupancy stamp, the
            ``orbitals`` initial screening partition is emitted as well
            (stamping only some blocks raises).
        kpoints: the explicit k-point list shared by the nscf and every
            block's wannier90 / pw2wannier90 (one node, so the k-ordering
            cannot drift between the steps).
        mp_grid: the Monkhorst-Pack dimensions ``kpoints`` was generated
            from. Carried separately because an explicit-list
            ``KpointsData`` cannot represent its parent mesh, and
            wannier90 requires ``mp_grid`` in the ``.win`` (it cannot
            re-derive it from the list).
        scf_kpoints: the mesh the shared scf samples. Unset falls back to
            the protocol's ``kpoints_distance``. Rejected together with
            ``nscf_remote_folder``, which skips the scf entirely.
        pseudo_family: pseudopotential family label.
        protocol: protocol name passed to both builders.
        overrides: optional :class:`WannierizeOverrides` — flat, semantic
            keys (``scf`` / ``nscf`` pw-protocol dicts feed
            :func:`RunWannierGroundState` when the internal scf + nscf runs; with an
            external ``nscf_remote_folder`` they instead reach every
            per-block wannier builder's nested ``get_builder_from_protocol``
            call, unused otherwise — see :func:`_builder_overrides`;
            ``wannier90`` / ``pw2wannier90`` flat keyword dicts feed every
            per-block wannier builder either way). Never the upstream
            namespace-nested shape.
        electronic_type / spin_type: forwarded to the wannier builder.
        nscf_remote_folder: an existing nscf scratch to build every block
            on. When given, the internal scf + nscf is skipped (and the
            ``nscf`` output namespace is absent); the caller owns keeping
            ``kpoints`` consistent with the scratch's k-list. This is how a
            workflow with one scratch shared *across* several
            ``WannierizeBlocks`` calls (e.g. one per spin channel) routes
            through here without rerunning the ground state. Incompatible
            with split mode, which needs the internal scf's density for its
            bands step. Without ``scf_remote_folder`` alongside it, the
            quality-check ``bands`` / ``projwfc`` outputs stay absent (no
            scf density to run the quality-check off).
        scf_remote_folder: a converged scf on this structure, skipping the
            internal scf either way. Given alone, also runs a fresh nscf
            on ``kpoints`` off it, populating the ``nscf`` output
            namespace. Paired with ``nscf_remote_folder``, only feeds the
            quality-check ``bands`` step. Refused in split mode.
        nscf_bands: the nscf eigenvalues. A disentangling block's frozen
            window is checked against them, so a ``dis_froz_max`` that would
            freeze more bands than the block Wannierises is rejected here
            instead of stopping wannier90 mid-run. Defaults to the internal
            nscf's ``output_band``; a caller supplying ``nscf_remote_folder``
            should pass the matching bands, since without them the window
            goes unchecked.
        split_threshold: minimum gap (eV) between consecutive bands for a
            split; setting it is one of the two split-mode triggers.
            ``None`` (with automatic-projections blocks supplying the other
            trigger) splits only at the occupied/empty boundary.
        bands_kpoints: the k-path for the pw.x quality-check ``bands`` run,
            independent of ``interpolation_kpoints`` (which falls back to
            this path when it is not given itself, and vice versa). Split
            mode requires it: the group detection reads the same run's
            eigenvalues.
        num_occ_bands: split mode only (required there) — occupied-band
            count of the channel (the detection always opens a new group at
            this boundary).
        wjl_options / cubic_pw2wannier90_options: split mode only — optional
            ``metadata.options`` for the Wannier.jl CalcJobs and the cubic
            pw2wannier90 run. Both CalcJobs carry their own ``resources``
            defaults, so normally leave these unset: a non-None dict
            currently trips aiida-wannierjl's options handling inside a
            graph body (dict graph inputs arrive as TaggedValue proxies,
            which node-graph refuses to assign into ``metadata.options``).
        subblock_wannier90_options: split mode only — optional
            ``metadata.options`` for the per-group re-wannierisation
            ``Wannier90Calculation`` (defaults to single-machine resources;
            that CalcJob has no resources default of its own).
        interpolation_kpoints: a labelled explicit-path ``KpointsData``
            along which every block's wannier90 interpolates its band
            structure; each ``blocks`` entry then populates
            ``interpolated_bands`` with its final gauge's interpolation
            (see :class:`WannierizeBlockOutputs`). In split mode the path
            reaches both the whole-block run and every per-group re-run
            (:func:`~aiida_koopmans.workgraphs.auto_wannierize.WannierizeAndSplitBlock`).
            Without ``bands_kpoints`` it also feeds the pw.x quality-check
            ``bands`` run (which needs the internal scf, so an external
            ``nscf_remote_folder`` leaves it out); with ``bands_kpoints``
            the two runs sample independent densities instead.
        external_projectors_path / external_projectors: the projector
            directory (one ``<element>.dat`` per element, on the
            pw2wannier90 code's computer) and the per-element orbital
            tables. Required together by any ``atomic_projectors_external``
            block, fed only to those blocks' wannier builders, and rejected
            when no block consumes them
            (:func:`validate_external_projector_inputs`).

    Returns:
        A :class:`WannierizeBlocksOutputs`: the ``blocks`` namespace keyed by
        block label, the unified ``centres`` / ``spreads``, the ``bands``
        quality-check run and its ``projwfc`` projected DOS (when they ran),
        the ``groups`` detection output (split mode), the ``orbitals``
        initial partition (only when every block carries a ``filled``
        stamp), the shared ``nscf`` outputs (whenever an nscf ran here),
        and ``scf_remote_folder`` (only when the internal scf + nscf ran
        here, for a later caller to reuse the density).
    """
    overrides = overrides or {}
    validate_parallelization(parallelization)
    _validate_blocks(blocks)
    validate_external_projector_inputs(
        [block["projection_type"] for block in blocks],
        external_projectors_path,
        external_projectors,
    )

    split = _resolve_split_mode(
        blocks=blocks,
        mp_grid=mp_grid,
        nscf_remote_folder=nscf_remote_folder,
        scf_remote_folder=scf_remote_folder,
        split_threshold=split_threshold,
        bands_kpoints=bands_kpoints,
        num_occ_bands=num_occ_bands,
        wjl_options=wjl_options,
        subblock_wannier90_options=subblock_wannier90_options,
        cubic_pw2wannier90_options=cubic_pw2wannier90_options,
        interpolation_kpoints=interpolation_kpoints,
    )

    # --- shared scf + nscf (run once, or reuse the caller's scratch) ---
    bands_outputs = None
    projwfc_outputs = None
    nscf_step = None
    if nscf_remote_folder is not None:
        # No internal scf normally means no density for the quality-check
        # bands run either — unless the caller also hands over the matching
        # scf scratch (``scf_remote_folder``), which lets the run happen off
        # it. The per-block Wannier interpolation (which reads only the nscf
        # scratch) runs either way.
        _reject_inputs_an_external_scratch_ignores(
            overrides, scf_kpoints, source="an external nscf_remote_folder"
        )
        scf_nscf = None
        nscf_scratch, block_bands, bands_outputs, projwfc_outputs = _reuse_external_nscf_scratch(
            codes=codes,
            structure=structure,
            nscf_remote_folder=nscf_remote_folder,
            scf_remote_folder=scf_remote_folder,
            interpolation_kpoints=interpolation_kpoints,
            nscf_bands=nscf_bands,
            overrides=overrides,
            pseudo_family=pseudo_family,
            protocol=protocol,
            electronic_type=electronic_type,
            parallelization=parallelization,
        )
    elif scf_remote_folder is not None:
        # The caller supplies a converged scf on this structure; run a
        # fresh nscf on ``kpoints`` off it instead of the internal scf.
        # ``_resolve_split_mode`` above rejects split mode on this path.
        _reject_inputs_an_external_scratch_ignores(
            overrides, scf_kpoints, source="an external scf_remote_folder"
        )
        scf_nscf = None
        nscf_step, nscf_scratch, block_bands, bands_outputs, projwfc_outputs = (
            _run_nscf_off_external_scf(
                codes=codes,
                structure=structure,
                kpoints=kpoints,
                scf_remote_folder=scf_remote_folder,
                bands_kpoints=bands_kpoints,
                interpolation_kpoints=interpolation_kpoints,
                nscf_bands=nscf_bands,
                overrides=overrides,
                pseudo_family=pseudo_family,
                protocol=protocol,
                electronic_type=electronic_type,
                parallelization=parallelization,
            )
        )
    else:
        scf_nscf_overrides: dict[str, Any] = {}
        if "scf" in overrides:
            scf_nscf_overrides["scf"] = overrides["scf"]
        if "nscf" in overrides:
            scf_nscf_overrides["nscf"] = overrides["nscf"]

        scf_nscf = RunWannierGroundState(
            pw_code=reference(codes, "pw"),
            structure=structure,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides=scf_nscf_overrides or None,
            # The blocks' wannier90 / pw2wannier90 read eigenstates on the
            # explicit ``kpoints`` mesh, so the nscf must run on exactly that
            # grid (not the protocol's kpoints_distance-derived one).
            nscf_kpoints=kpoints,
            scf_kpoints=scf_kpoints,
            parallelization=parallelization,
            metadata={"call_link_label": "scf_nscf", "label": "Ground state"},
        )
        nscf_scratch = scf_nscf["nscf_remote_folder"]
        # The eigenvalues each disentangling block's frozen window is checked
        # against; the caller's own copy wins so a shared ground state stays
        # the single source (the internal pair is skipped then).
        block_bands = nscf_bands if nscf_bands is not None else scf_nscf["nscf_output_band"]

        # --- quality-check / detection bands run (at most one per graph) ---
        # Nested under the internal scf + nscf on purpose: the run reads
        # this scf's remote folder (split mode rejects an external scratch,
        # and plain mode skips the run without an internal scf). Runs along
        # ``bands_kpoints`` when given — required in split mode, where it
        # doubles as the detection input — and falls back to
        # ``interpolation_kpoints`` otherwise, as the explicit eigenvalues
        # the Wannier interpolation is judged against. One run serves both
        # purposes when the caller passes the same node for both (how the
        # koopmans dispatcher wires a route with no independent density for
        # each).
        bands_outputs, projwfc_outputs = _run_explicit_bands_and_dos_steps(
            codes=codes,
            structure=structure,
            bands_kpoints=bands_kpoints,
            interpolation_kpoints=interpolation_kpoints,
            scf_remote_folder=scf_nscf["scf_remote_folder"],
            nscf_overrides=overrides.get("nscf"),
            pseudo_family=pseudo_family,
            protocol=protocol,
            electronic_type=electronic_type,
            parallelization=parallelization,
        )
        if split:
            detect = _detect_split_groups(
                bands_outputs,
                num_occ_bands,
                split_threshold,
                num_bands_total=sum(int(block["num_wann"]) for block in blocks),
            )

    _warn_unconstrained_blocks(blocks, overrides.get("wannier90"), electronic_type, split)

    # --- per-block Wannierisation: native for-loop fan-out ---
    # Each iteration adds an independent per-block graph (they share only
    # the read-only nscf scratch, so they run in parallel), collected into a
    # dict keyed by block label -> the ``blocks`` dynamic output namespace.
    # The parsed per-block outputs feed the unify task positionally, read
    # straight off each graph call (not off the ``blocks`` entries, whose
    # label keys would impose a sort order): the ``blocks`` input-list
    # order is the band-order authority.
    block_outputs: dict[str, Any] = {}
    collect_inputs: dict[str, Any] = {}
    for i, block in enumerate(blocks):
        external_kwargs = _external_kwargs_for(block, external_projectors_path, external_projectors)
        if split:
            from aiida_koopmans.workgraphs.auto_wannierize import WannierizeAndSplitBlock

            wannierized = WannierizeAndSplitBlock(
                # Wired unconditionally: WannierizeAndSplitBlock's own
                # SplitBlockCodes requires all four, so a missing
                # ``wannierjl`` surfaces there as the framework's structural
                # missing-input error.
                codes={
                    "pw": reference(codes, "pw"),
                    "pw2wannier90": reference(codes, "pw2wannier90"),
                    "wannier90": reference(codes, "wannier90"),
                    "wannierjl": reference(codes, "wannierjl"),
                },
                structure=structure,
                block=block,
                groups=detect.result,
                nscf_remote_folder=nscf_scratch,
                kpoints=kpoints,
                mp_grid=mp_grid,
                pseudo_family=pseudo_family,
                protocol=protocol,
                overrides=overrides or None,
                electronic_type=electronic_type,
                spin_type=spin_type,
                parallelization=parallelization,
                wjl_options=wjl_options,
                wannier90_options=subblock_wannier90_options,
                pw2wannier90_options=cubic_pw2wannier90_options,
                interpolation_kpoints=interpolation_kpoints,
                **external_kwargs,
                metadata={
                    "call_link_label": f"wannierize_split_{block['label']}",
                    "label": f"Split Wannierization ({block_display_name(block)})",
                },
            )
        else:
            wannierized = WannierizeBlock(
                codes={
                    "pw": reference(codes, "pw"),
                    "pw2wannier90": reference(codes, "pw2wannier90"),
                    "wannier90": reference(codes, "wannier90"),
                },
                structure=structure,
                block=block,
                projection_type=block["projection_type"],
                nscf_remote_folder=nscf_scratch,
                kpoints=kpoints,
                mp_grid=mp_grid,
                pseudo_family=pseudo_family,
                protocol=protocol,
                overrides=overrides or None,
                electronic_type=electronic_type,
                spin_type=spin_type,
                parallelization=parallelization,
                nscf_bands=block_bands,
                interpolation_kpoints=interpolation_kpoints,
                **external_kwargs,
                metadata={
                    "call_link_label": f"wannierize_{block['label']}",
                    "label": f"Wannierization ({block_display_name(block)})",
                },
            )
        # Both per-block graphs return the flat WannierizeBlockOutputs
        # shape, forwarded whole into the entry (split-mode entries leave
        # the plain-route-only folder sockets unpopulated).
        block_outputs[block["label"]] = wannierized
        collect_inputs[f"b{i:02d}"] = wannierized["output_parameters"]

    collected = collect_wannier_functions(
        output_parameters=collect_inputs,
        metadata={"call_link_label": "collect_wannier_functions"},
    )
    outputs = WannierizeBlocksOutputs(
        blocks=block_outputs,
        centres=collected["centres"],
        spreads=collected["spreads"],
    )
    _maybe_emit_orbital_partition(outputs, blocks)
    _wire_quality_check_outputs(outputs, bands_outputs, projwfc_outputs)
    if split:
        outputs["groups"] = detect.result
    _wire_scf_nscf_outputs(outputs, scf_nscf, nscf_step, nscf_scratch)
    return outputs
