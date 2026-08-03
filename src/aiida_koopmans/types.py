"""Shared types for the koopmans AiiDA workgraph layer.

Cross-module data shapes live here so workgraph builders, the kcp.x
CalcJob, parser, and tests can all import a single canonical definition.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, NotRequired, TypedDict, get_args

from aiida_koopmans.projections import (
    ProjectionBlock,
    block_occupancy,
    validate_projection_block,
)
from aiida_koopmans.spin import SpinChannel


class Correction(str, Enum):
    """The Koopmans correction (functional) the workflow applies.

    Members:

    * ``KI``: Koopmans-Integral correction (the default).
    * ``KIPZ``: Koopmans-Integral with Perdew-Zunger self-interaction
      correction on the variational orbitals — different sub-step
      parameters; see ``aiida_koopmans/workgraphs/kcp.py``.
    * ``PKIPZ``: perturbative KIPZ (trial KI, KIPZ correction applied
      post-hoc). Not yet implemented — accepted at the type level but
      not wired through the dispatcher.
    * ``PZ``: plain Perdew-Zunger orbital-dependent functional —
      strictly not a "Koopmans correction" but routed through the
      same orbital-dependent screening machinery
      (:func:`aiida_koopmans.workgraphs.kcp._build_orbdep_parameters`),
      e.g. the empty-orbital ``pz_print`` sub-step of a KI workflow.
    * ``NONE``: no Koopmans correction (plain DFT only).
    * ``ALL``: run KI / KIPZ / PKIPZ together (user-facing workflow
      control).
    """

    KI = "ki"
    KIPZ = "kipz"
    PKIPZ = "pkipz"
    PZ = "pz"
    NONE = "none"
    ALL = "all"


class VariationalOrbitalType(str, Enum):
    """Initial variational orbitals to use for the trial KI / KIPZ run.

    * ``PZ``: PZ-initialised variational orbitals.
    * ``KOHN_SHAM``: KS orbitals from the DFT init reused as
      variational (the currently-supported path; produces a
      KS-as-variational overlay so the trial KI's ``evc0N.dat`` is the
      DFT ``evcN.dat``).
    * ``MLWFS``: maximally-localised Wannier functions
      (Wannier90-based; not yet implemented in the kcp.x stream).
    * ``PROJWFS``: projected Wannier functions (not yet implemented in
      the kcp.x stream).
    """

    PZ = "pz"
    KOHN_SHAM = "kohn-sham"
    MLWFS = "mlwfs"
    PROJWFS = "projwfs"


class MLMode(str, Enum):
    """What the trajectory workflow does with the machine-learning dataset.

    ``NONE`` just runs the snapshots; ``TRAIN`` fits a screening model on
    the computed alphas; ``TEST`` scores a previously trained model
    against them; ``PREDICT`` applies a previously trained model in place
    of the per-orbital Delta-SCF refinement.
    """

    NONE = "none"
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"


class MLDescriptor(str, Enum):
    """Descriptor feeding the machine-learned screening model.

    Both descriptors are derived from the variational orbitals' densities:
    ``SELF_HARTREE`` is the scalar self-Hartree energy kcp.x reports, and
    ``POWER_SPECTRUM`` is the rotationally-invariant power spectrum of a
    pw2wannier90.x ``wan_mode='decompose'`` expansion of each density.
    """

    SELF_HARTREE = "self_hartree"
    POWER_SPECTRUM = "power_spectrum"


class VariationalOrbital(TypedDict):
    """Structured record for a single variational orbital.

    Carries spin / per-spin 1-based band index / filled-vs-empty plus
    its place in any grouping (``group_id``, ``representative``). The
    key names *are* the structural information — stable and never
    parsed back into parts, unlike a flat string label like
    ``"up_orb_5"``; use :func:`map_key_for` when a string label is
    needed (only at the ``aiida-workgraph`` ``Map`` zone boundary).

    On AiiDA round-trip ``spin`` comes back as a plain ``str`` rather
    than a :class:`SpinChannel` enum, so compare with ``==`` not
    ``is`` (``SpinChannel`` subclasses ``str``, so
    ``o["spin"] == SpinChannel.UP`` holds but ``is`` does not).

    ``manifold`` is the label of the projection block the orbital's
    Wannier function belongs to, set by callers that build orbitals
    from projection blocks. Workflows without blocks have no such
    label, so it is ``NotRequired``; the partition operator
    :func:`~aiida_koopmans.workgraphs.variational_orbitals.refine_by_key`
    demands it on every orbital before splitting on it.
    """

    spin: SpinChannel
    index: int  # 1-based per-spin band index
    filled: bool
    group_id: int
    representative: bool
    manifold: NotRequired[str]  # projection-block label; set by block-building callers


def map_key_for(orbital: VariationalOrbital) -> str:
    """Stable string label used as a key in ``Map`` zone iteration dicts.

    ``aiida-workgraph``'s ``Map`` zone iterates over a dict and uses
    the key as the iteration handle / resulting calc-node link label.
    Strings are required there. This is the *only* place where an
    orbital's identity is encoded as a string; the round-trip back to
    structured form goes through :func:`enumerate_variational_orbitals`
    at the gather boundary, never by parsing the string.

    Uses ``SpinChannel(...).value`` rather than ``str(spin)`` because
    Python 3.12+ changed ``str()`` on ``str``-Enums to return
    ``"<ClassName>.<member>"`` for non-trivial subclasses — explicit
    ``.value`` access stays "up" / "down" / "none" regardless of
    Python version, and also normalises post-AiiDA-round-trip values
    where ``spin`` arrives as a plain ``str`` rather than the enum.
    """
    spin = SpinChannel(orbital["spin"])
    tag = "" if spin is SpinChannel.NONE else f"{spin.value}_"
    return f"{tag}orb_{orbital['index']}"


class AlphaScreening(TypedDict):
    """Per-spin per-orbital screening parameters for the kcp.x ``file_alpharef``.

    Both ``filled`` and ``empty`` are dicts keyed by spin channel; each
    value is a list of one ``alpha`` per per-spin orbital, 1-indexed by
    list position.

    For ``nspin == 2``: keys are ``SpinChannel.UP`` and ``SpinChannel.DOWN``; the
    ``KcpCalculation`` flattens them into the kcp.x file format on write
    (block-spin: all ``SpinChannel.UP`` entries first, then ``SpinChannel.DOWN``).

    For ``nspin == 1``: the only key is ``SpinChannel.NONE``.
    """

    filled: dict[SpinChannel, list[float]]
    empty: dict[SpinChannel, list[float]]


# The QE code vocabulary, defined once. ``CODE_NAMES`` is the runtime tuple
# (the koopmans2 parallelization schema imports it for its ``ALL_CODES``); the
# ``CodeName`` ``Literal`` types dict keys / helper args so a typo is a static
# error, and ``validate_parallelization`` catches one that slips in at runtime.
CodeName = Literal["pw", "kcp", "kcw", "ph", "projwfc", "pw2wannier90", "wann2kcp", "wannier90"]
CODE_NAMES: tuple[str, ...] = get_args(CodeName)


class CodeParallelization(TypedDict, total=False):
    """One code's parallelization directive: MPI ranks, k-point pools, pencil decomp, threads.

    ``ntasks`` sets ``metadata.options.resources`` (``num_mpiprocs_per_machine``);
    ``npool`` becomes ``-npool`` and ``pd`` becomes ``-pd true`` on the QE
    command line; ``omp`` sets the per-rank OpenMP/BLAS thread count via a
    ``metadata.options.prepend_text`` export block (overriding the
    computer-level pin of one thread). Every field is optional
    (``total=False``); an absent one means the QE/AiiDA default. Mirrors the
    koopmans2 ``CodeParallelization`` pydantic model that produces these dicts.
    """

    ntasks: int
    npool: int
    pd: bool
    omp: int


# Per-code parallelization mapping threaded into every top-level graph: a plain
# dict keyed by code name, each value a :class:`CodeParallelization`. A dict
# alias (not a fixed-key TypedDict) so ``aiida-workgraph`` keeps it as one
# opaque input socket rather than expanding a typed namespace, and so a dynamic
# ``code`` lookup types cleanly. Which flags each code accepts is enforced by
# ``POOL_SUPPORTING_CODES`` / ``PD_SUPPORTING_CODES`` in ``aiida_koopmans.workgraphs``.
ParallelizationDict = dict[CodeName, CodeParallelization]


class MergeGroup(TypedDict):
    """A set of :class:`ProjectionBlock` instances merged into one kcp.x manifold.

    Blocks that share a filling (occupied vs empty) and spin are merged
    together (their per-block ``evcw`` wavefunctions are concatenated by
    ``merge_evc.x``) into a single ``evc_occupied`` / ``evc0_empty`` file
    that seeds the supercell kcp.x run.

    * ``filled``: ``True`` for the occupied manifold, ``False`` for empty.
    * ``spin``: the shared spin channel (``SpinChannel.NONE`` for nspin=1).
    * ``blocks``: the member blocks, in band order.
    """

    filled: bool
    spin: SpinChannel
    blocks: list[ProjectionBlock]


def group_blocks_to_merge(
    blocks: list[ProjectionBlock],
    num_occ_bands: dict[SpinChannel, int],
) -> list[MergeGroup]:
    """Group blocks into occupied / empty manifolds per spin.

    Each block's ``filled`` stamp decides which manifold it joins; an
    unstamped block raises, since the merge cannot place it.
    ``num_occ_bands`` maps each spin channel to its number of occupied
    bands (for ``nspin == 1``, the single key ``SpinChannel.NONE``) and is
    checked, not used to classify: a channel's occupied blocks must span
    exactly that many Wannier functions, since the merged
    ``evc_occupied`` file seeds the whole occupied manifold of the kcp.x
    run.

    Returns one :class:`MergeGroup` per ``(filled, spin)`` that has
    members, preserving the order in which blocks are first encountered so
    the downstream ``merge_evc.x`` concatenation is deterministic.
    """
    groups: list[MergeGroup] = []
    index: dict[tuple[bool, SpinChannel], MergeGroup] = {}
    for block in blocks:
        validate_projection_block(block)
        spin = SpinChannel(block["spin"])
        if spin not in num_occ_bands:
            raise KeyError(
                f"`num_occ_bands` has no entry for spin {spin!r}; provide one "
                f"occupied-band count per spin channel (use SpinChannel.NONE "
                f"for nspin==1)."
            )
        key = (block_occupancy(block), spin)
        group = index.get(key)
        if group is None:
            group = MergeGroup(filled=key[0], spin=spin, blocks=[])
            index[key] = group
            groups.append(group)
        group["blocks"].append(block)
    spins_present = {SpinChannel(block["spin"]) for block in blocks}
    for spin in SpinChannel:
        if spin not in spins_present:
            continue
        occupied = [
            block
            for block in blocks
            if SpinChannel(block["spin"]) == spin and block_occupancy(block)
        ]
        spanned = sum(int(block["num_wann"]) for block in occupied)
        n_occ = num_occ_bands[spin]
        if spanned != n_occ:
            labels = [block["label"] for block in occupied]
            raise ValueError(
                f"The occupied blocks of spin {spin.value!r} ({labels}) span {spanned} "
                f"Wannier functions but the channel has {n_occ} occupied bands. Every "
                "occupied band must be covered exactly once; check the `filled` stamps "
                "and the projections."
            )
    return groups


def merge_dest_filename(filled: bool, spin_index: int) -> str:
    """kcp.x-side filename for a merged manifold wavefunction.

    The supercell kcp.x run reads its initial variational orbitals from
    ``evc_occupied{n}.dat`` (occupied manifold) or ``evc0_empty{n}.dat``
    (empty manifold), where ``n`` is the 1-based kcp.x spin index
    (1 = up / unpolarized, 2 = down).
    """
    if spin_index not in (1, 2):
        raise ValueError(f"spin_index must be 1 or 2, got {spin_index!r}")
    if filled:
        return f"evc_occupied{spin_index}.dat"
    return f"evc0_empty{spin_index}.dat"
