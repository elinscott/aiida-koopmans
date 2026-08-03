"""Shared types for the koopmans AiiDA workgraph layer.

Cross-module data shapes live here so workgraph builders, the kcp.x
CalcJob, parser, and tests can all import a single canonical definition.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from itertools import pairwise
from typing import Literal, NotRequired, TypedDict, cast, get_args

from aiida_wannier90_workflows.common.types import WannierProjectionType


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


class SpinChannel(str, Enum):
    """Spin channel index used as a dict key in per-spin data structures.

    Use ``SpinChannel.NONE`` for ``nspin == 1`` calculations (no spin
    polarisation, single channel).
    """

    # Declaration order is the canonical channel walk order: iterating
    # the enum IS the ordering authority for representative stamping and
    # orbital emission. The position of NONE relative to UP/DOWN is
    # immaterial — the channels are mutually exclusive spin regimes, so
    # no calculation ever walks NONE alongside UP/DOWN; what matters is
    # only UP before DOWN.
    NONE = "none"
    UP = "up"
    DOWN = "down"
    SPINOR = "spinor"

    @property
    def axis(self) -> int:
        """Spin index into a stacked-by-spin array (axis-0 of ``(nspin, ...)``).

        ``NONE`` and ``UP`` both live at index 0 (kcp.x's nspin=1 file layout
        and the up channel of nspin=2 share the leading axis); ``DOWN`` is 1.
        ``SPINOR`` (noncollinear, nspin=4) has a single band index — 0.
        """
        return 1 if self is SpinChannel.DOWN else 0


class MLMode(str, Enum):
    """What the trajectory workflow does with the machine-learning dataset.

    ``NONE`` just runs the snapshots; ``TRAIN`` fits a screening model on
    the computed alphas; ``TEST`` scores a previously trained model
    against them. Prediction is not implemented.
    """

    NONE = "none"
    TRAIN = "train"
    TEST = "test"


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

    Carries spin / per-spin 1-indexed position / filled-vs-empty plus
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
    index: int  # 1-indexed per-spin band position
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


class OrbitalDict(TypedDict):
    """A single resolved Wannier orbital as a plain dict.

    A typed view over the dict AiiDA produces via
    ``Orbital.get_orbital_dict()`` for the ``core.realhydrogen`` orbital
    type (the one ``aiida-wannier90``'s ``OrbitalData`` stores);
    :func:`orbital_data_to_dicts` / :func:`dicts_to_orbital_data`
    round-trip it losslessly against ``OrbitalData``. Convert to
    ``OrbitalData`` only at the ``Wannier90WorkChain`` input boundary.

    The keys mirror ``aiida.tools.data.orbital.realhydrogen``'s fields
    exactly; ``test_projection_blocks`` asserts parity so an upstream
    schema change is caught rather than silently drifting.
    """

    _orbital_type: str
    position: list[float]
    angular_momentum: int
    magnetic_number: int
    radial_nodes: int
    kind_name: str
    spin: int
    x_orientation: list[float] | None
    z_orientation: list[float] | None
    spin_orientation: list[float] | None
    diffusivity: float | None


class _ProjectionBlockBase(TypedDict):
    """Band-bookkeeping shared by every projection block (any source).

    The per-block bookkeeping that is independent of *how* the Wannier
    functions are obtained: the block's label, spin, the counts
    (``num_wann`` is the common denominator across all projection
    sources), and which bands it covers. ``projection_type`` is the
    discriminator (a real :class:`WannierProjectionType`, registered for
    AiiDA serialization via the ``aiida.data`` entry points).

    Two counts, two meanings:

    * ``num_wann`` -- the Wannier functions the block produces.
    * ``num_bands`` -- the bands wannier90 reads. Exceeding ``num_wann``
      is what makes the block disentangle; the excess is its pool, and
      the pool always sits above the block's own bands.

    ``exclude_bands`` names every band of the pw.x run the block does not
    read, so ``len(exclude_bands) + num_bands`` is that run's band count --
    the identity wann2kcp.x checks a ``.chk`` against. Those two fields
    together fix where the block's Wannier functions sit in the channel's
    Wannier ordering, so those positions are derived rather than stored
    (:func:`get_wannier_indices`).

    ``filled`` is the block's occupancy: ``True`` when its Wannier
    functions come from the occupied manifold alone, ``False`` when they
    come from the empty manifold alone. Unset means *not yet known*, which
    is a legitimate state: a block derived from atomic projectors exists
    before anything has looked at the band structure, and its occupancy is
    settled only by the runtime band-group detection. A consumer that
    needs the occupancy and finds it unset raises
    (:func:`block_occupancy`).
    """

    label: str
    spin: SpinChannel
    filled: NotRequired[bool]
    num_wann: int
    num_bands: int
    exclude_bands: NotRequired[list[int] | None]
    projection_type: WannierProjectionType


class ExplicitProjectionBlock(_ProjectionBlockBase):
    """A block whose Wannier functions come from explicit projections.

    ``projection_type`` is ``WannierProjectionType.ANALYTIC``.
    ``projections`` is required: the Wannier90-format projection strings
    (``site:ang_mtm``), written verbatim into the ``.win`` projections
    block. Moving to resolved :class:`OrbitalDict` entries is the intended
    end state (hence the drift guard in ``test_projection_blocks``), but no
    route produces them yet.
    """

    projections: list[str]


class AutomaticProjectionBlock(_ProjectionBlockBase):
    """A block whose Wannier functions are found automatically.

    For ``projection_type`` in ``{SCDM, ATOMIC_PROJECTORS_QE,
    ATOMIC_PROJECTORS_EXTERNAL, RANDOM}`` there are no explicit projection
    orbitals -- the block is defined by ``num_wann`` (plus the frozen /
    disentanglement windows carried elsewhere).
    """


# Analytic blocks carry ``projections``; automatic blocks do not, so
# ``"projections" in block`` narrows the union to the explicit arm.
ProjectionBlock = ExplicitProjectionBlock | AutomaticProjectionBlock


def validate_projection_block(block: ProjectionBlock) -> None:
    """Reject a block whose band bookkeeping cannot describe a Wannierization.

    Lives beside the type because a ``TypedDict`` cannot validate at
    runtime; every entry point that takes blocks calls this first. The
    rule is ``num_bands >= num_wann >= 1``. Raise ``ValueError`` naming
    the block and the rule it breaks. Occupancy is not among the rules --
    it is allowed to be unknown here (:func:`block_occupancy`).
    """
    label = block["label"]
    num_wann = int(block["num_wann"])
    num_bands = int(block["num_bands"])
    if num_wann < 1:
        raise ValueError(
            f"Block {label!r} declares num_wann = {num_wann}; every block must "
            "carry at least one Wannier function."
        )
    if num_bands < num_wann:
        raise ValueError(
            f"Block {label!r} reads {num_bands} bands but Wannierises {num_wann} "
            "functions; wannier90 needs at least one band per Wannier function."
        )


def validate_projection_block_sequence(blocks: Sequence[ProjectionBlock]) -> None:
    """Reject blocks laid out so band indices stop being Wannier positions.

    ``blocks`` is every block a route Wannierises. Two rules, each checked
    per spin channel since the channels are independent orderings:

    * blocks ascend: each block's bands must start above the highest
      Wannier-function band of the block before it;
    * only a channel's uppermost block may disentangle
      (``num_bands > num_wann``): a lower block's pool inflates the bands
      every block above it reads.

    Under these rules a block's band indices are its positions in the
    channel's Wannier ordering, which is what :func:`get_wannier_indices`
    returns. Raise ``ValueError`` naming the offending blocks.
    """
    by_spin: dict[SpinChannel, list[ProjectionBlock]] = {}
    for block in blocks:
        by_spin.setdefault(SpinChannel(block["spin"]), []).append(block)
    for channel_blocks in by_spin.values():
        for prev, block in pairwise(channel_blocks):
            first = get_wannier_indices(block)[0]
            prev_top = get_wannier_indices(prev)[-1]
            if first <= prev_top:
                raise ValueError(
                    f"Block {block['label']!r} starts at band {first}, but block "
                    f"{prev['label']!r} before it already Wannierises band "
                    f"{prev_top}. List each spin channel's blocks in ascending "
                    "band order, with no band in two blocks."
                )
        top = channel_blocks[-1]["label"]
        for block in channel_blocks[:-1]:
            if int(block["num_bands"]) > int(block["num_wann"]):
                raise ValueError(
                    f"Block {block['label']!r} reads {block['num_bands']} bands for "
                    f"{block['num_wann']} Wannier functions, so it disentangles, but "
                    f"block {top!r} sits above it. Only a channel's uppermost block "
                    f"may disentangle: give the extra bands to {top!r}, or set this "
                    "block's `num_bands` equal to its `num_wann`."
                )


def get_wannier_indices(block: ProjectionBlock) -> list[int]:
    """Return the block's band indices: the lowest ``num_wann`` bands it reads.

    ``exclude_bands`` and ``num_bands`` say which bands wannier90 reads;
    the returned list holds the lowest ``num_wann`` of those, counted from
    1. These indices are also the block's positions in the channel's
    Wannier ordering, because :func:`validate_projection_block_sequence`
    enforces the layout that makes the two agree: blocks ascend within
    each channel, and only the uppermost may disentangle. A block that
    does disentangle optimizes its subspace out of every band it reads, so
    its indices say where the block sits and how wide it is, never which
    bands its Wannier functions came from. That is what the
    band-to-Wannier-function map needs
    (:func:`~aiida_koopmans.projections.groups_to_wannier_indices`);
    widening the list to the pool would mis-address the Wannier functions.
    """
    excluded = set(block.get("exclude_bands") or [])
    num_bands = int(block["num_bands"])
    read: list[int] = []
    band = 1
    while len(read) < num_bands:
        if band not in excluded:
            read.append(band)
        band += 1
    return read[: int(block["num_wann"])]


def block_occupancy(block: ProjectionBlock) -> bool:
    """Return whether the block is occupied, raising if it does not say.

    Raise ``ValueError`` naming the block when ``filled`` is unset: the
    occupancy of a block derived from atomic projectors is settled by the
    runtime band-group detection, and until then no caller may act on it.
    """
    if "filled" not in block:
        raise ValueError(
            f"Block {block['label']!r} does not say whether it is occupied or "
            "empty. Stamp `filled` where the occupancy is known -- from explicit "
            "projections, or from the band groups the runtime detection found. "
            "Its Wannier positions cannot settle it: a block that disentangles "
            "across the occupied/empty boundary draws its Wannier functions from "
            "both manifolds, and its positions still sit on one side."
        )
    return bool(block["filled"])


class ProjectionBlockId(TypedDict):
    """Identity-and-shape view of a :class:`ProjectionBlock`.

    Carries what downstream bookkeeping needs to enumerate a block's
    orbitals — the label, channel, occupancy, and Wannier-function
    count — and nothing else. Two reasons this view exists instead of
    passing full blocks:

    * provenance stays slim: the enumeration consumers never read
      ``projections`` / ``num_bands`` / ``exclude_bands``, so storing
      them on every partition task input would be noise;
    * a list of full blocks cannot ride a PyFunction input regardless:
      ``aiida-pythonjob`` dispatches its serializer registry on the
      *outer* type only, so ``list`` maps straight to ``orm.List`` and
      JSON storage rejects the nested ``projection_type`` enum — the
      registered top-level serializer for that enum (its ``EnumData``
      entry point) is never consulted for container internals. This
      JSON-pure view needs no registry at all. Reported upstream as
      https://github.com/aiidateam/aiida-pythonjob/issues/83 — once
      containers consult the registry, this second reason falls away
      and the view can be reconsidered.
    """

    label: str
    spin: SpinChannel
    filled: bool
    num_wann: int


def validate_projection_block_id(spec: ProjectionBlockId) -> None:
    """Reject a block view whose shape cannot describe real orbitals.

    Lives beside the type because a ``TypedDict`` cannot validate at
    runtime; every consumer that trusts the shape calls this first.
    Raise ``ValueError`` for a non-positive ``num_wann``.
    """
    if int(spec["num_wann"]) < 1:
        raise ValueError(
            f"Block {spec['label']!r} declares num_wann = {spec['num_wann']}; "
            "every block must carry at least one Wannier function."
        )


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


def block_w90_kwargs(block: ProjectionBlock) -> dict:
    """Return the Wannier90 input keywords for a single block.

    The per-block ``num_wann`` / ``num_bands`` / ``exclude_bands`` (and
    ``spin`` when the block is spin-resolved) that distinguish one block's
    Wannier90
    cycle from another's. ``projections`` is included only for an
    :class:`ExplicitProjectionBlock`; automatic blocks rely on
    ``projection_type`` instead. ``exclude_bands`` is omitted when the
    block excludes nothing.
    """
    kwargs: dict = {
        "num_wann": block["num_wann"],
        "num_bands": block["num_bands"],
    }
    exclude = block.get("exclude_bands")
    if exclude is not None:
        kwargs["exclude_bands"] = exclude
    if block["spin"] != SpinChannel.NONE:
        kwargs["spin"] = SpinChannel(block["spin"]).value
    if "projections" in block:
        kwargs["projections"] = cast("ExplicitProjectionBlock", block)["projections"]
    return kwargs


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
