"""Projection blocks and projection-spec accounting.

Holds the projection-block types (:class:`ProjectionBlock` and friends),
their validators and band bookkeeping, and the conversion of user
projection specs (``wannier90_input`` ``Projection`` models) into
Wannier90 ``.win`` projection strings and Wannier-function counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any, NotRequired, TypedDict, cast

import numpy as np
from aiida import orm
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.occupations import default_channel_nocc
from aiida_koopmans.spin import SpinChannel


def projection_win_string(projection: Any) -> str:
    """Format one projection as a Wannier90 ``.win`` projections line.

    ``projection`` is duck-typed on the ``wannier90_input`` ``Projection``
    model. Element-labelled sites render as ``<element>:<ang_mtm>``;
    single-point sites use Wannier90's ``f=x,y,z`` (crystal) / ``c=x,y,z``
    (Cartesian) forms. The ``ang_mtm`` quantum numbers stringify to
    Wannier90's own syntax (``l=-3`` for sp3, ...).
    """
    if projection.site is not None:
        return f"{projection.site}:{projection.ang_mtm}"
    fractional = getattr(projection, "fractional_site", None)
    if fractional is not None:
        return f"f={','.join(str(c) for c in fractional)}:{projection.ang_mtm}"
    cartesian = getattr(projection, "cartesian_site", None)
    if cartesian is not None:
        return f"c={','.join(str(c) for c in cartesian)}:{projection.ang_mtm}"
    raise ValueError(f"Projection {projection!r} defines no site.")


def projection_num_wann(structure: orm.StructureData, projection: Any) -> int:
    """Count the Wannier functions of one projection: site multiplicity x (2l+1).

    ``projection`` is duck-typed on the ``wannier90_input`` ``Projection``
    model (``.site`` element label or a ``fractional_site`` /
    ``cartesian_site`` single point, ``.ang_mtm`` quantum numbers).
    """
    if projection.site is not None:
        n_sites = sum(1 for site in structure.sites if site.kind_name == projection.site)
        if n_sites == 0:
            raise ValueError(
                f"Projection site '{projection.site}' does not match any atom in the structure."
            )
    elif (
        getattr(projection, "fractional_site", None) is not None
        or getattr(projection, "cartesian_site", None) is not None
    ):
        # An explicit point hosts exactly one set of orbitals.
        n_sites = 1
    else:
        raise ValueError(f"Projection {projection!r} defines no site.")
    quantum_numbers = projection.ang_mtm
    if quantum_numbers.m_r is not None:
        multiplicity = len(quantum_numbers.m_r)
    else:
        l_value = quantum_numbers.angular.value
        # Hybrids are encoded with negative l: sp=-1 (2 orbitals), sp2=-2 (3),
        # sp3=-3 (4), sp3d=-4 (5), sp3d2=-5 (6).
        multiplicity = 2 * l_value + 1 if l_value >= 0 else 1 - l_value
    return n_sites * multiplicity


def band_range_complement(start: int, end: int, nbnd: int) -> list[int] | None:
    """Return the wannier90 ``exclude_bands`` list complementing ``[start, end]``.

    A list of band indices (not the ``.win`` range string): aiida-wannier90's
    input writer expects integers and does the range compression itself.
    """
    excluded = [*range(1, start), *range(end + 1, nbnd + 1)]
    return excluded or None


def detect_band_blocks(
    energies: np.ndarray,
    num_occ_bands: int | None = None,
    threshold: float | None = None,
) -> list[list[int]]:
    """Group bands into energy-separated blocks (1-indexed band groups).

    Walks the bands of ``energies`` (shape ``(nkpts, nbands)``) in order and
    opens a new group whenever

    * the occupied/empty boundary is crossed (band ``num_occ_bands + 1``
      always starts a group), or
    * the band is separated from the previous one by an energy gap larger
      than ``threshold`` (eV) everywhere in the Brillouin zone (the minimum
      of band *i* lies more than ``threshold`` above the maximum of band
      *i - 1*). ``threshold=None`` disables gap detection.
    """
    boundary = -1 if num_occ_bands is None else num_occ_bands
    groups: list[list[int]] = [[1]]
    for i in range(1, energies.shape[1]):
        if i == boundary:
            groups.append([i + 1])
        elif threshold and energies[:, i].min() - energies[:, i - 1].max() > threshold:
            groups.append([i + 1])
        else:
            groups[-1].append(i + 1)
    return groups


def restrict_groups_to_block(groups: list[list[int]], include_bands: list[int]) -> list[list[int]]:
    """Restrict globally-detected band groups to the bands of one block.

    Keeps, from each group, the bands that belong to ``include_bands``
    (dropping groups with no overlap). The retained groups must cover the
    block exactly — a block band missing from every group means the groups
    were detected over too few bands.
    """
    include = set(include_bands)
    restricted = [[band for band in group if band in include] for group in groups]
    restricted = [group for group in restricted if group]
    covered = {band for group in restricted for band in group}
    if covered != include:
        raise ValueError(
            f"The detected band groups cover bands {sorted(covered)} of the block but the "
            f"block includes bands {sorted(include)}; the group detection must span every "
            "band of the block."
        )
    return restricted


def groups_to_wannier_indices(groups: list[list[int]], include_bands: list[int]) -> list[list[int]]:
    """Map global band-index groups onto a block's 1-based Wannier indices.

    The wannierjl split indexes the Wannier functions of the block's model
    (``1 .. num_wann``), not global band indices, so each band is replaced
    by its 1-based position within the block's (sorted) ``include_bands``.
    """
    position = {band: i + 1 for i, band in enumerate(sorted(include_bands))}
    return [[position[band] for band in group] for group in groups]


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
      is what makes the block require disentanglement; the extra bands
      always sit above the block's own.

    ``exclude_bands`` names every band of the pw.x run the block does not
    read, so ``len(exclude_bands) + num_bands`` is that run's band count --
    the identity wann2kcp.x checks a ``.chk`` against. Those two fields
    together fix where the block's Wannier functions sit in the channel's
    Wannier ordering, so those indices are derived rather than stored
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


def _read_bands(block: ProjectionBlock) -> list[int]:
    """Return the ``num_bands`` bands the block reads, in ascending order.

    ``exclude_bands`` names the bands wannier90 skips; the block reads the
    lowest ``num_bands`` of the rest.
    """
    excluded = set(block.get("exclude_bands") or [])
    num_bands = int(block["num_bands"])
    read: list[int] = []
    band = 1
    while len(read) < num_bands:
        if band not in excluded:
            read.append(band)
        band += 1
    return read


def validate_projection_block(block: ProjectionBlock) -> None:
    """Reject a block whose band bookkeeping cannot describe a Wannierization.

    Lives beside the type because a ``TypedDict`` cannot validate at
    runtime; every entry point that takes blocks calls this first. The
    rules are ``num_bands >= num_wann >= 1``, and the bands the block
    reads — its own and any extra disentanglement bands — must be contiguous:
    ``exclude_bands`` may name only bands below or above them. Raise
    ``ValueError`` naming the block and the rule it breaks. Occupancy is
    not among the rules -- it is allowed to be unknown here
    (:func:`block_occupancy`).
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
    read = _read_bands(block)
    gaps = sorted(set(range(read[0], read[-1] + 1)) - set(read))
    if gaps:
        raise ValueError(
            f"Block {label!r} reads bands {read}, but `exclude_bands` names "
            f"bands {gaps} inside that window. The bands a block reads must "
            "be contiguous; exclude only bands below or above the block."
        )


def validate_projection_block_sequence(blocks: Sequence[ProjectionBlock]) -> None:
    """Reject a block layout that breaks the band-to-Wannier-function match.

    ``blocks`` is every block a route Wannierises. Two rules, each checked
    per spin channel since the channels are independent orderings:

    * blocks ascend: each block's bands must start above the highest
      Wannier-function band of the block before it;
    * only a channel's uppermost block may require disentanglement
      (``num_bands > num_wann``): the extra bands of a lower block inflate
      the bands every block above it reads.

    Under these rules the indices :func:`get_wannier_indices` returns
    are exactly the block's own band indices. Raise ``ValueError`` naming
    the offending blocks.
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
    """Return the block's ascending Wannier-function indices within its spin channel.

    The returned list holds the ``num_wann`` indices the block's Wannier
    functions take among the channel's, counted from 1. They are derived
    from the bands the block reads — ``exclude_bands`` and ``num_bands``
    say which — as the lowest ``num_wann`` of those, because any extra
    disentanglement bands sit above the block's own. That these are also
    the block's band indices is guaranteed by
    :func:`validate_projection_block_sequence`: blocks ascend within each
    channel, and only the uppermost may require disentanglement. A block
    that does optimizes its subspace out of every band it reads, so its
    indices say where it sits and how wide it is, never which bands its
    Wannier functions came from. That is what the
    band-to-Wannier-function map needs
    (:func:`~aiida_koopmans.projections.groups_to_wannier_indices`);
    widening the list to the extra bands would mis-address the Wannier
    functions.
    """
    return _read_bands(block)[: int(block["num_wann"])]


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
            "Its Wannier-function indices cannot settle it: a block that "
            "disentangles across the occupied/empty boundary draws its Wannier "
            "functions from both manifolds, and its indices still sit on one side."
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
    filled: bool,
) -> list[ExplicitProjectionBlock]:
    """Materialise one manifold's per-block :class:`ExplicitProjectionBlock` list.

    Blocks cover consecutive band windows starting at ``first_band`` and
    carry the manifold's ``filled`` occupancy. Only the *last* block
    absorbs the manifold's ``extra_bands`` disentanglement bands
    (``num_bands > num_wann``), the band layout the u_dis merge in
    :func:`~aiida_koopmans.workgraphs.dfpt.prepare_kcw_wannier_files` relies
    on. A single-block manifold
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
                filled=filled,
                num_wann=num_wann,
                num_bands=num_bands,
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
    merged again before kcw.x by :func:`~aiida_koopmans.workgraphs.dfpt.prepare_kcw_wannier_files`.

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
        occupied, "occ", label_suffix, spin_channel, 1, nbnd, 0, filled=True
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
            filled=False,
        )

    return occ_blocks, emp_blocks, has_disentangle, num_wann_occ + num_wann_emp
