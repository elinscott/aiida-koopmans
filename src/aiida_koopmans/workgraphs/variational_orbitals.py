"""Variational-orbital grouping for screening-parameter reuse.

Group variational orbitals so that orbitals sharing a group receive a
single representative screening-parameter calculation, with the result
copied onto the rest of the group.

The model is a partition of the orbitals into screening-equivalence
groups, encoded in ``group_id``; every grouping criterion is a view of
that one object. :func:`refine_by_key` / :func:`refine_by_labels`
(exact categorical splits) and
:func:`refine_by_scalar` (per-group sort-and-cut with a tolerance,
i.e. single linkage) only ever split existing groups, never merge
them, so exact refinements compose in any order and both operators
are idempotent. Apply every exact refinement before the scalar one,
so a tolerance chain cannot bridge across a categorical boundary.
No workflow calls the operators yet — wiring them into the graphs
comes separately.

The currently-wired path is :func:`assign_orbital_groups`, which
clusters with ``scipy.cluster.hierarchy.fcluster`` using complete
linkage. Orbitals are partitioned by ``(spin, filled)`` first — never
grouped across spin channels or across the filled / empty boundary.
Within each subset, an "ill-separated" check (any inter-cluster gap
smaller than ``2 * tol``) triggers a fallback to ``0.9 * tol`` and the
clustering is rerun. If the tolerance shrinks below ``0.01 * default_tol``
the algorithm raises rather than emitting unreliable groups. The two
paths deliberately diverge for now. Single linkage makes
:func:`refine_by_scalar` idempotent: every adjacent gap inside a group
it forms is at most ``tol``, so re-applying the operator finds no new
cuts. Complete linkage instead bounds the overall cluster diameter, so
on a chain of closely spaced values (e.g. ``0.0, 0.25, 0.5, 0.75`` at
``tol=0.3``) it must break the chain somewhere, and re-clustering the
resulting groups need not reproduce them. The ill-separated guard also
has no operator equivalent — reconciling the two paths is deferred to
when the operators are wired in.

Identity-of-orbital flows through this module as
:class:`aiida_koopmans.types.VariationalOrbital` — a ``TypedDict``
that is a plain ``dict`` at runtime so ``list[VariationalOrbital]``
survives ``aiida-workgraph``'s storage path. The string form
(``f"up_orb_5"`` etc.) is only ever produced via :func:`map_key_for`
at the per-orbital fan-out boundary; it is never parsed back.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, TypedDict, cast

from aiida_workgraph import dynamic, task

from aiida_koopmans.types import (
    ProjectionBlockId,
    SpinChannel,
    VariationalOrbital,
    map_key_for,
    validate_projection_block_id,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    import numpy as np


class ExpandedAlphas(TypedDict):
    """Per-orbital alpha + error dicts after broadcasting from representatives.

    Keys are :func:`map_key_for` strings — the same labels the
    per-orbital fan-out gather uses. Returned as leaf ``dict`` sockets
    because :func:`assemble_alpha_screening` takes leaf dicts: the
    gather's namespace shape is fully consumed *inside*
    :func:`expand_alphas_by_group`, which packs the broadcast results
    into a flat per-orbital dict ready for the per-spin packing step.
    """

    filled_alphas: dict[str, float]
    empty_alphas: dict[str, float]
    filled_errors: dict[str, float]
    empty_errors: dict[str, float]


# ----------------------------------------------------------------------
# Pure helpers (no AiiDA, no @task)
# ----------------------------------------------------------------------


def enumerate_variational_orbitals(
    *, nelup: int, neldw: int, nbnd: int, spin_polarized: bool
) -> list[VariationalOrbital]:
    """Return every variational orbital the fan-out covers, in canonical order.

    Order matches the per-orbital iteration order: UP filled (1..nelup),
    UP empty (nelup+1..nbnd), DOWN filled (1..neldw), DOWN empty
    (neldw+1..nbnd) for ``spin_polarized=True``. Closed-shell
    (``spin_polarized=False``) emits a single representative channel
    keyed by :attr:`SpinChannel.NONE`: filled (1..nelup) then empty
    (nelup+1..nbnd).

    ``group_id`` and ``representative`` are initialised to "every
    orbital is its own group" — callers running :func:`assign_orbital_groups`
    overwrite them.
    """
    spin_list = [SpinChannel.UP, SpinChannel.DOWN] if spin_polarized else [SpinChannel.NONE]
    out: list[VariationalOrbital] = []
    next_group = 1
    for spin in spin_list:
        n_filled_this_spin = neldw if spin is SpinChannel.DOWN else nelup
        for i in range(n_filled_this_spin):
            out.append(
                VariationalOrbital(
                    spin=spin,
                    index=i + 1,
                    filled=True,
                    group_id=next_group,
                    representative=True,
                )
            )
            next_group += 1
        for i in range(max(0, nbnd - n_filled_this_spin)):
            orb_index = n_filled_this_spin + i + 1
            out.append(
                VariationalOrbital(
                    spin=spin,
                    index=orb_index,
                    filled=False,
                    group_id=next_group,
                    representative=True,
                )
            )
            next_group += 1
    return out


def _assign_groups_fcluster(
    data: np.ndarray,
    default_tol: float,
    revised_tol: float | None = None,
) -> list[int]:
    """Cluster ``data`` (Nx1 ndarray of floats) using complete-linkage hierarchical clustering.

    Recurses with ``0.9 * tol`` when the resulting clusters aren't
    well-separated (any pair of clusters with an inter-cluster gap
    < ``2 * tol``). Raises when ``tol`` shrinks below
    ``0.01 * default_tol``. Returns labels reordered to start at 1.
    """
    import numpy as np

    tol = revised_tol if revised_tol is not None else default_tol
    if tol < 0.01 * default_tol:
        raise RuntimeError(
            "Clustering algorithm failed: could not find well-separated "
            "groups even after shrinking the tolerance to below "
            "1% of the original."
        )

    from scipy.cluster.hierarchy import fcluster, linkage

    Z = linkage(data, method="complete")  # noqa: N806 — scipy convention for the linkage matrix
    labels = fcluster(Z, t=tol, criterion="distance")

    # Reject clusterings where any two clusters are within 2*tol of each
    # other (gap < 2*tol means a single orbital could conceivably belong
    # to either group, so the assignment is ambiguous).
    clustered = [data[labels == i] for i in set(labels)]
    edges = [(np.min(c, axis=0), np.max(c, axis=0)) for c in clustered]
    well_separated = True
    for i, edge in enumerate(edges):
        for j, other in enumerate(edges):
            if i == j:
                continue
            if any(np.abs(e1 - e2).sum() < 2 * tol for e1 in edge for e2 in other):
                well_separated = False
                break
        if not well_separated:
            break

    if not well_separated:
        return _assign_groups_fcluster(data=data, default_tol=default_tol, revised_tol=0.9 * tol)

    # Renumber labels so they start at 1 and increase monotonically.
    mapping: dict[int, int] = {}
    max_label = 0
    for label in labels:
        if label not in mapping:
            max_label += 1
            mapping[label] = max_label
    return [mapping[int(label)] for label in labels]


def _stamp_representatives(orbitals: list[VariationalOrbital]) -> None:
    """In-place: set ``representative`` for one orbital per group.

    Ordering: for filled orbitals, walk per-spin **highest → lowest**
    index; for empty orbitals, walk per-spin **lowest → highest** index.
    The first orbital encountered in each group becomes its
    representative; all others are marked non-representative.

    Raise ``ValueError`` when an orbital's spin is not a
    :class:`SpinChannel` value — a spin outside the walk order would
    otherwise leave its group with no representative at all.
    """
    seen: set[int] = set()
    unknown = {str(o["spin"]) for o in orbitals if o["spin"] not in SpinChannel}
    if unknown:
        raise ValueError(
            f"Unrecognized spin value(s) {sorted(unknown)}; expected one of "
            f"{[s.value for s in SpinChannel]}."
        )

    walk_order: list[VariationalOrbital] = []
    for spin in SpinChannel:
        walk_order.extend(
            sorted(
                (o for o in orbitals if o["spin"] == spin and o["filled"]),
                key=lambda o: -o["index"],
            )
        )
    for spin in SpinChannel:
        walk_order.extend(
            sorted(
                (o for o in orbitals if o["spin"] == spin and not o["filled"]),
                key=lambda o: o["index"],
            )
        )

    for o in orbitals:
        o["representative"] = False
    for o in walk_order:
        if o["group_id"] not in seen:
            seen.add(o["group_id"])
            o["representative"] = True


def _require_group_ids(orbitals: list[VariationalOrbital]) -> None:
    """Raise ``ValueError`` when any orbital carries no ``group_id`` field."""
    missing = [i for i, o in enumerate(orbitals) if "group_id" not in o]
    if missing:
        raise ValueError(
            f"Orbital(s) at position(s) {missing} carry no 'group_id' field, "
            f"so there is no partition to refine."
        )


def _renumber_and_stamp(
    orbitals: list[VariationalOrbital], subkeys: list[Hashable]
) -> list[VariationalOrbital]:
    """Copy ``orbitals`` with group ids assigned canonically from ``subkeys``.

    Orbitals sharing a subkey share a group. Ids are renumbered by first
    appearance in list order (the first orbital's group is 1, ids are
    contiguous), and ``representative`` flags are restamped via
    :func:`_stamp_representatives` so every group carries exactly one
    representative. The input list and its dicts are left untouched.
    """
    out = [cast("VariationalOrbital", dict(o)) for o in orbitals]
    numbering: dict[Hashable, int] = {}
    for o, subkey in zip(out, subkeys, strict=True):
        o["group_id"] = numbering.setdefault(subkey, len(numbering) + 1)
    _stamp_representatives(out)
    return out


def _refine_by_categories(
    orbitals: list[VariationalOrbital], labels: list[Hashable]
) -> list[VariationalOrbital]:
    """Split every existing group by equality of per-orbital labels.

    Shared core of :func:`refine_by_key` and :func:`refine_by_labels`:
    two orbitals stay in the same group only if they already shared one
    *and* their labels compare equal, so a label shared across two
    existing groups never merges them. Raise ``ValueError`` for an
    unhashable label or an orbital without ``group_id``.
    """
    _require_group_ids(orbitals)
    for pos, label in enumerate(labels):
        try:
            hash(label)
        except TypeError as exc:
            raise ValueError(
                f"Unhashable label {label!r} at position {pos}; labels must be hashable."
            ) from exc
    subkeys = cast(
        "list[Hashable]",
        [(o["group_id"], label) for o, label in zip(orbitals, labels, strict=True)],
    )
    return _renumber_and_stamp(orbitals, subkeys)


def refine_by_key(
    orbitals: list[VariationalOrbital],
    key: str,
) -> list[VariationalOrbital]:
    """Split every existing group by equality of one orbital field.

    ``key`` names a :class:`~aiida_koopmans.types.VariationalOrbital`
    field (``"filled"``, ``"spin"``, ``"manifold"``, ...); orbitals in
    the same group whose values for that field differ are separated.
    For categorical labels that live outside the orbital records use
    :func:`refine_by_labels`.

    Return a new list (inputs untouched) with group ids renumbered
    canonically and representatives restamped. Raise ``ValueError`` when
    the field is absent from any orbital or any orbital carries no
    ``group_id``.
    """
    missing = [i for i, o in enumerate(orbitals) if key not in o]
    if missing:
        raise ValueError(
            f"Cannot refine by {key!r}: orbital(s) at position(s) {missing} carry no {key!r} field."
        )
    labels = cast("list[Hashable]", [o[key] for o in orbitals])  # type: ignore[literal-required]
    return _refine_by_categories(orbitals, labels)


def refine_by_labels(
    orbitals: list[VariationalOrbital],
    labels: Sequence[Hashable],
) -> list[VariationalOrbital]:
    """Split every existing group by equality of externally supplied labels.

    ``labels`` is aligned with ``orbitals`` (one hashable label per
    orbital) — the categorical counterpart of :func:`refine_by_scalar`'s
    ``values``. Any hashable values work; only equality matters. For
    example, user-supplied index groups ``[[1, 2], [3, 4]]`` over four
    orbitals become ``labels=[0, 0, 1, 1]``, and per-block provenance
    could be expressed as ``labels=["occ_1", "occ_1", "emp_1", "emp_1"]``.
    A label shared across two existing groups never merges them, which
    gives user-supplied groups intersection semantics.

    Return a new list (inputs untouched) with group ids renumbered
    canonically and representatives restamped. Raise ``ValueError`` when
    the lengths mismatch, a label is unhashable, or any orbital carries
    no ``group_id``.
    """
    label_list = list(labels)
    if len(label_list) != len(orbitals):
        raise ValueError(f"Got {len(label_list)} labels for {len(orbitals)} orbitals.")
    return _refine_by_categories(orbitals, label_list)


def refine_by_scalar(
    orbitals: list[VariationalOrbital],
    values: Sequence[float],
    tol: float,
) -> list[VariationalOrbital]:
    """Split every existing group where its sorted scalar values gap by more than ``tol``.

    Sort each group's members by their value and cut wherever adjacent
    values differ by more than ``tol``; the connected runs become the
    subgroups (single-linkage clustering). The cut is a strict ``>``
    comparison in floating point. Groups are processed independently,
    so a value belonging to another group can never bridge two members
    of this one — apply after all exact refinements
    (:func:`refine_by_key` / :func:`refine_by_labels`).

    ``values`` is aligned with ``orbitals`` (one scalar per orbital,
    e.g. self-Hartree energies or Wannier spreads). Return a new list
    (inputs untouched) with group ids renumbered canonically and
    representatives restamped. Raise ``ValueError`` when ``tol`` is not
    positive, the lengths mismatch, any value is non-finite, or any
    orbital carries no ``group_id``.
    """
    _require_group_ids(orbitals)
    if not tol > 0:
        raise ValueError(f"tol must be positive, got {tol!r}.")
    if len(values) != len(orbitals):
        raise ValueError(f"Got {len(values)} values for {len(orbitals)} orbitals.")
    vals = [float(v) for v in values]
    bad = [i for i, v in enumerate(vals) if not math.isfinite(v)]
    if bad:
        raise ValueError(f"Non-finite scalar value(s) at position(s) {bad}.")

    positions_by_group: dict[int, list[int]] = {}
    for pos, o in enumerate(orbitals):
        positions_by_group.setdefault(o["group_id"], []).append(pos)

    cut_label = [0] * len(orbitals)
    for positions in positions_by_group.values():
        run = 0
        prev: float | None = None
        for pos in sorted(positions, key=lambda p: vals[p]):
            if prev is not None and vals[pos] - prev > tol:
                run += 1
            cut_label[pos] = run
            prev = vals[pos]

    subkeys = cast(
        "list[Hashable]",
        [(o["group_id"], cut_label[pos]) for pos, o in enumerate(orbitals)],
    )
    return _renumber_and_stamp(orbitals, subkeys)


# ----------------------------------------------------------------------
# Public tasks
# ----------------------------------------------------------------------


def ordered_block_specs(blocks: list[ProjectionBlockId]) -> list[ProjectionBlockId]:
    """Return the block records in emitted orbital order.

    The order :func:`initial_orbital_partition` walks: spin channels in
    the canonical walk order (:class:`SpinChannel` declaration order —
    the same fixed sequence :func:`enumerate_variational_orbitals` and
    the representative stamping use), and within each channel every
    occupied block before every empty one, ties kept in input order. Callers that
    pair the emitted orbitals with position-ordered arrays check their
    input against this.
    """
    return [
        spec
        for channel in SpinChannel
        for filled in (True, False)
        for spec in blocks
        if SpinChannel(spec["spin"]) == channel and bool(spec["filled"]) == filled
    ]


@task
def initial_orbital_partition(blocks: list[ProjectionBlockId]) -> list[VariationalOrbital]:
    """Build the coarsest orbital partition consistent with the blocks' exact splits.

    Emit one :class:`~aiida_koopmans.types.VariationalOrbital` per
    Wannier function, in per-channel iwann order — the kcw.x / kcp.x
    orbital order: within each spin channel every occupied block's
    functions (blocks in input order) precede every empty block's, and
    ``index`` is the 1-based running position in that order. Channels
    appear in the module's canonical walk order
    (:func:`ordered_block_specs`), never by list position. Each orbital
    carries ``manifold`` (its block's label), ``filled`` and ``spin``
    from its block; ``group_id`` encodes the coarsest partition
    consistent with the exact splits, obtained by refining a single
    all-orbital group by ``filled``, ``spin`` and ``manifold``
    (:func:`refine_by_key` — exact refinements compose in any order),
    with representatives stamped by the operators' walk-order semantics.
    Block labels are unique within a call and each maps onto a single
    (spin, filling) pair, so the ``manifold`` split alone already
    determines the partition; the ``filled`` / ``spin`` refinements add
    no cuts for such input and instead enforce the boundary against
    hypothetical label reuse. Scalar (tolerance) refinements are
    deliberately not applied here: they belong with the consumers that
    carry the metric.

    Raise ``ValueError`` for a block with a non-positive ``num_wann``.
    """
    for spec in blocks:
        validate_projection_block_id(spec)

    orbitals: list[VariationalOrbital] = []
    counters: dict[SpinChannel, int] = {}
    for spec in ordered_block_specs(blocks):
        channel = SpinChannel(spec["spin"])
        for _ in range(int(spec["num_wann"])):
            counters[channel] = counters.get(channel, 0) + 1
            orbitals.append(
                VariationalOrbital(
                    spin=channel,
                    index=counters[channel],
                    filled=bool(spec["filled"]),
                    group_id=1,
                    # Placeholder: the refinement chain below restamps
                    # representatives; it is the single authority for them.
                    representative=False,
                    manifold=str(spec["label"]),
                )
            )
    for key in ("filled", "spin", "manifold"):
        orbitals = refine_by_key(orbitals, key)
    return orbitals


@task
def extract_self_hartree_from_kcp(output_parameters: dict) -> list[list[float]]:
    """Pull ``self-Hartree`` per-spin / per-band array from a kcp.x ``output_parameters`` dict.

    Thin extractor: the trial KI's ``output_parameters`` can't be
    subscripted at build time (it's a socket-typed Dict), so one
    ``@task`` runs at AiiDA-runtime to walk the ``orbital_data`` sub-
    dict and feed the array into :func:`assign_orbital_groups`. Kept
    tiny and kcp-flavoured because non-kcp workflows will plumb their
    own metric in via a different extractor — :func:`assign_orbital_groups`
    itself is metric-agnostic.
    """
    return output_parameters["orbital_data"]["self-Hartree"]


@task
def spreads_metric_row(spreads: list, expected_count: int | None = None) -> list[list[float]]:
    """Wrap one channel's flat per-orbital spread list into the metric shape.

    :func:`assign_orbital_groups` takes a per-spin ``[nspin][n_orbitals]``
    metric; a DFPT channel's unified ``spreads``
    (the band-ordered, occupied-then-empty
    :func:`~aiida_koopmans.workgraphs.block_wannierize.WannierizeBlocks`
    output — exactly the order kcw.x counts its 1-based ``SCREEN.i_orb``
    orbital index in) is a single row consumed with
    ``spin_polarized=False``. A spin-polarized DFPT chain wraps each
    channel's own spreads separately.

    ``expected_count`` guards the orbital bookkeeping at runtime: a
    mismatch means the wannierized blocks do not cover the manifolds
    kcw.x will screen, and every downstream alpha would be misaligned.
    """
    row = [float(s) for s in spreads]
    if expected_count is not None and len(row) != int(expected_count):
        raise ValueError(
            f"Got {len(row)} Wannier spreads for {int(expected_count)} variational orbitals."
        )
    return [row]


@task
def assign_orbital_groups(
    metric: list[list[float]],
    nelup: int,
    neldw: int,
    nbnd: int,
    spin_polarized: bool,
    tol: float | None,
) -> list[VariationalOrbital]:
    """Cluster variational orbitals by a per-orbital scalar metric.

    ``metric`` is a per-spin, per-band array of shape ``[nspin][nbnd]``
    — typically the trial KI's ``orbital_data["self-Hartree"]``, but
    deliberately agnostic so the same task can be reused with any
    per-orbital quantity (e.g. ``spreads``) by a non-kcp workflow.
    The caller extracts the relevant array upstream and passes it in.

    When ``tol`` is ``None`` (the default), grouping is disabled:
    every orbital becomes its own group and is its own representative.
    This preserves the refine-every-orbital baseline.

    Returns ``list[VariationalOrbital]`` in the canonical order
    produced by :func:`enumerate_variational_orbitals`. Each entry
    carries ``group_id`` + ``representative`` reflecting the
    clustering decision; the other fields (``spin``, ``index``,
    ``filled``) are the orbital's identity.
    """
    orbitals = enumerate_variational_orbitals(
        nelup=nelup, neldw=neldw, nbnd=nbnd, spin_polarized=spin_polarized
    )

    # No grouping: every orbital is its own group + representative.
    if tol is None:
        return orbitals

    import numpy as np

    # Partition orbitals by (spin, filled) — clustering never crosses
    # these boundaries. The trial KI ran nspin=2 so the metric is
    # shape (2, nbnd); closed-shell (``spin_polarized=False``) emits
    # one ``SpinChannel.NONE`` orbital channel and we read off the
    # up-spin row as the representative.
    subsets: dict[tuple[SpinChannel, bool], list[VariationalOrbital]] = {}
    for o in orbitals:
        # ``o["spin"]`` round-trips through AiiDA storage as a plain
        # ``str`` — pass through :class:`SpinChannel` to normalise.
        spin = SpinChannel(o["spin"])
        subsets.setdefault((spin, o["filled"]), []).append(o)

    next_group_offset = 0
    for subset_key in sorted(subsets.keys(), key=lambda k: (k[0].value, not k[1])):
        members = subsets[subset_key]
        if len(members) == 1:
            labels = [1]
        else:
            spin = subset_key[0]
            spin_axis = 0 if spin is SpinChannel.NONE else spin.axis
            data = np.array([[metric[spin_axis][o["index"] - 1]] for o in members])
            labels = _assign_groups_fcluster(data=data, default_tol=tol, revised_tol=tol)
        for o, label in zip(members, labels, strict=True):
            o["group_id"] = label + next_group_offset
        next_group_offset += max(labels)

    _stamp_representatives(orbitals)
    return orbitals


@task
def expand_alphas_by_group(
    *,
    filled_rep_alphas: Annotated[dict | None, dynamic(float)] = None,
    filled_rep_errors: Annotated[dict | None, dynamic(float)] = None,
    empty_rep_alphas: Annotated[dict | None, dynamic(float)] = None,
    empty_rep_errors: Annotated[dict | None, dynamic(float)] = None,
    orbitals: list[VariationalOrbital],
) -> ExpandedAlphas:
    """Broadcast per-representative alphas onto every group member.

    The four ``*_rep_*`` inputs are the flat ``{map_key: float}`` dicts
    gathered out of the per-orbital fan-out loops — they only carry
    entries for the representative orbitals that actually ran a DSCF
    screening. ``orbitals`` is the full ``list[VariationalOrbital]``
    from :func:`assign_orbital_groups` (every orbital with its
    grouping decision).

    Returns flat ``{map_key: float}`` dicts (split into filled / empty)
    carrying one entry per orbital — non-representative members inherit
    their group's representative alpha and error.

    When no grouping ran upstream (every orbital is its own
    representative — the ``tol is None`` short-circuit), this is the
    identity on the inputs modulo the filled/empty split.
    """
    # Build {group_id: (alpha, error)} lookup from the representative
    # gather dicts. Filled and empty representatives live in different
    # input dicts because they scatter to separate fan-out loops; merging
    # by group id is unambiguous because subset partitioning keeps filled
    # and empty in distinct groups.
    rep_by_group: dict[int, tuple[float, float]] = {}
    for o in orbitals:
        if not o["representative"]:
            continue
        key = map_key_for(o)
        if o["filled"]:
            alphas, errors = filled_rep_alphas or {}, filled_rep_errors or {}
        else:
            alphas, errors = empty_rep_alphas or {}, empty_rep_errors or {}
        if key not in alphas:
            # Representative didn't run (e.g. its screening sub-graph
            # short-circuited on an upstream failure). Leave the group
            # un-broadcast; downstream NaN propagation surfaces it.
            continue
        rep_by_group[o["group_id"]] = (
            float(alphas[key]),
            float(errors.get(key, 0.0)),
        )

    filled_alphas: dict[str, float] = {}
    empty_alphas: dict[str, float] = {}
    filled_errors: dict[str, float] = {}
    empty_errors: dict[str, float] = {}
    for o in orbitals:
        key = map_key_for(o)
        if o["group_id"] in rep_by_group:
            alpha_val, err_val = rep_by_group[o["group_id"]]
        else:
            # No representative alpha available — propagate NaN so
            # downstream consumers surface the failure rather than
            # silently using zero.
            alpha_val = float("nan")
            err_val = float("nan")
        if o["filled"]:
            filled_alphas[key] = alpha_val
            filled_errors[key] = err_val
        else:
            empty_alphas[key] = alpha_val
            empty_errors[key] = err_val
    return ExpandedAlphas(
        filled_alphas=filled_alphas,
        empty_alphas=empty_alphas,
        filled_errors=filled_errors,
        empty_errors=empty_errors,
    )
