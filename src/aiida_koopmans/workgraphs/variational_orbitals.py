"""Variational-orbital grouping for screening-parameter reuse.

Cluster variational orbitals by a per-orbital scalar (default
self-Hartree energy) so that orbitals close in that value receive a
single representative screening-parameter calculation, with the result
copied onto the rest of the group.

The clustering uses ``scipy.cluster.hierarchy.fcluster`` with complete
linkage. Orbitals are partitioned by ``(spin, filled)`` first — never
grouped across spin channels or across the filled / empty boundary.
Within each subset, an "ill-separated" check (any inter-cluster gap
smaller than ``2 * tol``) triggers a fallback to ``0.9 * tol`` and the
clustering is rerun. If the tolerance shrinks below ``0.01 * default_tol``
the algorithm raises rather than emitting unreliable groups.

Identity-of-orbital flows through this module as
:class:`aiida_koopmans.types.VariationalOrbital` — a ``TypedDict``
that is a plain ``dict`` at runtime so ``list[VariationalOrbital]``
survives ``aiida-workgraph``'s storage path. The string form
(``f"up_orb_5"`` etc.) is only ever produced via :func:`map_key_for`
at the per-orbital fan-out boundary; it is never parsed back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypedDict

from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans.types import SpinChannel, VariationalOrbital, map_key_for

if TYPE_CHECKING:
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
    """
    seen: set[int] = set()
    spin_order = (SpinChannel.UP, SpinChannel.DOWN, SpinChannel.NONE)

    walk_order: list[VariationalOrbital] = []
    for spin in spin_order:
        walk_order.extend(
            sorted(
                (o for o in orbitals if o["spin"] == spin and o["filled"]),
                key=lambda o: -o["index"],
            )
        )
    for spin in spin_order:
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


# ----------------------------------------------------------------------
# Public tasks
# ----------------------------------------------------------------------


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


def _ordered_manifold_keys(entries: dict, manifold: str) -> list[str]:
    """Validate a manifold's block keys and return them in band order.

    ``_wannierize_manifold`` in ``dfpt.py`` keys each block's products
    ``b{i:02d}`` with ``i`` a contiguous zero-based block index,
    so lexicographic key order *is* band order — the convention
    :func:`prepare_kcw_wannier_files` merges by and kcw.x's 1-based
    ``SCREEN.i_orb`` counts in. Guard that invariant explicitly rather
    than trusting a bare ``sorted()`` walk: a refactor of the keying (or
    a dropped block) would otherwise silently reorder or truncate the
    spreads and misalign every downstream alpha.
    """
    keys = sorted(entries)
    expected = [f"b{i:02d}" for i in range(len(keys))]
    if keys != expected:
        raise ValueError(
            f"{manifold}-manifold wannier90 outputs must be keyed as a contiguous "
            f"zero-based ``bNN`` sequence ({expected or ['b00', '...']}); got {keys}."
        )
    return keys


@task
def extract_spreads_from_output_parameters(
    occ_output_parameters: Annotated[dict, dynamic(orm.Dict)],
    emp_output_parameters: Annotated[dict | None, dynamic(orm.Dict)] = None,
) -> list[list[float]]:
    """Pull per-Wannier-function spreads out of parsed wannier90 outputs.

    Sibling of :func:`extract_self_hartree_from_kcp` for the DFPT route:
    consumes each block's wannier90 ``output_parameters`` (the parsed
    socket ``aiida-wannier90``'s parser always emits — the spread
    clustering depends on the spreads, not on the raw retrieved folder)
    and concatenates the final-state spreads into one row. The per-WF
    entries live under ``wannier_functions_output`` — a list of
    ``{wf_ids, wf_centres, wf_spreads}`` dicts with 1-based ``wf_ids``
    and ``wf_spreads`` in Å² (final state; distinct from the
    manifold-total ``Omega_*`` scalars). ``orm.Dict`` inputs arrive as
    plain dicts (aiida-pythonjob's built-in ``Dict`` deserializer).

    Ordering contract: within each manifold the blocks are walked in
    lexicographic key order (the band-order keying convention of
    ``prepare_kcw_wannier_files``), occupied manifold first, then empty —
    exactly the order kcw.x counts its 1-based ``SCREEN.i_orb`` orbital
    index in. Within a block the entries are ordered by ``wf_ids``.

    Returned as a single-row ``[nspin=1][n_orbitals]`` array so it plugs
    into :func:`assign_orbital_groups`'s ``metric`` with
    ``spin_polarized=False`` (a spin-polarized DFPT chain runs one
    extraction per channel on that channel's outputs).
    """

    def block_spreads(parameters: dict, manifold: str) -> list[float]:
        wfs = parameters.get("wannier_functions_output") or []
        if len(wfs) != parameters.get("number_wfs"):
            raise ValueError(
                f"A {manifold}-manifold wannier90 ``output_parameters`` lists "
                f"{len(wfs)} final-state Wannier functions but the run declares "
                f"number_wfs = {parameters.get('number_wfs')}."
            )
        if any("wf_spreads" not in wf for wf in wfs):
            # A wannier90 restart-for-plotting run parses only wf_ids +
            # im_re_ratio per WF (no final-state spread table).
            raise ValueError(
                f"A {manifold}-manifold ``wannier_functions_output`` entry carries "
                "no ``wf_spreads`` — the run did not minimise to a final state "
                "(e.g. a restart-for-plotting run)."
            )
        return [float(wf["wf_spreads"]) for wf in sorted(wfs, key=lambda wf: int(wf["wf_ids"]))]

    spreads: list[float] = []
    for key in _ordered_manifold_keys(occ_output_parameters, "occupied"):
        spreads += block_spreads(occ_output_parameters[key], "occupied")
    emp = emp_output_parameters or {}
    for key in _ordered_manifold_keys(emp, "empty"):
        spreads += block_spreads(emp[key], "empty")
    return [spreads]


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
