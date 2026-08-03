"""Unit tests for the screening-equivalence partition operators.

Exercise :func:`refine_by_key` / :func:`refine_by_labels` /
:func:`refine_by_scalar` as pure functions on
``list[VariationalOrbital]`` — no AiiDA profile, no graph.
The graph-level uses of the grouping machinery are covered by
``test_kcp_workgraph`` and ``test_dfpt_workgraph``.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.types import VariationalOrbital
from aiida_koopmans.workgraphs.variational_orbitals import (
    ProjectionBlockId,
    initial_orbital_partition,
    refine_by_key,
    refine_by_labels,
    refine_by_scalar,
)


def orb(
    index: int,
    *,
    filled: bool = True,
    spin: SpinChannel = SpinChannel.NONE,
    group_id: int = 1,
    manifold: str | None = None,
) -> VariationalOrbital:
    """Build one orbital record; default everything into a single group."""
    o = VariationalOrbital(
        spin=spin, index=index, filled=filled, group_id=group_id, representative=False
    )
    if manifold is not None:
        o["manifold"] = manifold
    return o


def groups_of(orbitals: list[VariationalOrbital]) -> dict[int, frozenset[int]]:
    """Map each group id to the frozenset of member positions in the list."""
    members: dict[int, set[int]] = {}
    for pos, o in enumerate(orbitals):
        members.setdefault(o["group_id"], set()).add(pos)
    return {gid: frozenset(positions) for gid, positions in members.items()}


def assert_refines(before: list[VariationalOrbital], after: list[VariationalOrbital]) -> None:
    """Assert every output group is a subset of exactly one input group."""
    input_groups = list(groups_of(before).values())
    for output_group in groups_of(after).values():
        containers = [g for g in input_groups if output_group <= g]
        assert len(containers) == 1, (output_group, input_groups)


class TestRefineByKey:
    def test_splits_by_filling(self):
        orbitals = [orb(1), orb(2), orb(3, filled=False)]
        refined = refine_by_key(orbitals, "filled")
        assert groups_of(refined) == {1: frozenset({0, 1}), 2: frozenset({2})}
        assert_refines(orbitals, refined)

    def test_splits_by_manifold(self):
        orbitals = [
            orb(1, manifold="occ_a"),
            orb(2, manifold="occ_b"),
            orb(3, manifold="occ_a"),
        ]
        refined = refine_by_key(orbitals, "manifold")
        assert groups_of(refined) == {1: frozenset({0, 2}), 2: frozenset({1})}

    def test_exact_refinements_commute(self):
        orbitals = [
            orb(1, spin=SpinChannel.UP),
            orb(2, spin=SpinChannel.UP, filled=False),
            orb(1, spin=SpinChannel.DOWN),
            orb(2, spin=SpinChannel.DOWN, filled=False),
        ]
        filled_then_spin = refine_by_key(refine_by_key(orbitals, "filled"), "spin")
        spin_then_filled = refine_by_key(refine_by_key(orbitals, "spin"), "filled")
        assert filled_then_spin == spin_then_filled

    def test_idempotent(self):
        orbitals = [orb(1), orb(2, filled=False), orb(3, filled=False)]
        once = refine_by_key(orbitals, "filled")
        assert refine_by_key(once, "filled") == once

    def test_does_not_mutate_input(self):
        orbitals = [orb(1), orb(2, filled=False)]
        refine_by_key(orbitals, "filled")
        assert all(o["group_id"] == 1 for o in orbitals)
        assert all(not o["representative"] for o in orbitals)

    def test_missing_field_raises(self):
        orbitals = [orb(1, manifold="occ"), orb(2)]
        with pytest.raises(ValueError, match="manifold"):
            refine_by_key(orbitals, "manifold")

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="flavor"):
            refine_by_key([orb(1)], "flavor")


class TestRefineByLabels:
    def test_explicit_labels_intersect_never_merge(self):
        # Two existing groups; the user label "x" spans both. Refinement
        # splits along the labels within each group but never merges the
        # like-labelled members across the pre-existing boundary.
        orbitals = [
            orb(1, group_id=1),
            orb(2, group_id=1),
            orb(3, group_id=2),
            orb(4, group_id=2),
        ]
        refined = refine_by_labels(orbitals, ["x", "y", "x", "x"])
        assert groups_of(refined) == {
            1: frozenset({0}),
            2: frozenset({1}),
            3: frozenset({2, 3}),
        }
        assert_refines(orbitals, refined)

    def test_idempotent(self):
        orbitals = [orb(1), orb(2), orb(3)]
        labels = ["a", "b", "a"]
        once = refine_by_labels(orbitals, labels)
        assert refine_by_labels(once, labels) == once

    def test_label_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="2 labels for 3 orbitals"):
            refine_by_labels([orb(1), orb(2), orb(3)], ["a", "b"])

    def test_unhashable_label_raises(self):
        with pytest.raises(ValueError, match="Unhashable label"):
            refine_by_labels([orb(1)], [["not", "hashable"]])


class TestRefineByScalar:
    def test_filling_boundary_not_bridged(self):
        # The canonical counterexample: an empty orbital's value sits
        # between two filled orbitals' values. After the exact filling
        # refinement it cannot chain them together — the filled group's
        # own gap (0.4 > tol) splits, and the empty orbital stands alone.
        orbitals = [orb(1), orb(3, filled=False), orb(2)]
        values = [1.0, 1.2, 1.4]
        refined = refine_by_scalar(refine_by_key(orbitals, "filled"), values, tol=0.3)
        assert groups_of(refined) == {
            1: frozenset({0}),
            2: frozenset({1}),
            3: frozenset({2}),
        }

    def test_without_exact_refinement_the_chain_bridges(self):
        # Negative control for the counterexample: on the unrefined
        # partition the 1.0—1.2—1.4 chain (adjacent gaps 0.2 <= tol)
        # keeps all three orbitals in one group.
        orbitals = [orb(1), orb(3, filled=False), orb(2)]
        refined = refine_by_scalar(orbitals, [1.0, 1.2, 1.4], tol=0.3)
        assert groups_of(refined) == {1: frozenset({0, 1, 2})}

    def test_gap_equal_to_tol_does_not_cut(self):
        # Exactly representable floats, so the gap is exactly tol.
        refined = refine_by_scalar([orb(1), orb(2)], [1.0, 1.25], tol=0.25)
        assert groups_of(refined) == {1: frozenset({0, 1})}

    def test_cuts_only_within_each_group(self):
        orbitals = [
            orb(1, group_id=1),
            orb(2, group_id=1),
            orb(3, group_id=2),
            orb(4, group_id=2),
        ]
        refined = refine_by_scalar(orbitals, [0.0, 1.0, 0.4, 0.5], tol=0.3)
        assert groups_of(refined) == {
            1: frozenset({0}),
            2: frozenset({1}),
            3: frozenset({2, 3}),
        }
        assert_refines(orbitals, refined)

    def test_idempotent(self):
        orbitals = [orb(i) for i in range(1, 6)]
        values = [0.0, 0.1, 1.0, 1.05, 2.0]
        once = refine_by_scalar(orbitals, values, tol=0.3)
        assert refine_by_scalar(once, values, tol=0.3) == once

    def test_deterministic_and_canonically_numbered(self):
        orbitals = [orb(1), orb(2), orb(3), orb(4)]
        values = [2.0, 0.0, 2.1, 0.1]
        refined = refine_by_scalar(orbitals, values, tol=0.3)
        assert refined == refine_by_scalar(orbitals, values, tol=0.3)
        # Ids follow first appearance in list order: the first orbital's
        # group is 1 even though its value sorts last within the group.
        assert refined[0]["group_id"] == 1
        assert sorted(groups_of(refined)) == [1, 2]

    def test_value_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="1 values for 2 orbitals"):
            refine_by_scalar([orb(1), orb(2)], [1.0], tol=0.3)

    @pytest.mark.parametrize("tol", [0.0, -0.1])
    def test_nonpositive_tol_raises(self, tol):
        with pytest.raises(ValueError, match="tol must be positive"):
            refine_by_scalar([orb(1)], [1.0], tol=tol)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonfinite_value_raises(self, bad):
        with pytest.raises(ValueError, match="Non-finite"):
            refine_by_scalar([orb(1), orb(2)], [1.0, bad], tol=0.3)


class TestRepresentatives:
    def test_one_representative_per_group(self):
        orbitals = [orb(i, filled=i <= 2) for i in range(1, 6)]
        refined = refine_by_scalar(
            refine_by_key(orbitals, "filled"), [0.0, 0.1, 5.0, 5.1, 9.0], tol=0.3
        )
        for members in groups_of(refined).values():
            reps = [pos for pos in members if refined[pos]["representative"]]
            assert len(reps) == 1

    def test_walk_order_semantics(self):
        # Representatives follow the established walk: highest index for
        # a filled group, lowest index for an empty group.
        orbitals = [
            orb(1),
            orb(2),
            orb(3, filled=False),
            orb(4, filled=False),
        ]
        refined = refine_by_key(orbitals, "filled")
        by_position = {pos: o["representative"] for pos, o in enumerate(refined)}
        assert by_position == {0: False, 1: True, 2: True, 3: False}

    def test_spinor_orbitals_get_representatives(self):
        orbitals = [
            orb(1, spin=SpinChannel.SPINOR),
            orb(2, spin=SpinChannel.SPINOR),
            orb(3, spin=SpinChannel.SPINOR, filled=False),
        ]
        refined = refine_by_key(orbitals, "filled")
        for members in groups_of(refined).values():
            reps = [pos for pos in members if refined[pos]["representative"]]
            assert len(reps) == 1


class TestMalformedSubstrate:
    def test_unknown_spin_raises(self):
        broken = orb(1)
        broken["spin"] = "sideways"  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="sideways"):
            refine_by_key([broken], "filled")

    def test_missing_group_id_raises(self):
        broken = orb(1)
        del broken["group_id"]  # type: ignore[misc]
        with pytest.raises(ValueError, match="group_id"):
            refine_by_key([broken], "filled")
        with pytest.raises(ValueError, match="group_id"):
            refine_by_scalar([broken], [1.0], tol=0.3)


class TestRefinementInvariant:
    @pytest.mark.parametrize(
        "operator",
        [
            lambda orbs: refine_by_key(orbs, "filled"),
            lambda orbs: refine_by_key(orbs, "spin"),
            lambda orbs: refine_by_labels(orbs, ["u", "v", "u", "v", "u", "v"]),
            lambda orbs: refine_by_scalar(orbs, [0.0, 0.2, 0.9, 1.0, 1.05, 3.0], tol=0.3),
        ],
    )
    def test_every_output_group_nests_in_one_input_group(self, operator):
        orbitals = [
            orb(1, spin=SpinChannel.UP, group_id=1),
            orb(2, spin=SpinChannel.UP, filled=False, group_id=1),
            orb(3, spin=SpinChannel.UP, filled=False, group_id=2),
            orb(1, spin=SpinChannel.DOWN, group_id=1),
            orb(2, spin=SpinChannel.DOWN, filled=False, group_id=2),
            orb(3, spin=SpinChannel.DOWN, filled=False, group_id=3),
        ]
        refined = operator(orbitals)
        assert_refines(orbitals, refined)
        # And a second application still refines the first's output.
        assert_refines(refined, operator(refined))


def spec(label: str, num_wann: int, *, filled: bool, spin: SpinChannel = SpinChannel.NONE):
    """Build one reduced block record for :func:`initial_orbital_partition`."""
    return ProjectionBlockId(label=label, spin=spin, filled=filled, num_wann=num_wann)


class TestInitialOrbitalPartition:
    """Unit tests of the emission task via its raw ``._callable``."""

    def test_iwann_order_occupied_then_empty(self):
        """One orbital per WF, indexed 1..N across the occ/emp boundary."""
        result = initial_orbital_partition._callable(
            blocks=[spec("occ_block", 4, filled=True), spec("emp_block", 3, filled=False)]
        )
        assert [o["index"] for o in result] == [1, 2, 3, 4, 5, 6, 7]
        assert [o["filled"] for o in result] == [True] * 4 + [False] * 3
        assert [o["manifold"] for o in result] == ["occ_block"] * 4 + ["emp_block"] * 3
        assert all(o["spin"] == SpinChannel.NONE for o in result)

    def test_empty_before_occupied_input_is_normalized(self):
        """List position never orders the manifolds; occupancy does."""
        result = initial_orbital_partition._callable(
            blocks=[spec("emp_block", 3, filled=False), spec("occ_block", 4, filled=True)]
        )
        assert [o["manifold"] for o in result] == ["occ_block"] * 4 + ["emp_block"] * 3
        assert [o["index"] for o in result] == [1, 2, 3, 4, 5, 6, 7]

    def test_per_channel_indexing_and_channel_order(self):
        """Channels emit in the canonical up-then-down walk, each with its own 1-based iwann."""
        result = initial_orbital_partition._callable(
            blocks=[
                spec("occ_dw", 2, filled=True, spin=SpinChannel.DOWN),
                spec("occ_up", 2, filled=True, spin=SpinChannel.UP),
                spec("emp_dw", 1, filled=False, spin=SpinChannel.DOWN),
                spec("emp_up", 1, filled=False, spin=SpinChannel.UP),
            ]
        )
        assert [o["spin"] for o in result] == [SpinChannel.UP] * 3 + [SpinChannel.DOWN] * 3
        assert [o["index"] for o in result] == [1, 2, 3, 1, 2, 3]
        assert [o["filled"] for o in result] == [True, True, False] * 2

    def test_groups_are_the_filling_spin_manifold_splits(self):
        """The emitted partition matches the operators applied to a single group.

        The oracle is the merged refinement chain itself, run on a
        hand-enumerated single-group substrate of the same orbitals.
        """
        blocks = [
            spec("occ_a", 2, filled=True),
            spec("occ_b", 2, filled=True),
            spec("emp_a", 1, filled=False),
        ]
        result = initial_orbital_partition._callable(blocks=blocks)

        expected = [
            orb(1, manifold="occ_a"),
            orb(2, manifold="occ_a"),
            orb(3, manifold="occ_b"),
            orb(4, manifold="occ_b"),
            orb(5, filled=False, manifold="emp_a"),
        ]
        for key in ("filled", "spin", "manifold"):
            expected = refine_by_key(expected, key)
        assert result == expected
        assert groups_of(result) == {
            1: frozenset({0, 1}),
            2: frozenset({2, 3}),
            3: frozenset({4}),
        }

    def test_representatives_follow_the_walk_order(self):
        """Filled groups elect their highest index; empty groups their lowest."""
        result = initial_orbital_partition._callable(
            blocks=[spec("occ", 3, filled=True), spec("emp", 2, filled=False)]
        )
        reps = [o["index"] for o in result if o["representative"]]
        assert reps == [3, 4]

    def test_emitted_substrate_is_json_pure(self):
        """The list survives a JSON round-trip, the engine's storage shape."""
        import json

        result = initial_orbital_partition._callable(
            blocks=[spec("occ", 2, filled=True), spec("emp", 1, filled=False)]
        )
        assert json.loads(json.dumps(result)) == result

    def test_nonpositive_num_wann_raises(self):
        with pytest.raises(ValueError, match="num_wann"):
            initial_orbital_partition._callable(blocks=[spec("occ", 0, filled=True)])
