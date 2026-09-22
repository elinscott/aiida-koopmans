"""Tests for the block contract's join helper.

Order and manifold membership come from the merge groups; a block label
is a lookup key and carries no meaning. The misleading-label cases below
are the discriminating ones: they pass only for an implementation that
reads the groups, and fail for one that sorts keys or reads a prefix.
"""

import pytest

from aiida_koopmans.projections import MergeGroupId, ProjectionBlockId
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.workgraphs.utils.wannier_merge import block_nodes_by_group


def _block(label: str, *, filled: bool = True, num_wann: int = 1) -> ProjectionBlockId:
    return ProjectionBlockId(
        label=label, spin=SpinChannel.NONE.value, filled=filled, num_wann=num_wann
    )


def _group(*labels: str, filled: bool = True) -> MergeGroupId:
    return MergeGroupId(
        filled=filled,
        spin=SpinChannel.NONE.value,
        blocks=[_block(label, filled=filled) for label in labels],
    )


class TestBlockNodesByGroup:
    def test_returns_one_list_per_group_in_band_order(self):
        groups = [_group("occ_1", "occ_2"), _group("emp_1", filled=False)]
        nodes = {"occ_1": "A", "occ_2": "B", "emp_1": "C"}

        assert block_nodes_by_group(groups, nodes) == [["A", "B"], ["C"]]

    def test_band_order_is_the_group_order_not_the_label_order(self):
        """Labels in reverse alphabetical order still come back in band order."""
        groups = [_group("occ_z", "occ_a")]
        nodes = {"occ_a": "A", "occ_z": "Z"}

        assert block_nodes_by_group(groups, nodes) == [["Z", "A"]]

    def test_namespace_key_order_does_not_matter(self):
        groups = [_group("occ_1", "occ_2")]
        forwards = {"occ_1": "A", "occ_2": "B"}
        backwards = {"occ_2": "B", "occ_1": "A"}

        assert block_nodes_by_group(groups, forwards) == block_nodes_by_group(groups, backwards)

    def test_manifold_membership_ignores_the_label(self):
        """A block labelled ``occ_1`` sits in the empty manifold if the group says so."""
        groups = [_group("emp_7"), _group("occ_1", filled=False)]
        nodes = {"occ_1": "A", "emp_7": "B"}

        occupied, empty = block_nodes_by_group(groups, nodes)
        assert occupied == ["B"]
        assert empty == ["A"]

    def test_a_block_used_by_two_groups_is_returned_to_both(self):
        groups = [_group("shared"), _group("shared", filled=False)]

        assert block_nodes_by_group(groups, {"shared": "A"}) == [["A"], ["A"]]

    def test_empty_groups_accept_an_empty_namespace(self):
        assert block_nodes_by_group([], {}) == []

    def test_missing_label_raises_and_names_it(self):
        groups = [_group("occ_1", "occ_2")]

        with pytest.raises(ValueError, match=r"No Wannierization outputs for block") as exc:
            block_nodes_by_group(groups, {"occ_1": "A"})
        assert "occ_2" in str(exc.value)

    def test_unused_entry_raises_and_names_it(self):
        groups = [_group("occ_1")]

        with pytest.raises(ValueError, match=r"which no manifold\s+names") as exc:
            block_nodes_by_group(groups, {"occ_1": "A", "emp_9": "B"})
        assert "emp_9" in str(exc.value)

    def test_labels_are_matched_as_strings(self):
        """Socket-borne labels arrive as proxies, so the join stringifies both sides."""

        class Proxy(str):
            pass

        groups = [MergeGroupId(filled=True, spin="none", blocks=[_block(Proxy("occ_1"))])]

        assert block_nodes_by_group(groups, {"occ_1": "A"}) == [["A"]]


class TestMutantsFail:
    """Implementations that invent order or meaning must fail these tests."""

    @staticmethod
    def _sorts_keys(groups, block_nodes):
        return [[block_nodes[label] for label in sorted(block_nodes)] for _ in groups]

    @staticmethod
    def _sniffs_labels(groups, block_nodes):
        return [
            [
                v
                for k, v in sorted(block_nodes.items())
                if k.startswith("occ" if g["filled"] else "emp")
            ]
            for g in groups
        ]

    def test_key_sorting_mutant_breaks_band_order(self):
        groups = [_group("occ_z", "occ_a")]
        nodes = {"occ_a": "A", "occ_z": "Z"}

        assert block_nodes_by_group(groups, nodes) == [["Z", "A"]]
        assert self._sorts_keys(groups, nodes) == [["A", "Z"]]

    def test_label_sniffing_mutant_breaks_manifold_membership(self):
        groups = [_group("emp_7"), _group("occ_1", filled=False)]
        nodes = {"occ_1": "A", "emp_7": "B"}

        assert block_nodes_by_group(groups, nodes) == [["B"], ["A"]]
        assert self._sniffs_labels(groups, nodes) == [["A"], ["B"]]
