"""Unit tests for the projection-block data model (types.py).

Pure-function coverage of the block grouping / merge-filename / w90-kwargs
helpers, plus a drift guard asserting :class:`OrbitalDict` still mirrors
AiiDA's resolved-orbital schema. No daemon / profile needed for the pure
helpers; the parity test loads a profile only to build a real orbital.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.types import (
    OrbitalDict,
    SpinChannel,
    block_w90_kwargs,
    group_blocks_to_merge,
    merge_dest_filename,
    validate_projection_block,
)
from tests.fixtures import automatic_block as _automatic
from tests.fixtures import explicit_block as _explicit

# ----------------------------------------------------------------------
# group_blocks_to_merge
# ----------------------------------------------------------------------


class TestGroupBlocksToMerge:
    def test_silicon_one_occ_one_emp(self):
        # tutorial_2 silicon: a single occupied block + a single empty block.
        blocks = [
            _explicit("block_1", range(1, 5), filled=True),
            _explicit("block_2", range(5, 9), filled=False),
        ]
        groups = group_blocks_to_merge(blocks, {SpinChannel.NONE: 4})
        assert len(groups) == 2
        occ, emp = groups
        assert occ["filled"] is True and [b["label"] for b in occ["blocks"]] == ["block_1"]
        assert emp["filled"] is False and [b["label"] for b in emp["blocks"]] == ["block_2"]

    def test_zno_multi_block_occupied_merged(self):
        # ZnO: four occupied sub-blocks (Zn-s, Zn-p, O-s, Zn-d+O-p) all merge
        # into the occupied manifold; one empty block stands alone.
        blocks = [
            _explicit("block_1", range(1, 5), filled=True),
            _explicit("block_2", range(5, 11), filled=True),
            _explicit("block_3", range(11, 14), filled=True),
            _explicit("block_4", range(14, 19), filled=True),
            _explicit("block_5", range(19, 29), filled=False),
        ]
        groups = group_blocks_to_merge(blocks, {SpinChannel.NONE: 18})
        assert len(groups) == 2
        occ, emp = groups
        assert [b["label"] for b in occ["blocks"]] == ["block_1", "block_2", "block_3", "block_4"]
        assert [b["label"] for b in emp["blocks"]] == ["block_5"]

    def test_spin_polarized_four_groups(self):
        blocks = [
            _explicit("block_1_spin_up", range(1, 5), spin=SpinChannel.UP, filled=True),
            _explicit("block_2_spin_up", range(5, 9), spin=SpinChannel.UP, filled=False),
            _explicit("block_1_spin_down", range(1, 5), spin=SpinChannel.DOWN, filled=True),
            _explicit("block_2_spin_down", range(5, 9), spin=SpinChannel.DOWN, filled=False),
        ]
        groups = group_blocks_to_merge(blocks, {SpinChannel.UP: 4, SpinChannel.DOWN: 4})
        keys = {(g["filled"], g["spin"]) for g in groups}
        assert keys == {
            (True, SpinChannel.UP),
            (False, SpinChannel.UP),
            (True, SpinChannel.DOWN),
            (False, SpinChannel.DOWN),
        }

    def test_automatic_blocks_group_the_same(self):
        # Grouping reads only the common bookkeeping, so automatic (no
        # explicit projections) blocks group identically.
        blocks = [
            _automatic("block_1", range(1, 5), filled=True),
            _automatic("block_2", range(5, 9), filled=False),
        ]
        groups = group_blocks_to_merge(blocks, {SpinChannel.NONE: 4})
        assert [g["filled"] for g in groups] == [True, False]

    def test_stamp_decides_against_the_band_slots(self):
        """An empty block joins the empty manifold however its slots read.

        The slots deliberately contradict the occupancy: they sit inside
        the occupied range, which is what a disentangling block's slots
        can do (they say where the block sits, not which bands its
        Wannier functions came out of). Reading them instead of the stamp
        puts this block in the occupied manifold, and the empty manifold
        comes out of ``merge_evc.x`` missing.
        """
        blocks = [
            _explicit("occ", range(1, 5), filled=True),
            _explicit("emp", [3, 4], filled=False, num_bands=6),
        ]
        groups = group_blocks_to_merge(blocks, {SpinChannel.NONE: 4})
        assert [(g["filled"], [b["label"] for b in g["blocks"]]) for g in groups] == [
            (True, ["occ"]),
            (False, ["emp"]),
        ]

    def test_unstamped_block_raises(self):
        block = _explicit("block_1", range(1, 5))
        del block["filled"]
        with pytest.raises(ValueError, match="occupied or empty"):
            group_blocks_to_merge([block], {SpinChannel.NONE: 4})

    def test_occupied_blocks_must_span_the_occupied_bands(self):
        """The stamps are checked against the electron count, not trusted blindly."""
        blocks = [
            _explicit("block_1", range(1, 5), filled=True),
            _explicit("block_2", range(5, 9), filled=False),
        ]
        with pytest.raises(ValueError, match="span 4 Wannier functions but the channel has 6"):
            group_blocks_to_merge(blocks, {SpinChannel.NONE: 6})

    def test_missing_spin_count_raises(self):
        block = _explicit("block_1", range(1, 5), spin=SpinChannel.UP)
        with pytest.raises(KeyError, match="no entry for spin"):
            group_blocks_to_merge([block], {SpinChannel.NONE: 4})

    def test_preserves_first_seen_order(self):
        # Empty block encountered first -> empty group comes first.
        blocks = [
            _explicit("e", range(5, 9), filled=False),
            _explicit("o", range(1, 5), filled=True),
        ]
        groups = group_blocks_to_merge(blocks, {SpinChannel.NONE: 4})
        assert [g["filled"] for g in groups] == [False, True]


# ----------------------------------------------------------------------
# validate_projection_block
# ----------------------------------------------------------------------


class TestValidateProjectionBlock:
    def test_accepts_a_disentangling_block(self):
        # Four Wannier functions optimised out of six bands: four slots.
        validate_projection_block(_explicit("block_1", range(1, 5), num_bands=6, filled=True))

    def test_rejects_an_unstamped_block(self):
        block = _explicit("block_1", range(1, 5))
        del block["filled"]
        with pytest.raises(ValueError, match="occupied or empty"):
            validate_projection_block(block)

    def test_rejects_slots_widened_into_the_pool(self):
        """A pool belongs in ``num_bands``; widening the slots mis-addresses WFs."""
        block = _explicit("block_1", range(1, 7), num_bands=6, filled=True)
        block["num_wann"] = 4
        with pytest.raises(ValueError, match="names 6 band slots"):
            validate_projection_block(block)

    def test_rejects_fewer_bands_than_wannier_functions(self):
        block = _explicit("block_1", range(1, 5), num_bands=3, filled=True)
        with pytest.raises(ValueError, match="one band per Wannier function"):
            validate_projection_block(block)


# ----------------------------------------------------------------------
# merge_dest_filename
# ----------------------------------------------------------------------


class TestMergeDestFilename:
    def test_occupied(self):
        assert merge_dest_filename(True, 1) == "evc_occupied1.dat"
        assert merge_dest_filename(True, 2) == "evc_occupied2.dat"

    def test_empty(self):
        assert merge_dest_filename(False, 1) == "evc0_empty1.dat"
        assert merge_dest_filename(False, 2) == "evc0_empty2.dat"

    def test_bad_spin_index(self):
        with pytest.raises(ValueError, match="spin_index"):
            merge_dest_filename(True, 0)


# ----------------------------------------------------------------------
# block_w90_kwargs
# ----------------------------------------------------------------------


class TestBlockW90Kwargs:
    def test_explicit_includes_projections(self):
        block = _explicit("block_1", range(1, 5))
        kwargs = block_w90_kwargs(block)
        assert kwargs["num_wann"] == 4
        assert kwargs["num_bands"] == 4
        assert "projections" in kwargs
        assert "exclude_bands" not in kwargs  # nothing excluded
        assert "spin" not in kwargs  # SpinChannel.NONE

    def test_automatic_omits_projections(self):
        block = _automatic("block_1", range(1, 5))
        kwargs = block_w90_kwargs(block)
        assert "projections" not in kwargs
        assert kwargs["num_wann"] == 4

    def test_spin_and_exclude_emitted(self):
        block = _explicit("block_1_spin_up", range(5, 9), spin=SpinChannel.UP)
        block["exclude_bands"] = "1-4"
        kwargs = block_w90_kwargs(block)
        assert kwargs["spin"] == "up"
        assert kwargs["exclude_bands"] == "1-4"


# ----------------------------------------------------------------------
# OrbitalDict drift guard
# ----------------------------------------------------------------------


def test_orbital_dict_mirrors_realhydrogen_schema(aiida_profile):
    """OrbitalDict keys must match AiiDA's resolved-orbital dict exactly.

    Catches an upstream orbital-schema change (new/renamed field) instead
    of letting the TypedDict silently drift out of sync.
    """
    from aiida.orm import StructureData
    from aiida_wannier90.orbitals import generate_projections
    from ase.build import bulk

    structure = StructureData(ase=bulk("Si", "diamond", 5.43))
    orbital_data = generate_projections(
        [{"kind_name": "Si", "ang_mtm_l_list": 1}], structure=structure
    )
    real_keys = set(orbital_data.get_orbitals()[0].get_orbital_dict().keys())
    assert set(OrbitalDict.__annotations__.keys()) == real_keys
