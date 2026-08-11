"""Unit tests for the projection-block data model (types.py).

Pure-function coverage of the block grouping / merge-filename / w90-kwargs
helpers, plus a drift guard asserting :class:`OrbitalDict` still mirrors
AiiDA's resolved-orbital schema. No daemon / profile needed for the pure
helpers; the parity test loads a profile only to build a real orbital.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.projections import (
    OrbitalDict,
    ProjectionBlockError,
    block_w90_kwargs,
    get_wannier_indices,
    validate_projection_block,
    validate_projection_block_sequence,
)
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.workgraphs.utils.wannier_merge import group_blocks_to_merge, merge_dest_filename
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

    def test_stamp_decides_against_the_wannier_indices(self):
        """An empty block joins the empty manifold whatever indices it takes.

        The Wannier-function indices deliberately contradict the occupancy:
        they sit inside the occupied range, which is what a disentangling
        block's indices can do (they say where the block sits, not which bands
        its Wannier functions came out of). Reading them instead of the stamp
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
        """The merge cannot proceed on a block nobody has classified."""
        block = _explicit("block_1", range(1, 5))
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
        # Four Wannier functions optimised out of six bands.
        validate_projection_block(_explicit("block_1", range(1, 5), num_bands=6, filled=True))

    def test_accepts_a_block_of_unknown_occupancy(self):
        """Occupancy is not part of a block's structural bookkeeping.

        A block built from atomic projectors is well-formed before anything
        has classified it; only a consumer that needs the occupancy refuses.
        """
        validate_projection_block(_automatic("block_1", range(1, 5)))

    def test_rejects_fewer_bands_than_wannier_functions(self):
        block = _explicit("block_1", range(1, 5), num_bands=3, filled=True)
        with pytest.raises(ValueError, match="one band per Wannier function"):
            validate_projection_block(block)

    def test_rejects_gapped_bands(self):
        """An exclusion inside the window the block reads is refused.

        The derivation would read past the excluded bands and yield
        ``[1, 3, 5]``; no construction path builds such a block, so it is
        refused rather than tolerated.
        """
        block = _explicit("block_1", range(1, 4), exclude_bands=[2, 4])
        with pytest.raises(ValueError, match="names bands \\[2, 4\\] inside"):
            validate_projection_block(block)

    def test_rejects_a_gap_among_the_extra_bands(self):
        """An exclusion above the block's own bands, among the extra ones, is refused too.

        The Wannier-function indices never reach the extra bands, so a rule
        read off them alone would pass this block while wannier90 reads a
        gapped band list; the whole read window must be contiguous.
        """
        block = _explicit("block_1", range(1, 5), num_bands=8, exclude_bands=[6])
        with pytest.raises(ValueError, match="names bands \\[6\\] inside"):
            validate_projection_block(block)

    def test_rejects_zero_wannier_functions(self):
        """A block with no Wannier functions describes nothing to Wannierise."""
        from aiida_wannier90_workflows.common.types import WannierProjectionType

        from aiida_koopmans.projections import AutomaticProjectionBlock

        block = AutomaticProjectionBlock(
            label="block_1",
            spin=SpinChannel.NONE,
            num_wann=0,
            num_bands=0,
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        )
        with pytest.raises(ValueError, match="at least one Wannier function"):
            validate_projection_block(block)

    def test_raises_the_typed_class_through_except_valueerror(self):
        """A user fault arrives as its typed class through ``except ValueError``."""
        from aiida_koopmans.projections import BlockBoundaryError, block_occupancy

        block = _explicit("block_1", range(1, 5))
        try:
            block_occupancy(block)
        except ValueError as exc:
            assert type(exc) is BlockBoundaryError
            assert isinstance(exc, ProjectionBlockError)
            assert exc.label == "block_1"
        else:
            pytest.fail("BlockBoundaryError was not raised")

    def test_derivation_invariants_are_untyped(self):
        """A fault only the block derivation can produce carries no advice class.

        The discriminating half of the fault split: the same validator
        family raises the typed classes for user faults, so a plain
        ``ValueError`` here is what keeps the koopmans package from
        attaching projection advice to an internal bug.
        """
        block = _explicit("block_1", range(1, 5), num_bands=3, filled=True)
        with pytest.raises(ValueError, match="report it") as excinfo:
            validate_projection_block(block)
        assert not isinstance(excinfo.value, ProjectionBlockError)


# ----------------------------------------------------------------------
# validate_projection_block_sequence
# ----------------------------------------------------------------------


class TestValidateProjectionBlockSequence:
    def test_accepts_disentanglement_on_the_uppermost_block(self):
        blocks = [
            _explicit("occ", range(1, 5), filled=True),
            _explicit("emp", range(5, 9), filled=False, num_bands=8),
        ]
        validate_projection_block_sequence(blocks)

    def test_rejects_disentanglement_below_the_uppermost_block(self):
        """A lower disentangling block shifts every later block's Wannier indices.

        ``get_wannier_indices`` counts a block's indices off the bands
        the blocks below it read, so the four extra bands this block reads
        would move ``emp`` four places up the ordering while its own
        exclusions keep saying 5-8, and nothing downstream would notice.
        """
        blocks = [
            _explicit("occ", range(1, 5), filled=True, num_bands=8),
            _explicit("emp", range(5, 9), filled=False),
        ]
        with pytest.raises(ValueError, match="uppermost block of its spin channel"):
            validate_projection_block_sequence(blocks)

    def test_rejects_out_of_order_blocks(self):
        """A reversed channel is rejected, naming both blocks.

        This layout used to pass silently: the old check only asked
        whether any block but the last-listed one disentangles, taking the
        list order on trust.
        """
        blocks = [
            _explicit("emp", range(5, 9), filled=False),
            _explicit("occ", range(1, 5), filled=True),
        ]
        with pytest.raises(ValueError, match=r"'occ' starts at band 1, but block 'emp'"):
            validate_projection_block_sequence(blocks)

    def test_rejects_a_reversed_lower_disentangling_block(self):
        """Reversing the list no longer hides lower disentanglement from the check.

        The old rule took the last-listed block as the channel's top, so
        listing ``occ`` (which disentangles) after ``emp`` passed silently.
        The ordering rule rejects the list before disentanglement is judged.
        """
        blocks = [
            _explicit("emp", range(5, 9), filled=False),
            _explicit("occ", range(1, 5), filled=True, num_bands=8),
        ]
        with pytest.raises(ValueError, match="ascending"):
            validate_projection_block_sequence(blocks)

    def test_each_spin_channel_keeps_its_own_uppermost(self):
        """The up channel's disentangling top block is not judged against the down blocks.

        Concatenating the channels puts the up manifold's uppermost block
        in the middle of the list; a rule read off list position alone
        would reject the very layout the collinear route builds.
        """
        blocks = [
            _explicit("occ_up", range(1, 5), spin=SpinChannel.UP, filled=True),
            _explicit("emp_up", range(5, 9), spin=SpinChannel.UP, filled=False, num_bands=8),
            _explicit("occ_down", range(1, 5), spin=SpinChannel.DOWN, filled=True),
            _explicit("emp_down", range(5, 9), spin=SpinChannel.DOWN, filled=False, num_bands=8),
        ]
        validate_projection_block_sequence(blocks)

    def test_rejects_lower_disentanglement_in_the_second_spin_channel(self):
        blocks = [
            _explicit("occ_up", range(1, 5), spin=SpinChannel.UP, filled=True),
            _explicit("emp_up", range(5, 9), spin=SpinChannel.UP, filled=False),
            _explicit("occ_down", range(1, 5), spin=SpinChannel.DOWN, filled=True, num_bands=8),
            _explicit("emp_down", range(5, 9), spin=SpinChannel.DOWN, filled=False),
        ]
        with pytest.raises(ValueError, match="'occ_down'"):
            validate_projection_block_sequence(blocks)

    def test_rejects_a_repeated_label(self):
        """An otherwise-valid layout with a repeated label is rejected, naming it.

        The label is the join key wherever per-block products are keyed
        into a dynamic namespace; two blocks sharing one would collapse
        there silently, last writer winning.
        """
        blocks = [
            _explicit("occ", range(1, 5), filled=True),
            _explicit("occ", range(5, 9), filled=False),
        ]
        with pytest.raises(
            ValueError, match=r"Duplicate projection-block label\(s\) 'occ'"
        ) as excinfo:
            validate_projection_block_sequence(blocks)
        assert not isinstance(excinfo.value, ProjectionBlockError)

    def test_rejects_a_label_repeated_across_spin_channels(self):
        """Uniqueness holds across the whole list, not per spin channel.

        The label-keyed namespaces hold both channels' products, so a
        per-channel check would pass the very collision that matters.
        """
        blocks = [
            _explicit("occ", range(1, 5), spin=SpinChannel.UP, filled=True),
            _explicit("occ", range(1, 5), spin=SpinChannel.DOWN, filled=True),
        ]
        with pytest.raises(ValueError, match=r"Duplicate projection-block label\(s\) 'occ'"):
            validate_projection_block_sequence(blocks)

    def test_a_lone_block_may_disentangle(self):
        validate_projection_block_sequence([_explicit("block_1", range(1, 5), num_bands=10)])

    def test_accepts_no_blocks(self):
        validate_projection_block_sequence([])


# ----------------------------------------------------------------------
# get_wannier_indices
# ----------------------------------------------------------------------


class TestGetWannierIndices:
    def test_block_at_the_bottom_of_the_manifold(self):
        assert get_wannier_indices(_explicit("block_1", range(1, 5))) == [1, 2, 3, 4]

    def test_exclusions_below_shift_the_indices_up(self):
        assert get_wannier_indices(_explicit("block_2", range(5, 9))) == [5, 6, 7, 8]

    def test_extra_bands_stay_out_of_the_indices(self):
        """A disentangling block takes only ``num_wann`` Wannier-function indices.

        The extra bands are what ``num_bands`` counts beyond ``num_wann``,
        and they sit above the block, so the indices are the lowest bands
        it reads. Were the extra bands among them the
        band-to-Wannier-function map would mis-address every function
        above the block.
        """
        block = _explicit("block_2", range(5, 9), num_bands=10)
        assert get_wannier_indices(block) == [5, 6, 7, 8]

    def test_exclusions_above_do_not_add_indices(self):
        """Bands excluded above the block never join its Wannier-function indices."""
        block = _explicit("block_2", range(5, 9), exclude_bands=[1, 2, 3, 4, 9, 10])
        assert get_wannier_indices(block) == [5, 6, 7, 8]


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


def test_unknown_projection_site_raises_the_typed_class(aiida_profile):
    """A ``site`` label matching no atom raises ``ProjectionSiteError``.

    The one projection fault raised before any block exists; the class
    carries the offending label for the koopmans package's advice.
    """
    from aiida.orm import StructureData
    from ase.build import bulk
    from wannier90_input.models.parameters import Projection

    from aiida_koopmans.projections import ProjectionSiteError, projection_num_wann

    structure = StructureData(ase=bulk("Si", "diamond", 5.43))
    projection = Projection(site="Ge", ang_mtm="s")
    with pytest.raises(ProjectionSiteError, match="does not match any atom") as excinfo:
        projection_num_wann(structure, projection)
    assert excinfo.value.site == "Ge"
    assert isinstance(excinfo.value, ProjectionBlockError)


def test_point_site_hosts_one_orbital_set(aiida_profile):
    """A fractional-site projection counts one set of orbitals, not per-atom."""
    from aiida.orm import StructureData
    from ase.build import bulk
    from wannier90_input.models.parameters import Projection

    from aiida_koopmans.projections import projection_num_wann

    structure = StructureData(ase=bulk("Si", "diamond", 5.43))
    point = Projection(fractional_site=[0.25, 0.25, 0.25], ang_mtm="p")
    assert projection_num_wann(structure, point) == 3
