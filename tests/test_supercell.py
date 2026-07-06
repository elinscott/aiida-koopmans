"""Unit tests for the primitive → supercell helpers.

Covers the plain scaling helpers and the ``primitive_to_supercell``
calcfunction (invoked directly — calcfunctions run in-process).
"""

from __future__ import annotations

import pytest

from aiida_koopmans.workgraphs.supercell import (
    primitive_to_supercell,
    scale_extensive,
    supercell_size,
)


class TestScalingHelpers:
    def test_supercell_size_is_grid_product(self):
        assert supercell_size([2, 3, 4]) == 24
        assert supercell_size([1, 1, 1]) == 1

    def test_scale_extensive_multiplies(self):
        assert scale_extensive(10, 8) == 80

    def test_scale_extensive_passes_none_through(self):
        assert scale_extensive(None, 8) is None


@pytest.fixture
def tagged_zno_structure(aiida_profile):
    """Return a 2-atom periodic structure with a custom kind name.

    The custom ``Zn1`` kind is the regression guard for the direct
    site-replication: an ASE round-trip would collapse it back to ``Zn``.
    """
    from aiida.orm import StructureData

    structure = StructureData(cell=[[3.25, 0.0, 0.0], [0.0, 3.25, 0.0], [0.0, 0.0, 5.2]], pbc=True)
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Zn", name="Zn1")
    structure.append_atom(position=(1.625, 1.625, 2.6), symbols="O", name="O")
    return structure


class TestPrimitiveToSupercell:
    def _run(self, structure, kgrid):
        from aiida.orm import List

        return primitive_to_supercell._callable(structure, List(list=kgrid))

    def test_identity_grid_reproduces_primitive(self, tagged_zno_structure):
        supercell = self._run(tagged_zno_structure, [1, 1, 1])
        assert supercell.cell == tagged_zno_structure.cell
        assert len(supercell.sites) == len(tagged_zno_structure.sites)

    def test_site_count_and_cell_scale_with_grid(self, tagged_zno_structure):
        kgrid = [2, 1, 3]
        supercell = self._run(tagged_zno_structure, kgrid)
        assert len(supercell.sites) == 6 * len(tagged_zno_structure.sites)
        for n, primitive_vector, super_vector in zip(
            kgrid, tagged_zno_structure.cell, supercell.cell, strict=True
        ):
            assert super_vector == pytest.approx([n * c for c in primitive_vector])

    def test_replication_is_cell_major(self, tagged_zno_structure):
        supercell = self._run(tagged_zno_structure, [1, 1, 2])
        kind_names = [site.kind_name for site in supercell.sites]
        assert kind_names == ["Zn1", "O", "Zn1", "O"]
        # Second copy of the basis shifted by one c lattice vector.
        assert supercell.sites[2].position == pytest.approx((0.0, 0.0, 5.2))
        assert supercell.sites[3].position == pytest.approx((1.625, 1.625, 7.8))

    def test_custom_kind_names_survive(self, tagged_zno_structure):
        supercell = self._run(tagged_zno_structure, [2, 2, 2])
        assert {kind.name for kind in supercell.kinds} == {"Zn1", "O"}

    def test_pbc_preserved(self, tagged_zno_structure):
        supercell = self._run(tagged_zno_structure, [2, 1, 1])
        assert supercell.pbc == tagged_zno_structure.pbc

    def test_rejects_bad_grid(self, tagged_zno_structure):
        with pytest.raises(ValueError, match="three positive integers"):
            self._run(tagged_zno_structure, [2, 2])
        with pytest.raises(ValueError, match="three positive integers"):
            self._run(tagged_zno_structure, [2, 0, 2])
