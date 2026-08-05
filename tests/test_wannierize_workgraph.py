"""Construction-level tests for the full ``Wannierize`` graph (wannier90.py).

Builds the graph (no daemon, no real code execution) and checks the wrapped
``Wannier90WorkChain`` task and the returned output sockets. A run without
disentanglement leaves the ``disentanglement_data`` / ``spread_data`` sockets
absent; the graph's ``NotRequired`` outputs must link cleanly regardless.
"""

from __future__ import annotations

import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.workgraphs.wannier90 import Wannierize
from tests.fixtures import assert_graph_roundtrips, si_external_projector_tables


class TestWannierizeGraphBuild:
    def test_full_graph_eager_builds(self, fake_cutoffs_family, silicon_structure, wannier_codes):
        """The whole ``Wannierize`` graph builds with its ``NotRequired`` outputs."""
        wg = Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
        )
        names = [t.name for t in wg.tasks]
        assert "Wannier90WorkChain" in names

        # Every declared output socket links; the ``NotRequired`` extras
        # (``disentanglement_data`` / ``spread_data``) do not break the linking
        # even though a plain run omits them.
        output_names = [socket._name for socket in wg.outputs]
        for expected in ("scf", "nscf", "wannier90", "wannier90_up", "wannier90_down", "projwfc"):
            assert expected in output_names, output_names

        assert_graph_roundtrips(wg)

    def test_exclude_semicore_with_external_projectors_raises(
        self, fake_cutoffs_family, silicon_structure, wannier_codes, tmp_path
    ):
        """Semicore exclusion cannot combine with label-free external tables."""
        with pytest.raises(ValueError, match="`exclude_semicore` is not supported"):
            Wannierize.build(
                codes=wannier_codes,
                structure=silicon_structure,
                pseudo_family=fake_cutoffs_family.label,
                projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
                external_projectors_path=str(tmp_path),
                external_projectors=si_external_projector_tables(),
                exclude_semicore=True,
            )


class TestKpointMesh:
    """The caller's Brillouin-zone sampling replaces the protocol's."""

    @staticmethod
    def _mesh(dimensions):
        from aiida import orm

        kpoints = orm.KpointsData()
        kpoints.set_kpoints_mesh(dimensions)
        return kpoints

    def _build(self, fake_cutoffs_family, silicon_structure, wannier_codes, **kwargs):
        from aiida_wannier90_workflows.utils.kpoints import get_explicit_kpoints

        mesh = self._mesh([2, 2, 2])
        kwargs.setdefault("kpoints", get_explicit_kpoints(mesh))
        kwargs.setdefault("mp_grid", [2, 2, 2])
        kwargs.setdefault("scf_kpoints", mesh)
        return Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            **kwargs,
        )

    def test_mesh_reaches_scf_nscf_and_win(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        """Scf mesh, nscf k-list and ``mp_grid`` all come from the caller.

        Unset, each is derived from the protocol's ``kpoints_distance``,
        which for silicon is far denser than 2x2x2.
        """
        wg = self._build(fake_cutoffs_family, silicon_structure, wannier_codes)
        inputs = wg.tasks["Wannier90WorkChain"].inputs

        assert inputs["scf"]["kpoints"].value.get_kpoints_mesh()[0] == [2, 2, 2]
        assert inputs["scf"]["kpoints_distance"].value is None

        # One node feeds both, so the nscf and wannier90 k-orderings agree.
        nscf_kpoints = inputs["nscf"]["kpoints"].value
        assert len(nscf_kpoints.get_kpoints()) == 8
        assert nscf_kpoints.uuid == inputs["wannier90"]["wannier90"]["kpoints"].value.uuid

        parameters = inputs["wannier90"]["wannier90"]["parameters"].value.get_dict()
        assert parameters["mp_grid"] == [2, 2, 2]

    def test_unset_mesh_keeps_the_protocol(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        """Stating no mesh leaves the protocol to choose one."""
        wg = Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
        )
        inputs = wg.tasks["Wannier90WorkChain"].inputs
        assert inputs["scf"]["kpoints"].value is None
        assert inputs["scf"]["kpoints_distance"].value is not None

    def test_kpoints_without_mp_grid_raises(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        """An explicit k-list cannot state the dimensions the ``.win`` needs."""
        with pytest.raises(ValueError, match="was given without `mp_grid`"):
            self._build(fake_cutoffs_family, silicon_structure, wannier_codes, mp_grid=None)
