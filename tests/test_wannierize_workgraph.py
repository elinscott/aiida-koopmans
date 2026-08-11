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
from tests.fixtures import (
    assert_graph_roundtrips,
    count_pw_bands_runs,
    si_external_projector_tables,
)


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


class TestBandInterpolation:
    """A bands path switches wannier90's interpolation on; no path leaves it off."""

    @staticmethod
    def _w90_inputs(wg):
        return wg.tasks["Wannier90WorkChain"].inputs["wannier90"]["wannier90"]

    def _build(self, fake_cutoffs_family, silicon_structure, wannier_codes, **kwargs):
        return Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            **kwargs,
        )

    def test_bands_kpoints_sets_bands_plot(
        self, fake_cutoffs_family, silicon_structure, wannier_codes, labelled_kpath
    ):
        """The path reaches the wannier90 step together with ``bands_plot``.

        ``Wannier90WorkChain`` never sets ``bands_plot`` itself, and without
        it wannier90 interpolates nothing, so the keyword must ride with the
        path.
        """
        wg = self._build(
            fake_cutoffs_family, silicon_structure, wannier_codes, bands_kpoints=labelled_kpath
        )
        inputs = self._w90_inputs(wg)
        assert inputs["parameters"].value.get_dict()["bands_plot"] is True
        assert inputs["bands_kpoints"].value.uuid == labelled_kpath.uuid
        assert_graph_roundtrips(wg)

    def test_bands_kpoints_add_the_quality_check_and_projected_dos(
        self, fake_cutoffs_family, silicon_structure, pdos_codes, labelled_kpath
    ):
        """The explicit path adds one pw.x bands run and one projwfc step.

        pw.x samples the same explicit k-list off the workchain's scf
        scratch, so the interpolated and computed bands share their
        k-points one-to-one; projwfc — ``pdos_codes`` carrying its code
        — reads the bands run's scratch, so the projections resolve along
        the path.
        """
        wg = self._build(
            fake_cutoffs_family, silicon_structure, pdos_codes, bands_kpoints=labelled_kpath
        )
        names = [t.name for t in wg.tasks]
        assert count_pw_bands_runs(wg) == 1
        assert names.count("projwfc") == 1

        bands_task = wg.tasks["bands"]
        assert bands_task.inputs["kpoints"].value.uuid == labelled_kpath.uuid
        params = bands_task.inputs["pw"]["parameters"].value.get_dict()
        assert params["CONTROL"]["calculation"] == "bands"
        links = bands_task.inputs["pw"]["parent_folder"]._links
        assert [link.from_task.name for link in links] == ["Wannier90WorkChain"]

        # The workchain builder resolves ``nbnd`` internally, so the bands
        # run must be handed the resolved value explicitly — without it
        # pw.x computes only the occupied bands and the reference curve
        # stops at the valence top.
        nscf_params = (
            wg.tasks["Wannier90WorkChain"].inputs["nscf"]["pw"]["parameters"].value.get_dict()
        )
        assert nscf_params["SYSTEM"]["nbnd"] is not None
        assert params["SYSTEM"]["nbnd"] == nscf_params["SYSTEM"]["nbnd"]

        links = wg.tasks["projwfc"].inputs["projwfc"]["parent_folder"]._links
        assert [link.from_task.name for link in links] == ["bands"]
        assert wg.outputs["bands"]["output_band"]._links
        # The chained step displaces the workchain's own (SCDM-only)
        # projwfc namespace as the graph output's source.
        links = wg.outputs["projwfc"]["Dos"]._links
        assert [link.from_task.name for link in links] == ["projwfc"]
        assert_graph_roundtrips(wg)

    def test_kpoint_path_sets_bands_plot(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        """The Dict form of the path also switches ``bands_plot`` on.

        It stays symbolic (wannier90 discretizes it itself), so no pw.x
        quality-check run can sample it: the ``bands`` / ``projwfc`` steps
        stay out.
        """
        path = {
            "path": [["GAMMA", "X"]],
            "point_coords": {"GAMMA": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0]},
        }
        wg = self._build(fake_cutoffs_family, silicon_structure, wannier_codes, kpoint_path=path)
        inputs = self._w90_inputs(wg)
        assert inputs["parameters"].value.get_dict()["bands_plot"] is True
        value = inputs["kpoint_path"].value
        assert (value.get_dict() if hasattr(value, "get_dict") else dict(value)) == path
        names = [t.name for t in wg.tasks]
        assert "bands" not in names
        assert "projwfc" not in names
        assert_graph_roundtrips(wg)

    def test_no_path_leaves_interpolation_off(
        self, fake_cutoffs_family, silicon_structure, pdos_codes
    ):
        """Negative control: without a path the parameters carry no ``bands_plot``.

        No path also means no quality-check bands run and no projected DOS,
        projwfc code or not: the ``projwfc`` output namespace falls back to
        the wrapped workchain's own (SCDM-only) namespace.
        """
        wg = self._build(fake_cutoffs_family, silicon_structure, pdos_codes)
        inputs = self._w90_inputs(wg)
        assert "bands_plot" not in inputs["parameters"].value.get_dict()
        assert inputs["bands_kpoints"].value is None
        assert inputs["kpoint_path"].value is None
        names = [t.name for t in wg.tasks]
        assert "bands" not in names
        assert "projwfc" not in names
        assert not wg.outputs["bands"]["output_band"]._links
        links = wg.outputs["projwfc"]["Dos"]._links
        assert [link.from_task.name for link in links] == ["Wannier90WorkChain"]


class TestProjectedDosGate:
    """The PP_PSWFC capability gate both wannierize graphs consult."""

    def test_capable_family_passes_silently(
        self, fake_cutoffs_family, silicon_structure, aiida_profile
    ):
        import warnings as warnings_module

        from aiida_koopmans.workgraphs.wannier90 import projected_dos_supported

        with warnings_module.catch_warnings(record=True) as caught:
            warnings_module.simplefilter("always")
            assert projected_dos_supported(fake_cutoffs_family.label, silicon_structure) is True
        assert not [w for w in caught if "projected DOS" in str(w.message)]

    def test_family_without_pswfc_warns_and_skips(
        self, fake_family_without_pswfc, silicon_structure, aiida_profile
    ):
        """A pseudo promising no atomic wavefunctions names itself in the warning."""
        from aiida_koopmans.workgraphs.wannier90 import projected_dos_supported

        with pytest.warns(UserWarning, match=r"pseudopotentials for Si have no `PP_PSWFC`"):
            supported = projected_dos_supported(fake_family_without_pswfc.label, silicon_structure)
        assert supported is False

    def test_unreadable_upf_warns_and_skips(
        self, fake_family_unreadable_upf, silicon_structure, aiida_profile
    ):
        """A header the reader cannot parse skips the pDOS instead of failing."""
        from aiida_koopmans.workgraphs.wannier90 import projected_dos_supported

        with pytest.warns(UserWarning, match=r"UPF files for Si could not be parsed"):
            supported = projected_dos_supported(fake_family_unreadable_upf.label, silicon_structure)
        assert supported is False

    def test_no_family_warns_and_skips(self, silicon_structure, aiida_profile):
        """Without a family label there is nothing to check, so the pDOS skips."""
        from aiida_koopmans.workgraphs.wannier90 import projected_dos_supported

        with pytest.warns(UserWarning, match="No pseudopotential family"):
            assert projected_dos_supported(None, silicon_structure) is False


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
