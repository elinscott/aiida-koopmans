"""Tests for the ΔSCF route's own band-structure wiring in ``workgraphs/kcp.py``.

Two seams: the final KI printing and retrieving its Koopmans Hamiltonians,
and ``_dscf_manifold_specs`` turning the initialisation's ``merge_groups``
partition into the ``ManifoldSpec`` list ``KoopmansBandStructureTask`` fans
out over. The fan-out and merge themselves are route-agnostic and tested in
``test_ui_band_structure.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from aiida import orm

from aiida_koopmans.functionals import Correction
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.variational_orbitals import VariationalOrbitalType
from aiida_koopmans.workgraphs.kcp import _validate_scope
from tests.fixtures import occ_emp_merge_groups

# ----------------------------------------------------------------------
# Seam 1: kcp.x prints and retrieves the Koopmans Hamiltonians
# ----------------------------------------------------------------------


class TestFinalKiWritesTheHamiltonians:
    """``write_hr`` is what makes the one indispensable UI input exist."""

    def _build(self, *, ozone_structure, kcp_code, ozone_pseudo_family, write_hr):
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import RunFinalKI

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        return RunFinalKI.build(
            kcp_code=kcp_code,
            structure=ozone_structure,
            pseudos=family.get_pseudos(structure=ozone_structure),
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            nelec=18,
            nelup=9,
            neldw=9,
            correction=Correction.KI,
            alphas={"filled": {SpinChannel.NONE: [0.6] * 9}, "empty": {SpinChannel.NONE: [0.6]}},
            parent_folder=orm.RemoteData(remote_path="/nonexistent/fake"),
            write_hr=write_hr,
        )

    @staticmethod
    def _kcp_task(wg):
        return next(task for task in wg.tasks if "Kcp" in task.identifier)

    def test_write_hr_asks_kcp_for_the_hamiltonians_and_keeps_them(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        wg = self._build(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            write_hr=True,
        )
        task = self._kcp_task(wg)
        parameters = task.inputs["parameters"].value
        assert parameters["CONTROL"]["write_hr"] is True
        settings = task.inputs["settings"].value.get_dict()
        assert settings["additional_retrieve_list"] == ["ham_occ_*.dat", "ham_emp_*.dat"]

    def test_without_write_hr_nothing_is_printed_or_kept(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Negative control: the default final KI is untouched by the new knob."""
        wg = self._build(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            write_hr=False,
        )
        task = self._kcp_task(wg)
        assert task.inputs["parameters"].value["CONTROL"]["write_hr"] is False
        assert task.inputs["settings"].value is None


class TestHamiltonianFilenames:
    """The names kcp.x writes, per QE ``CPV/write_hamiltonian.f90``."""

    def test_names_follow_the_manifold_and_spin_index(self):
        from aiida_koopmans.workgraphs.kcp_files import kcp_hamiltonian_filename

        assert kcp_hamiltonian_filename(filled=True, spin_index=1) == "ham_occ_1.dat"
        assert kcp_hamiltonian_filename(filled=False, spin_index=1) == "ham_emp_1.dat"
        assert kcp_hamiltonian_filename(filled=True, spin_index=2) == "ham_occ_2.dat"
        assert kcp_hamiltonian_filename(filled=False, spin_index=2) == "ham_emp_2.dat"

    def test_a_third_spin_index_is_rejected(self):
        from aiida_koopmans.workgraphs.kcp_files import kcp_hamiltonian_filename

        with pytest.raises(ValueError, match="spin_index"):
            kcp_hamiltonian_filename(filled=True, spin_index=3)


# ----------------------------------------------------------------------
# Seam 2: turning ``merge_groups`` into the fan-out's ``ManifoldSpec`` list
# ----------------------------------------------------------------------


class TestDscfManifoldSpecs:
    """``_dscf_manifold_specs`` reads the initialisation's own partition."""

    def test_one_spec_per_filling(self):
        from aiida_koopmans.workgraphs.kcp import _dscf_manifold_specs

        specs = _dscf_manifold_specs(occ_emp_merge_groups(), spin_polarized=False)

        by_filled = {spec["filled"]: spec for spec in specs}
        assert by_filled[True]["filename"] == "ham_occ_1.dat"
        assert by_filled[True]["blocks"] == ["occ"]
        assert by_filled[True]["spin"] == SpinChannel.NONE
        assert by_filled[False]["filename"] == "ham_emp_1.dat"
        assert by_filled[False]["blocks"] == ["emp"]

    def test_spin_polarized_names_the_down_channel_spin_index_two(self):
        from aiida_koopmans.workgraphs.kcp import _dscf_manifold_specs

        merge_groups = occ_emp_merge_groups("up") + occ_emp_merge_groups("down")
        merge_groups[0]["blocks"] = [{"label": "occ_up"}]
        merge_groups[1]["blocks"] = [{"label": "emp_up"}]
        merge_groups[2]["blocks"] = [{"label": "occ_down"}]
        merge_groups[3]["blocks"] = [{"label": "emp_down"}]

        specs = _dscf_manifold_specs(merge_groups, spin_polarized=True)

        by_key = {(spec["filled"], spec["spin"]): spec for spec in specs}
        assert by_key[True, SpinChannel.UP]["filename"] == "ham_occ_1.dat"
        assert by_key[True, SpinChannel.DOWN]["filename"] == "ham_occ_2.dat"
        assert by_key[False, SpinChannel.DOWN]["filename"] == "ham_emp_2.dat"

    def test_a_missing_manifold_names_itself(self):
        """Interpolating needs an occupied and an empty manifold per channel."""
        from aiida_koopmans.workgraphs.kcp import _dscf_manifold_specs

        with pytest.raises(ValueError, match="occupied and an empty projection manifold"):
            _dscf_manifold_specs(
                [{"filled": True, "spin": "none", "blocks": [{"label": "occ"}]}],
                spin_polarized=False,
            )


# ----------------------------------------------------------------------
# The workflow-level gate
# ----------------------------------------------------------------------


def _kpath():
    kpath = orm.KpointsData()
    kpath.set_kpoints(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]))
    return kpath


class TestBandPathScope:
    """A ``kpath`` asks for the interpolation; only the Wannier route serves it."""

    def test_the_molecular_route_cannot_serve_a_path(self, ozone_structure):
        with pytest.raises(NotImplementedError, match="Wannier basis"):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                fix_spin_contamination=False,
                structure=ozone_structure,
                kpath=_kpath(),
            )

    def test_the_molecular_route_without_a_path_is_untouched(self, ozone_structure):
        """Negative control: it is the path that the molecular route refuses."""
        _validate_scope(
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            fix_spin_contamination=False,
            structure=ozone_structure,
        )

    def test_the_wannier_route_takes_a_path(self, periodic_ozone_structure, kmesh):
        from tests.fixtures import ozone_projection_blocks

        _validate_scope(
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            fix_spin_contamination=False,
            structure=periodic_ozone_structure,
            blocks=ozone_projection_blocks(),
            kgrid=[2, 1, 1],
            kpoints=kmesh,
            kpath=_kpath(),
        )


class TestInterpolationKnobs:
    @staticmethod
    def _resolve(knobs, **smooth):
        """Resolve ``knobs`` for a run that asks for bands."""
        from aiida_koopmans.workgraphs.kcp import _resolve_band_interpolation_knobs

        return _resolve_band_interpolation_knobs(
            knobs,
            kpath=_kpath(),
            smooth_kpoints=smooth.get("smooth_kpoints"),
            smooth_mp_grid=smooth.get("smooth_mp_grid"),
        )

    @staticmethod
    def _dense_mesh():
        mesh = orm.KpointsData()
        mesh.set_kpoints_mesh([4, 4, 4])
        return mesh

    def test_a_scaled_factor_with_its_mesh_asks_for_smooth_interpolation(self):
        assert self._resolve(
            {"smooth_int_factor": [2, 2, 2]},
            smooth_kpoints=self._dense_mesh(),
            smooth_mp_grid=[4, 4, 4],
        ) == (True, True, True)

    def test_a_scaled_factor_without_its_mesh_names_what_is_missing(self):
        with pytest.raises(ValueError, match="smooth_kpoints and smooth_mp_grid"):
            self._resolve({"smooth_int_factor": [2, 2, 2]})

    def test_a_mesh_without_a_scaled_factor_is_refused(self):
        """A denser mesh nothing interpolates against would take no effect."""
        with pytest.raises(ValueError, match="smooth_int_factor"):
            self._resolve(
                {"smooth_int_factor": [1, 1, 1]},
                smooth_kpoints=self._dense_mesh(),
                smooth_mp_grid=[4, 4, 4],
            )

    def test_an_unscaled_factor_passes_and_the_knobs_come_through(self):
        assert self._resolve({"smooth_int_factor": [1, 1, 1], "do_dos": False}) == (
            True,
            False,
            False,
        )

    def test_the_defaults_are_ws_distance_and_a_dos(self):
        assert self._resolve(None) == (True, True, False)

    def test_knobs_without_a_path_are_refused(self):
        from aiida_koopmans.workgraphs.kcp import _resolve_band_interpolation_knobs

        with pytest.raises(ValueError, match="without a `kpath`"):
            _resolve_band_interpolation_knobs(
                {"do_dos": True}, kpath=None, smooth_kpoints=None, smooth_mp_grid=None
            )

    def test_no_knobs_and_no_path_is_silent(self):
        """Negative control: it is the settings, not the missing path, that raise."""
        from aiida_koopmans.workgraphs.kcp import _resolve_band_interpolation_knobs

        assert _resolve_band_interpolation_knobs(
            None, kpath=None, smooth_kpoints=None, smooth_mp_grid=None
        ) == (True, True, False)


class TestTheWorkflowGatesOnTheBandPath:
    """The whole ΔSCF workflow adds the stage exactly when it is given a path."""

    @staticmethod
    def _build(*, periodic_ozone_structure, kcp_code, mlwf_codes, ozone_pseudo_family, kmesh, **kw):
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow
        from tests.fixtures import ozone_projection_blocks

        return KoopmansDSCFWorkflow.build(
            structure=periodic_ozone_structure,
            pseudo_family=ozone_pseudo_family,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            codes={**mlwf_codes, "kcp": kcp_code},
            blocks=ozone_projection_blocks(),
            kgrid=[2, 1, 1],
            kpoints=kmesh,
            **kw,
        )

    def test_a_path_adds_the_stage_and_prints_the_hamiltonians(
        self, periodic_ozone_structure, kcp_code, mlwf_codes, ozone_pseudo_family, kmesh
    ):
        wg = self._build(
            periodic_ozone_structure=periodic_ozone_structure,
            kcp_code=kcp_code,
            mlwf_codes=mlwf_codes,
            ozone_pseudo_family=ozone_pseudo_family,
            kmesh=kmesh,
            kpath=_kpath(),
        )
        names = [task.name for task in wg.tasks]
        assert "interpolate_band_structure" in names, names
        assert (
            wg.tasks["interpolate_band_structure"].inputs["metadata"]["label"].value
            == "Band interpolation"
        )
        # The stage reads Hamiltonians that exist only because the final KI
        # was asked to print them.
        final_ki = next(task for task in wg.tasks if task.name.startswith("RunFinalKI"))
        assert final_ki.inputs["write_hr"].value is True

    def test_without_a_path_neither_happens(
        self, periodic_ozone_structure, kcp_code, mlwf_codes, ozone_pseudo_family, kmesh
    ):
        """Negative control: the same route, one input short, builds neither."""
        wg = self._build(
            periodic_ozone_structure=periodic_ozone_structure,
            kcp_code=kcp_code,
            mlwf_codes=mlwf_codes,
            ozone_pseudo_family=ozone_pseudo_family,
            kmesh=kmesh,
        )
        names = [task.name for task in wg.tasks]
        assert "interpolate_band_structure" not in names, names
        final_ki = next(task for task in wg.tasks if task.name.startswith("RunFinalKI"))
        assert final_ki.inputs["write_hr"].value is False


class TestInterpolateBandsCentres:
    """The interpolation takes centres, never a ``.wout`` to re-parse."""

    def test_an_unread_centre_is_named(self, aiida_profile, silicon_structure):
        from aiida_koopmans.workgraphs.ui import interpolate_bands

        with pytest.raises(ValueError, match="unread coordinate"):
            interpolate_bands._callable(
                kc_ham_file=orm.SinglefileData.from_string("x"),
                centres=[[0.0, None, 0.0]],
                structure=silicon_structure,
                kpath=_kpath(),
                kgrid=[1, 1, 1],
            )

    def test_centres_that_are_not_three_vectors_are_named(self, aiida_profile, silicon_structure):
        """A per-band table of the wrong width cannot be a set of centres."""
        from aiida_koopmans.workgraphs.ui import interpolate_bands

        with pytest.raises(ValueError, match=r"one \[x, y, z\] per Wannier function"):
            interpolate_bands._callable(
                kc_ham_file=orm.SinglefileData.from_string("x"),
                centres=[[0.0, 0.0]],
                structure=silicon_structure,
                kpath=_kpath(),
                kgrid=[1, 1, 1],
            )
