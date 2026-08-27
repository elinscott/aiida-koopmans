"""Tests for the MLWF / projected-WF initialisation pipeline.

Three layers, none of which run a daemon:

* unit tests for the consistency check (invoked via the task's raw
  callable) against the gap/energy thresholds;
* unit tests for the ``dft_dummy`` / Wannier-seeded ``dft_init`` kcp.x
  parameter builders;
* construction-level graph builds of ``MlwfInitialization`` and of a
  periodic-mlwfs ``KoopmansDSCFWorkflow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_koopmans.functionals import Correction
from aiida_koopmans.variational_orbitals import VariationalOrbitalType
from aiida_koopmans.workgraphs.kcp import KcpBaseInputs
from aiida_koopmans.workgraphs.mlwf_init import (
    MlwfInitialization,
    _build_dft_dummy_parameters,
    _build_dft_init_from_wannier_parameters,
    check_wannier_initialization,
)
from tests.fixtures import ozone_projection_blocks as _ozone_blocks

# ----------------------------------------------------------------------
# Consistency check
# ----------------------------------------------------------------------


def _bands_data(*, eigenvalues, occupations):
    from aiida.orm import BandsData

    bands = BandsData()
    eigenvalues = np.array(eigenvalues)
    # (nkpoints, nbands) or, on a collinear run, (nspin, nkpoints, nbands).
    bands.set_kpoints(np.zeros((eigenvalues.shape[-2], 3)))
    bands.set_bands(eigenvalues, occupations=np.array(occupations))
    return bands


def _init_parameters(*, homo=-1.0, lumo=1.03, energies=(-100.0, -100.0)):
    return {
        "homo_energy": homo,
        "lumo_energy": lumo,
        "energy": energies[-1],
        "convergence": {"filled": [{"iteration": 1, "eff_iteration": 1, "Etot": energies[0]}]},
    }


def _run_check(*, bands=None, init_parameters=None):
    if bands is None:
        # PW gap of 2.0 eV: homo at -1.0, lumo at +1.0.
        bands = _bands_data(
            eigenvalues=[[-2.0, -1.0, 1.0], [-2.5, -1.5, 1.5]],
            occupations=[[2.0, 2.0, 0.0], [2.0, 2.0, 0.0]],
        )
    if init_parameters is None:
        init_parameters = _init_parameters()
    return check_wannier_initialization._callable(
        nscf_output_parameters={},
        nscf_bands=bands,
        init_output_parameters=init_parameters,
    )


class TestCheckWannierInitialization:
    def test_consistent_run_returns_report(self, aiida_profile):
        # CP gap 2.03 eV vs PW gap 2.0 eV: within the 2% (0.04 eV) window.
        report = _run_check()
        assert report["pw_gap"] == pytest.approx(2.0)
        assert report["cp_gap"] == pytest.approx(2.03)

    def test_gap_mismatch_raises(self, aiida_profile):
        # CP gap 2.2 eV vs PW gap 2.0 eV: 0.2 > 0.04 tolerance.
        with pytest.raises(ValueError, match="band gaps are not consistent"):
            _run_check(init_parameters=_init_parameters(lumo=1.2))

    def test_energy_drift_raises(self, aiida_profile):
        # |Efin - Eini| = 0.1 > 1e-6 * 100.
        with pytest.raises(ValueError, match="initial and final CP energies"):
            _run_check(init_parameters=_init_parameters(energies=(-100.1, -100.0)))

    def test_tiny_energy_drift_passes(self, aiida_profile):
        report = _run_check(init_parameters=_init_parameters(energies=(-100.00000001, -100.0)))
        assert report["final_energy"] == pytest.approx(-100.0)

    def test_missing_occupations_raises(self, aiida_profile):
        from aiida.orm import BandsData

        bands = BandsData()
        bands.set_kpoints(np.zeros((1, 3)))
        bands.set_bands(np.array([[-1.0, 1.0]]))
        with pytest.raises(ValueError, match="no occupations"):
            _run_check(bands=bands)

    def test_all_bands_occupied_raises(self, aiida_profile):
        bands = _bands_data(eigenvalues=[[-2.0, -1.0]], occupations=[[2.0, 2.0]])
        with pytest.raises(ValueError, match="no empty bands"):
            _run_check(bands=bands)

    def test_collinear_bands_use_cross_channel_extrema(self, aiida_profile):
        # (nspin, nkpoints, nbands): the HOMO comes from the up channel
        # (-1.0) and the LUMO from the down channel (1.0). The cross-channel
        # gap (2.0 eV) differs from both per-channel gaps (2.5 / 2.2 eV), so
        # this pins the kcp.x-matching semantics.
        bands = _bands_data(
            eigenvalues=[[[-2.0, -1.0, 1.5]], [[-2.5, -1.2, 1.0]]],
            occupations=[[[1.0, 1.0, 0.0]], [[1.0, 1.0, 0.0]]],
        )
        report = _run_check(bands=bands)
        assert report["pw_gap"] == pytest.approx(2.0)

    def test_wrong_rank_raises(self, aiida_profile):
        from aiida.orm import BandsData

        # set_bands itself refuses ranks other than 2 and 3, so plant the
        # arrays directly to reach the check's own guard.
        bands = BandsData()
        bands.set_kpoints(np.zeros((1, 3)))
        bands.set_array("bands", np.zeros((2, 2, 1, 3)))
        # Mixed occupations: an all-occupied array would trip the pre-existing
        # no-empty-bands error before the rank check, masking what this pins.
        occupations = np.ones((2, 2, 1, 3))
        occupations[..., -1] = 0.0
        bands.set_array("occupations", occupations)
        with pytest.raises(ValueError, match="band array has shape"):
            _run_check(bands=bands)

    def test_missing_cp_lumo_raises(self, aiida_profile):
        params = _init_parameters()
        params["lumo_energy"] = None
        with pytest.raises(ValueError, match="no HOMO / LUMO"):
            _run_check(init_parameters=params)


# ----------------------------------------------------------------------
# kcp.x parameter builders for the dummy / Wannier-restart steps
# ----------------------------------------------------------------------


_SUPERCELL_BASE = KcpBaseInputs(
    ecutwfc=65.0,
    ecutrho=260.0,
    nspin=2,
    nelec=36,
    ntyp=1,
    mt_correction=False,  # periodic supercell
    nelup=18,
    neldw=18,
    tot_magnetization=None,
)


class TestDftDummyParameters:
    def test_from_scratch_without_outer_loops_or_nbnd(self):
        params = _build_dft_dummy_parameters(_SUPERCELL_BASE)
        assert params["CONTROL"]["restart_mode"] == "from_scratch"
        assert "nbnd" not in params["SYSTEM"]
        assert params["ELECTRONS"]["do_outerloop"] is False
        assert params["ELECTRONS"]["do_outerloop_empty"] is False
        assert "empty_states_maxstep" not in params["ELECTRONS"]

    def test_plain_dft(self):
        params = _build_dft_dummy_parameters(_SUPERCELL_BASE)
        assert params["SYSTEM"]["do_orbdep"] is False
        # EE machinery always on; periodic -> no countercharge.
        assert params["EE"]["which_compensation"] == "none"


class TestDftInitFromWannierParameters:
    def test_restarts_from_wannier(self):
        params = _build_dft_init_from_wannier_parameters(_SUPERCELL_BASE, nbnd=20)
        assert params["CONTROL"]["restart_mode"] == "restart"
        assert params["SYSTEM"]["restart_from_wannier_pwscf"] is True
        assert params["SYSTEM"]["nbnd"] == 20

    def test_outer_loop_on_but_no_empty_minimisation(self):
        # Solids rule: the filled manifold is minimised, the empty manifold
        # stays the folded Wannier functions.
        params = _build_dft_init_from_wannier_parameters(_SUPERCELL_BASE, nbnd=20)
        assert params["ELECTRONS"]["do_outerloop"] is True
        assert params["ELECTRONS"]["do_outerloop_empty"] is False
        assert "empty_states_maxstep" not in params["ELECTRONS"]


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


@pytest.fixture
def kmesh(aiida_profile):
    """Return the explicit k-mesh matching the [2, 1, 1] test kgrid."""
    from aiida.orm import KpointsData

    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([2, 1, 1])
    return kpoints


class TestMlwfInitializationGraphBuild:
    def test_graph_wires_the_five_stages(
        self, mlwf_codes, periodic_ozone_structure, ozone_real_pseudos, kmesh
    ):
        from aiida.orm import List

        from aiida_koopmans.workgraphs.supercell import primitive_to_supercell

        supercell = primitive_to_supercell._callable(periodic_ozone_structure, List(list=[2, 1, 1]))
        wg = MlwfInitialization.build(
            codes={**mlwf_codes, "kcp": mlwf_codes["pw"]},
            structure=periodic_ozone_structure,
            supercell=supercell,
            pseudos=ozone_real_pseudos,
            blocks=_ozone_blocks(),
            kpoints=kmesh,
            kgrid=[2, 1, 1],
            nelec=36,
            nelup=18,
            neldw=18,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=20,
            pseudo_family="unused-here",
        )
        names = [t.name for t in wg.tasks]
        for expected in (
            "wannierize",
            "fold_to_supercell",
            "dft_dummy",
            "dft_init",
            "consistency_check",
        ):
            assert expected in names, names

        # `wg.run()` reconstructs the graph from its serialized form before
        # executing; the nested WannierizeBlocks wiring must survive that.
        from tests.fixtures import assert_graph_roundtrips

        assert_graph_roundtrips(wg)


class TestKoopmansDSCFPeriodicMlwfsBuild:
    def test_outer_graph_takes_the_wannier_init_route(
        self, periodic_ozone_structure, kcp_code, mlwf_codes, ozone_pseudo_family, kmesh
    ):
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        wg = KoopmansDSCFWorkflow.build(
            structure=periodic_ozone_structure,
            pseudo_family=ozone_pseudo_family,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            codes={**mlwf_codes, "kcp": kcp_code},
            blocks=_ozone_blocks(),
            kgrid=[2, 1, 1],
            kpoints=kmesh,
        )
        names = [t.name for t in wg.tasks]
        # The Wannier route replaces the molecular DFT-init chain with the
        # supercell conversion + MlwfInitialization sub-graph.
        assert "make_supercell" in names, names
        assert "wannier_initialization" in names, names
        assert not any("dft_init_nspin" in name for name in names), names
        assert any(name.startswith("ComputeScreeningParameters") for name in names), names
        assert any(name.startswith("RunFinalKI") for name in names), names

    def test_missing_wannier_route_codes_raises_structurally(
        self, periodic_ozone_structure, kcp_code, ozone_pseudo_family, kmesh
    ):
        """The Wannier route builds without its codes; ``run`` catches the gap.

        ``_validate_scope`` no longer checks code membership: entering the
        Wannier route is decided by ``init_orbitals`` alone, and the five
        Wannier-route codes are wired unconditionally into
        ``MlwfInitialization``'s own required ``codes`` spec. Omitting them
        here still builds — the missing members surface as the framework's
        structural missing-input error, naming the nested
        ``wannier_initialization.codes.*`` sockets, not a build-time
        ``ValueError``.
        """
        from aiida_workgraph.errors import MissingRequiredInputsError

        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        wg = KoopmansDSCFWorkflow.build(
            structure=periodic_ozone_structure,
            pseudo_family=ozone_pseudo_family,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            codes={"kcp": kcp_code},
            blocks=_ozone_blocks(),
            kgrid=[2, 1, 1],
            kpoints=kmesh,
        )
        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.check_before_run()
        missing = {entry.socket_path for entry in excinfo.value.missing}
        for member in ("pw", "pw2wannier90", "wannier90", "wann2kcp", "merge_evc"):
            assert any(
                path.endswith(f"wannier_initialization.codes.{member}") for path in missing
            ), (member, missing)


class TestKoopmansDSCFSmoothInterpolationBuild:
    """``smooth_kpoints`` / ``smooth_mp_grid`` add the denser-mesh wannierization.

    ``_wannierize_smooth_mesh`` (``workgraphs/kcp.py``) is otherwise only
    exercised through its ``do_smooth=False`` early return — every other
    ``KoopmansDSCFWorkflow`` build in this module omits ``kpath`` or the
    smooth mesh inputs. These build the outer graph with both, on the same
    periodic-mlwfs route ``TestKoopmansDSCFPeriodicMlwfsBuild`` covers.
    """

    @staticmethod
    def _build(
        periodic_ozone_structure,
        kcp_code,
        mlwf_codes,
        ozone_pseudo_family,
        kmesh,
        labelled_kpath,
        *,
        smooth: bool,
    ):
        from aiida.orm import KpointsData

        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        inputs = {
            "structure": periodic_ozone_structure,
            "pseudo_family": ozone_pseudo_family,
            "ecutwfc": 65.0,
            "ecutrho": 260.0,
            "nbnd": 10,
            "nspin": 2,
            "correction": Correction.KI,
            "init_orbitals": VariationalOrbitalType.MLWFS,
            "codes": {**mlwf_codes, "kcp": kcp_code},
            "blocks": _ozone_blocks(),
            "kgrid": [2, 1, 1],
            "kpoints": kmesh,
            "kpath": labelled_kpath,
        }
        if smooth:
            dense = KpointsData()
            dense.set_kpoints(np.zeros((8, 3)))
            inputs.update(
                smooth_kpoints=dense,
                smooth_mp_grid=[4, 1, 1],
                unfold_and_interpolate={"smooth_int_factor": [2, 1, 1]},
            )
        return KoopmansDSCFWorkflow.build(**inputs)

    def test_a_denser_mesh_adds_the_second_wannierization(
        self,
        periodic_ozone_structure,
        kcp_code,
        mlwf_codes,
        ozone_pseudo_family,
        kmesh,
        labelled_kpath,
    ):
        wg = self._build(
            periodic_ozone_structure,
            kcp_code,
            mlwf_codes,
            ozone_pseudo_family,
            kmesh,
            labelled_kpath,
            smooth=True,
        )
        names = [t.name for t in wg.tasks]
        assert "wannierize_smooth" in names, names

    def test_the_dense_wannierization_keeps_the_coarse_scf_mesh(
        self,
        periodic_ozone_structure,
        kcp_code,
        mlwf_codes,
        ozone_pseudo_family,
        kmesh,
        labelled_kpath,
    ):
        """``scf_kpoints`` stays the primitive route's own coarse ``kpoints``.

        The mp_grid assertion is the discriminating one: passing the
        coarse mesh's own dimensions there instead of ``smooth_mp_grid``
        would build without error but wannierize the wrong grid.
        """
        wg = self._build(
            periodic_ozone_structure,
            kcp_code,
            mlwf_codes,
            ozone_pseudo_family,
            kmesh,
            labelled_kpath,
            smooth=True,
        )
        task = {t.name: t for t in wg.tasks}["wannierize_smooth"]
        assert task.inputs["smooth_mp_grid"].value == [4, 1, 1]
        assert task.inputs["scf_kpoints"].value.uuid == kmesh.uuid

    def test_its_blocks_reach_the_interpolation(
        self,
        periodic_ozone_structure,
        kcp_code,
        mlwf_codes,
        ozone_pseudo_family,
        kmesh,
        labelled_kpath,
    ):
        wg = self._build(
            periodic_ozone_structure,
            kcp_code,
            mlwf_codes,
            ozone_pseudo_family,
            kmesh,
            labelled_kpath,
            smooth=True,
        )
        interpolate = {t.name: t for t in wg.tasks}["interpolate_band_structure"]
        assert interpolate.inputs["smooth_block_wannierizations"]._links

    def test_without_a_denser_mesh_no_second_wannierization_runs(
        self,
        periodic_ozone_structure,
        kcp_code,
        mlwf_codes,
        ozone_pseudo_family,
        kmesh,
        labelled_kpath,
    ):
        """Negative control: a ``kpath`` alone still asks for interpolation, not smoothing."""
        wg = self._build(
            periodic_ozone_structure,
            kcp_code,
            mlwf_codes,
            ozone_pseudo_family,
            kmesh,
            labelled_kpath,
            smooth=False,
        )
        names = [t.name for t in wg.tasks]
        assert "wannierize_smooth" not in names, names
        interpolate = {t.name: t for t in wg.tasks}["interpolate_band_structure"]
        assert not interpolate.inputs["smooth_block_wannierizations"]._links

    def test_the_graph_survives_a_dict_round_trip(
        self,
        periodic_ozone_structure,
        kcp_code,
        mlwf_codes,
        ozone_pseudo_family,
        kmesh,
        labelled_kpath,
    ):
        """The smooth wannierization's blocks output is a typed dynamic namespace.

        Those have broken ``from_dict`` before, and ``wg.run()``
        reconstructs the graph through exactly this round-trip before
        executing anything.
        """
        from tests.fixtures import assert_graph_roundtrips

        wg = self._build(
            periodic_ozone_structure,
            kcp_code,
            mlwf_codes,
            ozone_pseudo_family,
            kmesh,
            labelled_kpath,
            smooth=True,
        )
        assert_graph_roundtrips(wg)


class TestWannierOverridesThreading:
    """``wannier_overrides`` reach the wannierize step unchanged."""

    def test_windows_reach_the_wannierize_task(
        self, mlwf_codes, periodic_ozone_structure, ozone_real_pseudos
    ):
        """The window keywords land on the wannierize task's overrides socket.

        Regression for koopmans#94: the hop between the DSCF initialisation
        and the per-block wannierization — a window lost here never reaches
        any block.
        """
        from aiida.orm import KpointsData, List

        from aiida_koopmans.workgraphs.supercell import primitive_to_supercell
        from tests.fixtures import explicit_block

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([2, 1, 1])
        supercell = primitive_to_supercell._callable(periodic_ozone_structure, List(list=[2, 1, 1]))
        wg = MlwfInitialization.build(
            codes={**mlwf_codes, "kcp": mlwf_codes["pw"]},
            structure=periodic_ozone_structure,
            supercell=supercell,
            pseudos=ozone_real_pseudos,
            blocks=[explicit_block("block_1", range(1, 10), filled=True)],
            kpoints=kpoints,
            kgrid=[2, 1, 1],
            nelec=36,
            nelup=18,
            neldw=18,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=20,
            pseudo_family="unused-here",
            wannier_overrides={"wannier90": {"dis_froz_max": 1.0}},
        )
        overrides = wg.tasks["wannierize"].inputs["overrides"]["wannier90"].value
        assert overrides["dis_froz_max"] == 1.0
