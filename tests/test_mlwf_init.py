"""Tests for the MLWF / projected-WF initialisation pipeline.

Three layers, none of which run a daemon:

* unit tests for the consistency check (invoked via the task's raw
  callable) against the legacy thresholds;
* unit tests for the ``dft_dummy`` / Wannier-seeded ``dft_init`` kcp.x
  parameter builders;
* construction-level graph builds of ``MlwfInitialization`` and of a
  periodic-mlwfs ``KoopmansDSCFWorkflow``.
"""

from __future__ import annotations

import numpy as np
import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.types import (
    Correction,
    ExplicitProjectionBlock,
    SpinChannel,
    VariationalOrbitalType,
)
from aiida_koopmans.workgraphs.kcp import KcpBaseInputs
from aiida_koopmans.workgraphs.mlwf_init import (
    MlwfInitialization,
    _build_dft_dummy_parameters,
    _build_dft_init_from_wannier_parameters,
    check_wannier_initialization,
)

# ----------------------------------------------------------------------
# Consistency check (legacy ``_koopmans_dscf.py:1250-1262``)
# ----------------------------------------------------------------------


def _bands_data(*, eigenvalues, occupations):
    from aiida.orm import BandsData

    bands = BandsData()
    kpoints = np.zeros((len(eigenvalues), 3))
    bands.set_kpoints(kpoints)
    bands.set_bands(np.array(eigenvalues), occupations=np.array(occupations))
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
        assert "EE" not in params  # periodic: no Martyna-Tuckerman


class TestDftInitFromWannierParameters:
    def test_restarts_from_wannier(self):
        params = _build_dft_init_from_wannier_parameters(_SUPERCELL_BASE, nbnd=20)
        assert params["CONTROL"]["restart_mode"] == "restart"
        assert params["SYSTEM"]["restart_from_wannier_pwscf"] is True
        assert params["SYSTEM"]["nbnd"] == 20

    def test_outer_loop_on_but_no_empty_minimisation(self):
        # Legacy solids rule (``_koopmans_dscf.py:1123-1126``): the filled
        # manifold is minimised, the empty manifold stays the folded
        # Wannier functions.
        params = _build_dft_init_from_wannier_parameters(_SUPERCELL_BASE, nbnd=20)
        assert params["ELECTRONS"]["do_outerloop"] is True
        assert params["ELECTRONS"]["do_outerloop_empty"] is False
        assert "empty_states_maxstep" not in params["ELECTRONS"]


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


def _ozone_blocks():
    """Periodic-ozone projections: 9 occupied + 1 empty band, nspin=1."""

    def _block(label, include):
        n = len(include)
        return ExplicitProjectionBlock(
            label=label,
            spin=SpinChannel.NONE,
            num_wann=n,
            num_bands=n,
            include_bands=list(include),
            projection_type=WannierProjectionType.ANALYTIC,
            projections=[],
        )

    return [_block("block_occ", range(1, 10)), _block("block_emp", range(10, 11))]


@pytest.fixture
def mlwf_codes(aiida_localhost):
    """Return stand-in InstalledCode nodes for the full mlwfs-init code set."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    def _code(label: str, entry_point: str):
        try:
            return InstalledCode.collection.get(label=label)
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=aiida_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point,
            ).store()

    return {
        "pw": _code("mlwf-pw", "quantumespresso.pw"),
        "wannier90": _code("mlwf-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("mlwf-p2w", "quantumespresso.pw2wannier90"),
        "projwfc": _code("mlwf-pjw", "quantumespresso.projwfc"),
        "wann2kcp": _code("mlwf-w2k", "koopmans.wann2kcp"),
        "merge_evc": _code("mlwf-merge", "koopmans.merge_evc"),
    }


@pytest.fixture
def ozone_pseudo_family(ozone_real_pseudos):
    """Register (or fetch) a one-pseudo family covering ozone's O kind."""
    from aiida_pseudo.groups.family import PseudoPotentialFamily

    family, _ = PseudoPotentialFamily.collection.get_or_create(label="test-ozone-family")
    if family.count() == 0:
        pseudo = ozone_real_pseudos["O"]
        if not pseudo.is_stored:
            pseudo.store()
        family.add_nodes([pseudo])
    return family.label


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


class TestKoopmansDSCFPeriodicMlwfsBuild:
    @pytest.fixture
    def kcp_code(self, aiida_local_code_factory):
        return aiida_local_code_factory(executable="true", entry_point="koopmans.kcp")

    def test_outer_graph_takes_the_wannier_init_route(
        self, periodic_ozone_structure, kcp_code, mlwf_codes, ozone_pseudo_family, kmesh
    ):
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        wg = KoopmansDSCFWorkflow.build(
            code=kcp_code,
            structure=periodic_ozone_structure,
            pseudo_family=ozone_pseudo_family,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            codes=mlwf_codes,
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
