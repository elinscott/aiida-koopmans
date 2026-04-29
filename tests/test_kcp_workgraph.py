"""Unit tests for the kcp.x workgraph builders.

Covers the pure-function building blocks (parameter dicts, scope guards,
utility helpers). A full end-to-end WorkGraph construction test is deferred
to the Phase-5 regression harness, which will have a real SG15 pseudo
family available.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.types import SpinChannel
from aiida_koopmans.utils import count_electrons, filled_and_empty_counts
from aiida_koopmans.workgraphs.kcp import (
    _build_dft_parameters,
    _build_ki_parameters,
    _validate_scope,
)

# ----------------------------------------------------------------------
# _validate_scope — every NotImplementedError path
# ----------------------------------------------------------------------


class TestValidateScope:
    def test_supported_baseline_passes(self, ozone_structure):
        _validate_scope(
            functional="ki",
            init_orbitals="kohn-sham",
            alpha_numsteps=1,
            fix_spin_contamination=False,
            structure=ozone_structure,
        )

    @pytest.mark.parametrize("functional", ["kipz", "pkipz", "dft", ""])
    def test_non_ki_functional_raises(self, ozone_structure, functional):
        with pytest.raises(NotImplementedError, match="functional="):
            _validate_scope(
                functional=functional,
                init_orbitals="kohn-sham",
                alpha_numsteps=1,
                fix_spin_contamination=False,
                structure=ozone_structure,
            )

    @pytest.mark.parametrize("init_orbitals", ["mlwfs", "projwfs", "pz"])
    def test_non_kohn_sham_init_raises(self, ozone_structure, init_orbitals):
        with pytest.raises(NotImplementedError, match="init_orbitals="):
            _validate_scope(
                functional="ki",
                init_orbitals=init_orbitals,
                alpha_numsteps=1,
                fix_spin_contamination=False,
                structure=ozone_structure,
            )

    def test_alpha_numsteps_greater_than_one_raises(self, ozone_structure):
        with pytest.raises(NotImplementedError, match="alpha_numsteps="):
            _validate_scope(
                functional="ki",
                init_orbitals="kohn-sham",
                alpha_numsteps=3,
                fix_spin_contamination=False,
                structure=ozone_structure,
            )

    def test_spin_contamination_raises(self, ozone_structure):
        with pytest.raises(NotImplementedError, match="fix_spin_contamination"):
            _validate_scope(
                functional="ki",
                init_orbitals="kohn-sham",
                alpha_numsteps=1,
                fix_spin_contamination=True,
                structure=ozone_structure,
            )

    def test_periodic_structure_raises(self, periodic_ozone_structure):
        with pytest.raises(NotImplementedError, match="Periodic systems"):
            _validate_scope(
                functional="ki",
                init_orbitals="kohn-sham",
                alpha_numsteps=1,
                fix_spin_contamination=False,
                structure=periodic_ozone_structure,
            )


# ----------------------------------------------------------------------
# Parameter builders
# ----------------------------------------------------------------------


_OZONE_KW = {
    "ecutwfc": 65.0,
    "ecutrho": 260.0,
    "nbnd": 10,
    "nspin": 2,
    "nelec": 18,
    "nelup": 9,
    "neldw": 9,
    "tot_magnetization": None,
    "mt_correction": False,
}

_KI_KW = {**_OZONE_KW, "functional": "ki"}


class TestBuildDftParameters:
    def test_has_expected_namelists(self):
        params = _build_dft_parameters(**_OZONE_KW)
        assert set(params.keys()) == {"CONTROL", "SYSTEM", "ELECTRONS", "IONS"}
        assert "NKSIC" not in params
        assert "EE" not in params

    def test_dft_control_is_from_scratch(self):
        # ndr/ndw are owned by ``KcpCalculation._inject_owned_keys`` (universal
        # 50/60 across all kcp.x runs). The builder shouldn't set them.
        params = _build_dft_parameters(**_OZONE_KW)
        assert params["CONTROL"]["restart_mode"] == "from_scratch"
        assert params["CONTROL"]["calculation"] == "cp"
        assert "ndr" not in params["CONTROL"]
        assert "ndw" not in params["CONTROL"]

    def test_dft_system_no_orbdep(self):
        params = _build_dft_parameters(**_OZONE_KW)
        assert params["SYSTEM"]["do_orbdep"] is False
        assert params["SYSTEM"]["nelec"] == 18
        assert params["SYSTEM"]["nelup"] == 9
        assert params["SYSTEM"]["neldw"] == 9
        assert params["SYSTEM"]["nbnd"] == 10
        assert params["SYSTEM"]["ecutwfc"] == 65.0
        assert params["SYSTEM"]["ecutrho"] == 260.0

    def test_dft_outerloop_enabled(self):
        params = _build_dft_parameters(**_OZONE_KW)
        assert params["ELECTRONS"]["do_outerloop"] is True
        assert params["ELECTRONS"]["do_outerloop_empty"] is True

    def test_conv_thr_scales_with_nelec(self):
        params = _build_dft_parameters(**_OZONE_KW)
        assert params["ELECTRONS"]["conv_thr"] == pytest.approx(1.8e-8)

    def test_nspin_one_skips_spin_keys(self):
        kw = {**_OZONE_KW, "nspin": 1, "nelup": None, "neldw": None}
        params = _build_dft_parameters(**kw)
        assert params["SYSTEM"]["nspin"] == 1
        assert "nelup" not in params["SYSTEM"]
        assert "neldw" not in params["SYSTEM"]


class TestBuildKiParameters:
    def test_has_nksic(self):
        params = _build_ki_parameters(**_KI_KW)
        assert "NKSIC" in params
        assert params["NKSIC"]["which_orbdep"] == "nki"
        assert params["NKSIC"]["odd_nkscalfact"] is True
        assert params["NKSIC"]["odd_nkscalfact_empty"] is True
        assert params["NKSIC"]["do_bare_eigs"] is True

    def test_ki_control_is_restart(self):
        # See ``test_dft_control_is_from_scratch``: ndr/ndw live on the
        # CalcJob, not the builder.
        params = _build_ki_parameters(**_KI_KW)
        assert params["CONTROL"]["restart_mode"] == "restart"
        assert "ndr" not in params["CONTROL"]
        assert "ndw" not in params["CONTROL"]

    def test_ki_enables_orbdep_and_disables_outerloop(self):
        params = _build_ki_parameters(**_KI_KW)
        assert params["SYSTEM"]["do_orbdep"] is True
        assert params["ELECTRONS"]["do_outerloop"] is False
        assert params["ELECTRONS"]["do_outerloop_empty"] is False

    def test_periodic_omits_ee_namelist(self):
        # Periodic systems (mt_correction=False) emit no &EE block; do_ee=False
        # in &SYSTEM keeps kcp.x from trying to read it.
        params = _build_ki_parameters(**_KI_KW)
        assert "EE" not in params
        assert params["SYSTEM"]["do_ee"] is False

    def test_aperiodic_emits_tcc(self):
        kw = {**_KI_KW, "mt_correction": True}
        params = _build_ki_parameters(**kw)
        assert params["EE"]["which_compensation"] == "tcc"
        assert params["SYSTEM"]["do_ee"] is True

    def test_ki_disables_innerloop(self):
        # ``do_innerloop`` is True only for PZ; KI / KIPZ run no inner loop.
        # See legacy decision tree in koopmans/workflows/_koopmans_dscf.py:1129-1138.
        params = _build_ki_parameters(**_KI_KW)
        assert params["NKSIC"]["do_innerloop"] is False

    def test_pz_enables_innerloop(self):
        kw = {**_KI_KW, "functional": "pz"}
        params = _build_ki_parameters(**kw)
        assert params["NKSIC"]["do_innerloop"] is True


# ----------------------------------------------------------------------
# Utility helpers (aiida_koopmans/utils.py)
# ----------------------------------------------------------------------


class TestCountElectrons:
    def test_nspin_two_closed_shell(self, ozone_structure, ozone_pseudos):
        nelec, nelup, neldw = count_electrons(
            ozone_structure, ozone_pseudos, nspin=2, tot_magnetization=None
        )
        assert (nelec, nelup, neldw) == (18, 9, 9)

    def test_nspin_one_returns_none_spin_counts(self, ozone_structure, ozone_pseudos):
        nelec, nelup, neldw = count_electrons(ozone_structure, ozone_pseudos, nspin=1)
        assert (nelec, nelup, neldw) == (18, None, None)

    def test_tot_magnetization_two(self, ozone_structure, ozone_pseudos):
        nelec, nelup, neldw = count_electrons(
            ozone_structure, ozone_pseudos, nspin=2, tot_magnetization=2
        )
        assert (nelec, nelup, neldw) == (18, 10, 8)

    def test_inconsistent_magnetization_raises(self, ozone_structure, ozone_pseudos):
        with pytest.raises(ValueError, match="non-integer spin populations"):
            count_electrons(ozone_structure, ozone_pseudos, nspin=2, tot_magnetization=1)

    def test_non_integer_total_charge_raises(self, ozone_structure, fake_upf):
        pseudos = {"O": fake_upf(z_valence=5.7)}
        with pytest.raises(ValueError, match="Non-integer total valence charge"):
            count_electrons(ozone_structure, pseudos, nspin=2)


class TestFilledAndEmptyCounts:
    def test_closed_shell_nspin_two(self):
        # Ozone DFT: 9 filled + 1 empty per spin channel → 18 filled + 2 empty
        n_filled, n_empty = filled_and_empty_counts(nspin=2, nbnd=10, nelec=18, nelup=9, neldw=9)
        assert (n_filled, n_empty) == (18, 2)

    def test_open_shell_unequal_spins(self):
        # 15 electrons, nelup=8 neldw=7, nbnd=10: empty = (10-8) + (10-7) = 5
        n_filled, n_empty = filled_and_empty_counts(nspin=2, nbnd=10, nelec=15, nelup=8, neldw=7)
        assert (n_filled, n_empty) == (15, 5)

    def test_no_empty_when_nbnd_equals_filled(self):
        n_filled, n_empty = filled_and_empty_counts(nspin=2, nbnd=9, nelec=18, nelup=9, neldw=9)
        assert (n_filled, n_empty) == (18, 0)

    def test_nspin_one(self):
        n_filled, n_empty = filled_and_empty_counts(
            nspin=1, nbnd=10, nelec=18, nelup=None, neldw=None
        )
        assert (n_filled, n_empty) == (9, 1)

    def test_nspin_two_missing_spin_counts_raises(self):
        with pytest.raises(ValueError, match="required when nspin=2"):
            filled_and_empty_counts(nspin=2, nbnd=10, nelec=18, nelup=None, neldw=None)


# ----------------------------------------------------------------------
# Alpha formula — eq. 10 of Nguyen et al. (2018)
# ----------------------------------------------------------------------


class TestComputeAlphaFromDscf:
    """Pin the alpha-update formula against known inputs.

    Reference: legacy ``_koopmans_dscf.py:944`` —
    ``alpha = alpha_guess * (dE - lambda_0) / (lambda_a - lambda_0)``.

    Both energies and lambdas are in eV (the parser converts from Hartree
    via ``qe_tools.CONSTANTS``); units cancel on division.
    """

    def _make_inputs(self, *, energy_trial, energy_perturbed, lam_a, lam_0):
        import numpy as np
        from aiida import orm

        trial = orm.Dict(dict={"energy": energy_trial})
        pert = orm.Dict(dict={"energy": energy_perturbed})
        # Stacked ``(nspin, n, n)`` matching ``KcpParser._parse_lambdas``.
        # nspin=2 here so ``SpinChannel.UP.index == 0`` selects the up channel.
        lambdas = orm.ArrayData()
        lambdas.set_array(
            "lambdas",
            np.array([[[lam_a + 0j]], [[0j]]], dtype=np.complex128),
        )
        bare = orm.ArrayData()
        bare.set_array(
            "lambdas",
            np.array([[[lam_0 + 0j]], [[0j]]], dtype=np.complex128),
        )
        return trial, pert, lambdas, bare

    def _run(self, **kwargs):
        """Invoke the calcfunction via a one-shot WorkGraph.

        The idiomatic aiida-workgraph way to exercise a single task.
        Returns ``(alpha, error)`` as plain floats.
        """
        from aiida_workgraph import WorkGraph

        from aiida_koopmans.workgraphs.kcp import compute_alpha_from_dscf

        wg = WorkGraph("compute_alpha_unit")
        wg.add_task(compute_alpha_from_dscf, name="alpha", **kwargs)
        wg.run()
        return wg.tasks.alpha.outputs.alpha.value.value, wg.tasks.alpha.outputs.error.value.value

    def test_filled_orbital(self, aiida_profile):
        # dE = E_trial - E_perturbed = -1296.0 - (-1290.0) = -6.0
        # alpha = 0.6 * (-6.0 - (-10.0)) / (-8.0 - (-10.0)) = 0.6 * 4 / 2 = 1.2
        # error = |dE - lambda_a| = |-6.0 - (-8.0)| = 2.0
        trial, pert, lambdas, bare = self._make_inputs(
            energy_trial=-1296.0, energy_perturbed=-1290.0, lam_a=-8.0, lam_0=-10.0
        )
        alpha, error = self._run(
            trial_output_parameters=trial,
            perturbed_output_parameters=pert,
            trial_lambdas=lambdas,
            trial_bare_lambdas=bare,
            spin_channel=SpinChannel.UP,
            band_index=0,
            alpha_guess=0.6,
            filled=True,
        )
        assert alpha == pytest.approx(1.2)
        assert error == pytest.approx(2.0)

    def test_empty_orbital_flips_de_sign(self, aiida_profile):
        # For empty: dE = E_perturbed - E_trial = -1290 - (-1296) = +6.0
        # alpha = 0.6 * (6.0 - (-10.0)) / (-8.0 - (-10.0)) = 0.6 * 16 / 2 = 4.8
        # error = |dE - lambda_a| = |6 - (-8)| = 14
        trial, pert, lambdas, bare = self._make_inputs(
            energy_trial=-1296.0, energy_perturbed=-1290.0, lam_a=-8.0, lam_0=-10.0
        )
        alpha, error = self._run(
            trial_output_parameters=trial,
            perturbed_output_parameters=pert,
            trial_lambdas=lambdas,
            trial_bare_lambdas=bare,
            spin_channel=SpinChannel.UP,
            band_index=0,
            alpha_guess=0.6,
            filled=False,
        )
        assert alpha == pytest.approx(4.8)
        assert error == pytest.approx(14.0)
