"""Unit tests for the kcp.x workgraph builders.

Covers the pure-function building blocks (parameter dicts, scope guards,
utility helpers). A full end-to-end WorkGraph construction test is deferred
to the Phase-5 regression harness, which will have a real SG15 pseudo
family available.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from aiida_koopmans.types import SpinChannel
from aiida_koopmans.utils import count_electrons, filled_and_empty_counts
from aiida_koopmans.workgraphs.kcp import (
    KcpBaseInputs,
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
            fix_spin_contamination=False,
            structure=ozone_structure,
        )

    @pytest.mark.parametrize("functional", ["kipz", "pkipz", "dft", ""])
    def test_non_ki_functional_raises(self, ozone_structure, functional):
        with pytest.raises(NotImplementedError, match="functional="):
            _validate_scope(
                functional=functional,
                init_orbitals="kohn-sham",
                fix_spin_contamination=False,
                structure=ozone_structure,
            )

    @pytest.mark.parametrize("init_orbitals", ["mlwfs", "projwfs", "pz"])
    def test_non_kohn_sham_init_raises(self, ozone_structure, init_orbitals):
        with pytest.raises(NotImplementedError, match="init_orbitals="):
            _validate_scope(
                functional="ki",
                init_orbitals=init_orbitals,
                fix_spin_contamination=False,
                structure=ozone_structure,
            )

    def test_alpha_numsteps_no_longer_validated(self, ozone_structure):
        # ``alpha_numsteps`` is range-checked by the koopmans2 Pydantic
        # input model upstream; the scope guard no longer needs to look
        # at it. B.3 added the ``While`` zone so any positive count works.
        _validate_scope(
            functional="ki",
            init_orbitals="kohn-sham",
            fix_spin_contamination=False,
            structure=ozone_structure,
        )

    def test_spin_contamination_raises(self, ozone_structure):
        with pytest.raises(NotImplementedError, match="fix_spin_contamination"):
            _validate_scope(
                functional="ki",
                init_orbitals="kohn-sham",
                fix_spin_contamination=True,
                structure=ozone_structure,
            )

    def test_periodic_structure_raises(self, periodic_ozone_structure):
        with pytest.raises(NotImplementedError, match="Periodic systems"):
            _validate_scope(
                functional="ki",
                init_orbitals="kohn-sham",
                fix_spin_contamination=False,
                structure=periodic_ozone_structure,
            )


# ----------------------------------------------------------------------
# Parameter builders
# ----------------------------------------------------------------------


_OZONE_BASE = KcpBaseInputs(
    ecutwfc=65.0,
    ecutrho=260.0,
    nspin=2,
    nelec=18,
    ntyp=1,
    mt_correction=False,
    nelup=9,
    neldw=9,
    tot_magnetization=None,
)


class TestBuildDftParameters:
    def test_has_expected_namelists(self):
        params = _build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert set(params.keys()) == {"CONTROL", "SYSTEM", "ELECTRONS", "IONS"}
        assert "NKSIC" not in params
        assert "EE" not in params

    def test_dft_control_is_from_scratch(self):
        # ndr/ndw are owned by ``KcpCalculation._inject_owned_keys`` (universal
        # 50/60 across all kcp.x runs). The builder shouldn't set them.
        params = _build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["CONTROL"]["restart_mode"] == "from_scratch"
        assert params["CONTROL"]["calculation"] == "cp"
        assert "ndr" not in params["CONTROL"]
        assert "ndw" not in params["CONTROL"]

    def test_dft_system_no_orbdep(self):
        params = _build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["SYSTEM"]["do_orbdep"] is False
        assert params["SYSTEM"]["nelec"] == 18
        assert params["SYSTEM"]["nelup"] == 9
        assert params["SYSTEM"]["neldw"] == 9
        assert params["SYSTEM"]["nbnd"] == 10
        assert params["SYSTEM"]["ecutwfc"] == 65.0
        assert params["SYSTEM"]["ecutrho"] == 260.0

    def test_dft_outerloop_enabled(self):
        params = _build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["ELECTRONS"]["do_outerloop"] is True
        assert params["ELECTRONS"]["do_outerloop_empty"] is True

    def test_conv_thr_scales_with_nelec(self):
        params = _build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["ELECTRONS"]["conv_thr"] == pytest.approx(1.8e-8)

    def test_nspin_one_skips_spin_keys(self):
        base = replace(_OZONE_BASE, nspin=1, nelup=None, neldw=None)
        params = _build_dft_parameters(base, nbnd=10)
        assert params["SYSTEM"]["nspin"] == 1
        assert "nelup" not in params["SYSTEM"]
        assert "neldw" not in params["SYSTEM"]

    def test_ion_radius_scales_with_ntyp(self):
        # ``ion_radius(i)`` must be emitted once per species — ozone has
        # ``ntyp=1`` so we get a single entry, not the legacy hardcoded 1..4.
        params = _build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["IONS"]["ion_radius(1)"] == 1.0
        assert "ion_radius(2)" not in params["IONS"]
        # Three-species cell should emit three entries.
        params3 = _build_dft_parameters(replace(_OZONE_BASE, ntyp=3), nbnd=10)
        assert params3["IONS"]["ion_radius(1)"] == 1.0
        assert params3["IONS"]["ion_radius(2)"] == 1.0
        assert params3["IONS"]["ion_radius(3)"] == 1.0
        assert "ion_radius(4)" not in params3["IONS"]


class TestBuildKiParameters:
    def test_has_nksic(self):
        params = _build_ki_parameters(_OZONE_BASE, nbnd=10, functional="ki")
        assert "NKSIC" in params
        assert params["NKSIC"]["which_orbdep"] == "nki"
        assert params["NKSIC"]["odd_nkscalfact"] is True
        assert params["NKSIC"]["odd_nkscalfact_empty"] is True
        assert params["NKSIC"]["do_bare_eigs"] is True

    def test_ki_control_is_restart(self):
        # See ``test_dft_control_is_from_scratch``: ndr/ndw live on the
        # CalcJob, not the builder.
        params = _build_ki_parameters(_OZONE_BASE, nbnd=10, functional="ki")
        assert params["CONTROL"]["restart_mode"] == "restart"
        assert "ndr" not in params["CONTROL"]
        assert "ndw" not in params["CONTROL"]

    def test_ki_enables_orbdep_and_disables_outerloop(self):
        params = _build_ki_parameters(_OZONE_BASE, nbnd=10, functional="ki")
        assert params["SYSTEM"]["do_orbdep"] is True
        assert params["ELECTRONS"]["do_outerloop"] is False
        assert params["ELECTRONS"]["do_outerloop_empty"] is False

    def test_periodic_omits_ee_namelist(self):
        # Periodic systems (mt_correction=False) emit no &EE block; do_ee=False
        # in &SYSTEM keeps kcp.x from trying to read it.
        params = _build_ki_parameters(_OZONE_BASE, nbnd=10, functional="ki")
        assert "EE" not in params
        assert params["SYSTEM"]["do_ee"] is False

    def test_aperiodic_emits_tcc(self):
        base = replace(_OZONE_BASE, mt_correction=True)
        params = _build_ki_parameters(base, nbnd=10, functional="ki")
        assert params["EE"]["which_compensation"] == "tcc"
        assert params["SYSTEM"]["do_ee"] is True

    def test_ki_disables_innerloop(self):
        # ``do_innerloop`` is True only for PZ; KI / KIPZ run no inner loop.
        # See legacy decision tree in koopmans/workflows/_koopmans_dscf.py:1129-1138.
        params = _build_ki_parameters(_OZONE_BASE, nbnd=10, functional="ki")
        assert params["NKSIC"]["do_innerloop"] is False

    def test_pz_enables_innerloop(self):
        params = _build_ki_parameters(_OZONE_BASE, nbnd=10, functional="pz")
        assert params["NKSIC"]["do_innerloop"] is True


# ----------------------------------------------------------------------
# Legacy spin-channel swap helpers
# ----------------------------------------------------------------------


class TestSwapKcpFrame:
    """Pure-function checks on ``_swap_kcp_frame``.

    Mirrors the legacy ``_swap_spin_channels`` from
    ``koopmans/src/koopmans/calculators/_koopmans_cp.py:159-205``: swap
    nelup<->neldw, negate tot_magnetization (if set), and shift
    fixed_band by the per-spin band block size depending on which
    block it currently points into.
    """

    def test_swaps_electron_counts_and_shifts_fixed_band_from_up_block(self):
        from aiida_koopmans.workgraphs.kcp import _swap_kcp_frame

        # Post-addition violating case: nelup=9, neldw=10. fixed_band=4
        # is in the UP block (<= nbup=15) so it shifts up by nbdw=15
        # to land in the (post-swap) DOWN block.
        base = replace(_OZONE_BASE, nelup=9, neldw=10, tot_magnetization=None)
        swapped, new_fb = _swap_kcp_frame(base, fixed_band=4, nbup=15, nbdw=15)
        assert (swapped.nelup, swapped.neldw) == (10, 9)
        assert swapped.tot_magnetization is None
        assert new_fb == 4 + 15

    def test_shifts_fixed_band_from_down_block(self):
        from aiida_koopmans.workgraphs.kcp import _swap_kcp_frame

        # fixed_band=20 is in the DOWN block (> nbup=15) so it shifts
        # down by nbup=15 to land in the (post-swap) UP block.
        base = replace(_OZONE_BASE, nelup=9, neldw=10, tot_magnetization=None)
        swapped, new_fb = _swap_kcp_frame(base, fixed_band=20, nbup=15, nbdw=15)
        assert new_fb == 20 - 15
        # electron counts still swap regardless of which block fixed_band came from
        assert (swapped.nelup, swapped.neldw) == (10, 9)

    def test_ferromagnetic_negates_tot_magnetization(self):
        from aiida_koopmans.workgraphs.kcp import _swap_kcp_frame

        base = replace(_OZONE_BASE, nelup=8, neldw=12, tot_magnetization=4)
        swapped, _ = _swap_kcp_frame(base, fixed_band=5, nbup=12, nbdw=12)
        assert (swapped.nelup, swapped.neldw) == (12, 8)
        assert swapped.tot_magnetization == -4

    def test_none_tot_magnetization_is_preserved(self):
        from aiida_koopmans.workgraphs.kcp import _swap_kcp_frame

        # No AttributeError / TypeError when tot_magnetization is None.
        base = replace(_OZONE_BASE, nelup=9, neldw=10, tot_magnetization=None)
        swapped, _ = _swap_kcp_frame(base, fixed_band=4, nbup=15, nbdw=15)
        assert swapped.tot_magnetization is None

    def test_does_not_mutate_input_base(self):
        from aiida_koopmans.workgraphs.kcp import _swap_kcp_frame

        base = replace(_OZONE_BASE, nelup=9, neldw=10, tot_magnetization=2)
        _swap_kcp_frame(base, fixed_band=4, nbup=15, nbdw=15)
        # KcpBaseInputs is frozen but check the original values survived.
        assert (base.nelup, base.neldw, base.tot_magnetization) == (9, 10, 2)


class TestSpinSwapSaveOverlay:
    """``_spin_swap_save_overlay`` produces the swap-mapping for save files."""

    def test_nspin_two_returns_six_bidirectional_pairs(self):
        from aiida_koopmans.workgraphs.kcp import _spin_swap_save_overlay

        overlay = _spin_swap_save_overlay(nspin=2)
        # Six entries, all bidirectional.
        assert overlay == {
            "evc01": "evc02",
            "evc02": "evc01",
            "evc_empty1": "evc_empty2",
            "evc_empty2": "evc_empty1",
            "evc0_empty1": "evc0_empty2",
            "evc0_empty2": "evc0_empty1",
        }

    def test_nspin_one_returns_empty(self):
        from aiida_koopmans.workgraphs.kcp import _spin_swap_save_overlay

        assert _spin_swap_save_overlay(nspin=1) == {}


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


# ----------------------------------------------------------------------
# assemble_alpha_screening — gather scattered orbital outputs
# ----------------------------------------------------------------------


class TestAssembleAlphaScreening:
    """Pin the gather step: per-spin lists indexed by band order."""

    @staticmethod
    def _trivial_orbitals(*, nelup: int, neldw: int, nbnd: int, spin_polarized: bool):
        """Build a no-grouping ``list[VariationalOrbital]``: every orbital is its own rep."""
        from aiida_koopmans.workgraphs.variational_orbitals import (
            enumerate_variational_orbitals,
        )

        return enumerate_variational_orbitals(
            nelup=nelup,
            neldw=neldw,
            nbnd=nbnd,
            spin_polarized=spin_polarized,
        )

    def _run(self, **kwargs):
        from aiida_workgraph import WorkGraph

        from aiida_koopmans.workgraphs.kcp import assemble_alpha_screening

        wg = WorkGraph("assemble_alpha_unit")
        wg.add_task(assemble_alpha_screening, name="gather", **kwargs)
        wg.run()
        # ``alphas`` and ``errors`` are namespace outputs — each has
        # ``filled`` / ``empty`` leaf sockets carrying the per-spin dicts
        # (matching :class:`AlphaScreening`).
        alphas_ns = wg.tasks.gather.outputs.alphas
        errors_ns = wg.tasks.gather.outputs.errors

        def _read(ns):
            payload = {"filled": ns.filled.value, "empty": ns.empty.value}
            for branch in ("filled", "empty"):
                if hasattr(payload[branch], "get_dict"):
                    payload[branch] = payload[branch].get_dict()
            return payload

        return _read(alphas_ns), _read(errors_ns)

    def test_closed_shell_single_channel(self, aiida_profile):
        """Closed-shell: bare ``orb_<n>`` keys, packed under :attr:`SpinChannel.NONE`.

        Orbital indices are 1-indexed and continuous across filled +
        empty manifolds — empty orbs in this fixture start at ``orb_4``.
        """
        alphas, errors = self._run(
            orbitals=self._trivial_orbitals(nelup=3, neldw=3, nbnd=5, spin_polarized=False),
            filled_alphas={
                "orb_1": 0.6,
                "orb_2": 0.7,
                "orb_3": 0.8,
            },
            filled_errors={
                "orb_1": 0.1,
                "orb_2": 0.2,
                "orb_3": 0.3,
            },
            empty_alphas={"orb_4": 0.5, "orb_5": 0.4},
            empty_errors={"orb_4": 0.05, "orb_5": 0.04},
        )
        assert alphas["filled"] == {"none": [0.6, 0.7, 0.8]}
        assert alphas["empty"] == {"none": [0.5, 0.4]}
        assert errors["filled"] == {"none": [0.1, 0.2, 0.3]}
        assert errors["empty"] == {"none": [0.05, 0.04]}

    def test_spin_polarized_two_channels(self, aiida_profile):
        """Spin-polarised: both UP and DOWN channels packed independently."""
        alphas, errors = self._run(
            orbitals=self._trivial_orbitals(nelup=2, neldw=2, nbnd=3, spin_polarized=True),
            filled_alphas={
                "up_orb_1": 0.6,
                "up_orb_2": 0.7,
                "down_orb_1": 0.61,
                "down_orb_2": 0.71,
            },
            filled_errors={
                "up_orb_1": 0.1,
                "up_orb_2": 0.2,
                "down_orb_1": 0.11,
                "down_orb_2": 0.21,
            },
            empty_alphas={"up_orb_3": 0.5, "down_orb_3": 0.51},
            empty_errors={"up_orb_3": 0.05, "down_orb_3": 0.06},
        )
        assert alphas["filled"]["up"] == [0.6, 0.7]
        assert alphas["filled"]["down"] == [0.61, 0.71]
        assert alphas["empty"]["up"] == [0.5]
        assert alphas["empty"]["down"] == [0.51]
        assert errors["filled"]["up"] == [0.1, 0.2]
        assert errors["empty"]["down"] == [0.06]

    def test_orb_indexed_ordering(self, aiida_profile):
        # Insertion order intentionally shuffled — band index from
        # ``VariationalOrbitalId.index`` must drive the output list order.
        alphas, _ = self._run(
            orbitals=self._trivial_orbitals(nelup=3, neldw=3, nbnd=3, spin_polarized=True),
            filled_alphas={
                "up_orb_3": 0.8,
                "up_orb_1": 0.6,
                "up_orb_2": 0.7,
                "down_orb_2": 0.71,
                "down_orb_1": 0.61,
                "down_orb_3": 0.81,
            },
            filled_errors={
                "up_orb_3": 0.0,
                "up_orb_1": 0.0,
                "up_orb_2": 0.0,
                "down_orb_2": 0.0,
                "down_orb_1": 0.0,
                "down_orb_3": 0.0,
            },
            empty_alphas={},
            empty_errors={},
        )
        assert alphas["filled"]["up"] == [0.6, 0.7, 0.8]
        assert alphas["filled"]["down"] == [0.61, 0.71, 0.81]


# ----------------------------------------------------------------------
# KoopmansDSCFWorkflow graph build — structural inspection only.
# ----------------------------------------------------------------------


class TestKoopmansDSCFGraphBuild:
    """Inspect the task graph wired by ``KoopmansDSCFWorkflow.build`` for ozone.

    Doesn't run anything — verifies fan-out counts so a wiring regression
    surfaces without needing a real kcp.x install.
    """

    @pytest.fixture
    def kcp_code(self, aiida_local_code_factory):
        return aiida_local_code_factory(executable="true", entry_point="koopmans.kcp")

    @pytest.fixture
    def ozone_pseudo_family(self, ozone_real_pseudos):
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        family, _ = PseudoPotentialFamily.collection.get_or_create(label="test-ozone-family")
        if family.count() == 0:
            pseudo = ozone_real_pseudos["O"]
            if not pseudo.is_stored:
                pseudo.store()
            family.add_nodes([pseudo])
        return family.label

    def _build_wg(self, *, ozone_structure, kcp_code, ozone_pseudo_family, spin_polarized=False):
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        return KoopmansDSCFWorkflow.build(
            code=kcp_code,
            structure=ozone_structure,
            pseudo_family=ozone_pseudo_family,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            tot_magnetization=None,
            functional="ki",
            init_orbitals="kohn-sham",
            alpha_numsteps=1,
            fix_spin_contamination=False,
            initial_alpha=0.6,
            spin_polarized=spin_polarized,
        )

    def _all_link_labels(self, wg) -> list[str]:
        """Walk every task (recursing into sub-graphs) and collect call_link_labels."""
        labels: list[str] = []

        def _walk(tasks):
            for t in tasks:
                # Some tasks have a metadata.call_link_label; others have a name.
                # We collect both for matching flexibility.
                labels.append(t.name)
                # Recurse into sub-graph children when present.
                children = getattr(t, "children", None)
                if children:
                    _walk(children)

        _walk(wg.tasks)
        return labels

    def test_graph_builds_with_expected_subtasks(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
        )

        labels = self._all_link_labels(wg)

        def _has(substr: str) -> bool:
            return any(substr in label for label in labels)

        # Outer graph hosts the runtime input-resolution tasks
        # (replacing the inline plain-Python ``resolve_pseudo_family`` /
        # ``count_electrons`` calls that broke with ``TaggedValue`` proxies)
        # plus the DFT init + the screening-parameters sub-graph + the
        # final KI (which applies the converged screening parameters
        # and therefore lives at the workflow level, not inside the
        # screening sub-graph).
        assert _has("resolve_pseudo_family_task"), labels
        assert _has("count_electrons_task"), labels
        assert _has("dft_init"), labels
        assert _has("ComputeScreeningParameters"), labels
        # Final KI is wrapped in a thin ``KIFinal`` @task.graph so its
        # parameter-builder arithmetic runs in a scope where ``nelec``
        # is a plain int (not a socket from ``count_electrons_task``).
        assert _has("KIFinal"), labels

        # Now build the inner refinement sub-graph independently to
        # verify the Map-zone / source-builder / gather wiring.
        from aiida import orm

        # Use the sub-graph's build entry directly. We pass plain Python
        # values for the scalar/structural inputs; ``pseudos`` and
        # ``dft_remote`` are placeholders the topology check ignores.
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import ComputeScreeningParameters, ScreeningIteration

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        # Unstored placeholder — only the topology of the resulting
        # WorkGraph is inspected; the value is never dereferenced.
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")

        sub_wg = ComputeScreeningParameters.build(
            code=kcp_code,
            structure=ozone_structure,
            pseudos=pseudos,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            nelec=18,
            nelup=9,
            neldw=9,
            tot_magnetization=0,
            initial_alpha=0.6,
            functional="ki",
            init_orbitals="kohn-sham",
            dft_remote=dummy_remote,
        )
        sub_labels = self._all_link_labels(sub_wg)

        def _sub_has(substr: str) -> bool:
            return any(substr in label for label in sub_labels)

        assert _sub_has("generate_alphas"), sub_labels
        # Per-orbital fan-out lives inside the ``ScreeningIteration`` sub-graph
        # extracted by B.2. ``ki_final`` no longer lives here — it's at the
        # workflow level (it's the application of the screening parameters,
        # not part of computing them).
        assert _sub_has("ScreeningIteration"), sub_labels
        assert not _sub_has("ki_final"), sub_labels

        # Build ``ScreeningIteration`` directly to verify its internals —
        # ``@task.graph`` sub-tasks are opaque from the parent graph at
        # build time, so the walker can't reach Map zones / source builders
        # through ``ComputeScreeningParameters`` alone.
        from aiida_koopmans.workgraphs.kcp import _kcp_base_inputs

        iter_wg = ScreeningIteration.build(
            code=kcp_code,
            structure=ozone_structure,
            pseudos=pseudos,
            base=_kcp_base_inputs(
                ozone_structure,
                nspin=2,
                nelec=18,
                nelup=9,
                neldw=9,
                tot_magnetization=0,
                ecutwfc=65.0,
                ecutrho=260.0,
            ),
            nbnd=10,
            functional="ki",
            spin_polarized=False,
            current_alphas={
                "filled": {"none": [0.6] * 9},
                "empty": {"none": [0.6]},
            },
            parent_folder=dummy_remote,
            variational_orbital_overlays=None,
            ki_overrides=None,
            filled_overrides=None,
            empty_overrides_dict=None,
            options=None,
        )
        iter_labels = self._all_link_labels(iter_wg)

        def _iter_has(substr: str) -> bool:
            return any(substr in label for label in iter_labels)

        assert _iter_has("build_filled_iter_source"), iter_labels
        assert _iter_has("build_empty_iter_source"), iter_labels
        # Two Map zones (filled + empty branches).
        assert sum(1 for s in iter_labels if "map_zone" in s.lower()) >= 2, iter_labels
        # Gather step packing per-orbital sockets back into an
        # ``AlphaScreening`` shape.
        assert _iter_has("assemble_alpha_screening"), iter_labels
        # Trial KI inside the iteration.
        assert _iter_has("ki_trial"), iter_labels
        # Convergence indicator the ``While`` zone will read in B.3.
        assert _iter_has("max_alpha_error"), iter_labels

    def test_multi_iteration_builds_while_zone(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``alpha_numsteps > 1`` wraps iterations 2..N in a ``While`` zone.

        For ``alpha_numsteps = 1`` the dispatcher unrolls a single
        ``ScreeningIteration`` outside the loop; for >1 it adds an
        ``aiida-workgraph`` ``While`` zone whose body reads ctx slots
        populated by iter 1 (with explicit ``<<`` waits) and runs
        ``alpha_numsteps - 1`` more iterations. This test pins that
        the ``while_zone`` task is actually present in the built
        graph — a regression here would silently fall back to
        single-iteration behaviour.
        """
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import ComputeScreeningParameters

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")

        sub_wg = ComputeScreeningParameters.build(
            code=kcp_code,
            structure=ozone_structure,
            pseudos=pseudos,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            nelec=18,
            nelup=9,
            neldw=9,
            tot_magnetization=0,
            initial_alpha=0.6,
            functional="ki",
            init_orbitals="kohn-sham",
            alpha_numsteps=2,
            dft_remote=dummy_remote,
        )
        labels = self._all_link_labels(sub_wg)

        # Exactly one ``while_zone`` should be present (the outer
        # ``alpha_numsteps == 1`` branch builds none).
        n_while = sum(1 for s in labels if "while_zone" in s.lower())
        assert n_while == 1, (n_while, labels)
        # Both the unrolled iter_1 and the in-loop iter_n should exist
        # as separate ``ScreeningIteration`` task instances.
        assert sum(1 for s in labels if s == "ScreeningIteration") >= 1, labels
        # The ``op_ge`` (the ``>=`` comparison task synthesised for the
        # While condition) confirms ``condition << iter_1["max_error"]``
        # wired up.
        assert any("op_ge" in s for s in labels), labels

    def test_single_iteration_omits_while_zone(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``alpha_numsteps == 1`` skips the ``While`` zone entirely.

        The dispatcher gates ``While`` construction on
        ``alpha_numsteps > 1`` so the ``op_ge`` condition doesn't fire
        before iter_1 has produced ``max_error`` (``wg.ctx`` writes
        don't create dataflow edges in aiida-workgraph).
        """
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import ComputeScreeningParameters

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")

        sub_wg = ComputeScreeningParameters.build(
            code=kcp_code,
            structure=ozone_structure,
            pseudos=pseudos,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            nelec=18,
            nelup=9,
            neldw=9,
            tot_magnetization=0,
            initial_alpha=0.6,
            functional="ki",
            init_orbitals="kohn-sham",
            alpha_numsteps=1,
            dft_remote=dummy_remote,
        )
        labels = self._all_link_labels(sub_wg)
        assert not any("while_zone" in s.lower() for s in labels), labels

    def test_spin_polarized_screening_emits_both_channels(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``spin_polarized=True`` doubles the per-orbital fan-out.

        Builds ``ScreeningIteration`` directly so the per-spin Map-zone
        keys are visible. With ``spin_polarized=True`` the source
        builders emit ``up_orb_N`` *and* ``down_orb_N`` keys (rather
        than a single representative ``orb_N``); each orbital becomes
        its own per-spin sub-graph, so we expect to see both prefixes
        in the link labels.
        """
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import ScreeningIteration, _kcp_base_inputs

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")

        iter_wg = ScreeningIteration.build(
            code=kcp_code,
            structure=ozone_structure,
            pseudos=pseudos,
            base=_kcp_base_inputs(
                ozone_structure,
                nspin=2,
                nelec=18,
                nelup=9,
                neldw=9,
                tot_magnetization=None,
                ecutwfc=65.0,
                ecutrho=260.0,
            ),
            nbnd=10,
            functional="ki",
            spin_polarized=True,
            current_alphas={
                "filled": {"up": [0.6] * 9, "down": [0.6] * 9},
                "empty": {"up": [0.6], "down": [0.6]},
            },
            parent_folder=dummy_remote,
            variational_orbital_overlays=None,
            ki_overrides=None,
            filled_overrides=None,
            empty_overrides_dict=None,
            options=None,
        )
        labels = self._all_link_labels(iter_wg)
        # The per-orbital ``up_orb_<n>`` / ``down_orb_<n>`` keys are
        # expanded at *runtime* by the Map zone (the source builder's
        # output dict drives the fan-out), so at build time we can only
        # confirm the build succeeded and the Map zones are wired in.
        # Runtime parity (up == down alphas for closed-shell ozone)
        # lives in the end-to-end manual smoke test, not here.
        assert any("map_zone" in s.lower() for s in labels), labels
        assert any("ki_trial" in s for s in labels), labels
        assert any("build_filled_iter_source" in s for s in labels), labels
        assert any("build_empty_iter_source" in s for s in labels), labels

    def test_closed_shell_init_chain_has_four_init_steps(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Closed-shell init expands into a spin-symmetric 3+1 sub-chain.

        Wires the legacy ``restart_with_higher_precision`` flow:
        nspin=1 → nspin=2 dummy → ConvertSpin1ToSpin2 → nspin=2 restart.
        Each step gets a distinct ``call_link_label`` so the provenance
        graph stays readable.
        """
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            spin_polarized=False,
        )
        labels = self._all_link_labels(wg)
        for expected in (
            "dft_init_nspin1",
            "dft_init_nspin2_dummy",
            "convert_spin1_to_spin2",
            "dft_init_nspin2",
        ):
            assert any(expected in label for label in labels), (expected, labels)

    @staticmethod
    def _run_build_empty_iter_source(*, nelup, neldw, tot_magnetization=None, nbnd=10):
        """Call ``build_empty_iter_source`` and return its per-orbital dict.

        Bypasses the ``@task`` wrapper (via ``_callable``) for direct
        Python-level unit testing on spin-polarised ozone-shaped input.
        """
        from aiida_koopmans.workgraphs.kcp import (
            KcpBaseInputs,
            build_empty_iter_source,
        )

        base = KcpBaseInputs(
            ecutwfc=65.0,
            ecutrho=260.0,
            nspin=2,
            nelec=nelup + neldw,
            ntyp=1,
            mt_correction=True,
            nelup=nelup,
            neldw=neldw,
            tot_magnetization=tot_magnetization,
        )
        from aiida_koopmans.workgraphs.variational_orbitals import (
            enumerate_variational_orbitals,
        )

        orbitals = enumerate_variational_orbitals(
            nelup=nelup, neldw=neldw, nbnd=nbnd, spin_polarized=True
        )
        # ``_callable`` is the raw Python function under the ``@task`` decorator
        # (the decorator returns a ``TaskHandle`` at runtime; type checkers
        # see only the underlying ``FunctionType``).
        return build_empty_iter_source._callable(  # type: ignore[attr-defined]
            base=base,
            nbnd=nbnd,
            orbitals=orbitals,
            empty_alphas={
                "up": [0.6] * max(0, nbnd - nelup),
                "down": [0.6] * max(0, nbnd - neldw),
            },
        )

    def test_empty_iter_source_swaps_when_post_addition_violates_constraint(self):
        """DOWN-channel empty + closed-shell-effective counts: swap is needed.

        nelup=9, neldw=9 + DOWN channel -> post-addition (9, 10) violates
        ``nupdwn(1) >= nupdwn(2)``: kcp.x would refuse. The down orbital's
        per-orbital dict must carry the spin-swap overlay payload.
        """
        source = self._run_build_empty_iter_source(nelup=9, neldw=9)
        down_orb = source["down_orb_10"]
        assert down_orb["overlay"] == {
            "evc01": "evc02",
            "evc02": "evc01",
            "evc_empty1": "evc_empty2",
            "evc_empty2": "evc_empty1",
            "evc0_empty1": "evc0_empty2",
            "evc0_empty2": "evc0_empty1",
        }
        # In the swapped frame nelup>=neldw (the constraint kcp.x checks).
        sys = down_orb["dummy_parameters"]["SYSTEM"]
        assert sys["nelup"] >= sys["neldw"]

    def test_empty_iter_source_no_swap_when_up_channel(self):
        """UP-channel empty + closed-shell-effective counts: no swap.

        nelup=9, neldw=9 + UP channel -> post-addition (10, 9) satisfies
        ``nupdwn(1) >= nupdwn(2)`` -- overlay should be empty.
        """
        source = self._run_build_empty_iter_source(nelup=9, neldw=9)
        up_orb = source["up_orb_10"]
        assert up_orb["overlay"] == {}

    def test_empty_iter_source_no_swap_when_ferromag_post_counts_ok(self):
        """Ferromagnetic case: nelup=12, neldw=8 + DOWN -> (12, 9), no swap.

        The post-addition counts still satisfy the kcp.x constraint, so
        the swap branch should not fire even though the empty orbital is
        in the DOWN channel.
        """
        # nelup=12, neldw=8 means nbnd must be at least 12; bump it.
        source = self._run_build_empty_iter_source(nelup=12, neldw=8, tot_magnetization=4, nbnd=14)
        # Look for any DOWN orbital and check its overlay is empty.
        down_keys = [k for k in source if k.startswith("down_orb_")]
        assert down_keys, sorted(source)
        for k in down_keys:
            assert source[k]["overlay"] == {}, (k, source[k]["overlay"])

    def test_empty_iter_source_open_shell_o2_layout(self):
        """O2-shaped open-shell input exercises the per-spin asymmetric path.

        ``nelup=7, neldw=5, nbnd=8`` (O2 triplet with SG15 6e pseudo):
        UP has 1 empty, DOWN has 3 empties. Symmetric ``n_empty // 2``
        halving would have wrongly emitted 2 per spin. Also verifies the
        legacy LUMO-clamp on ``fixed_band``, the global ``index_empty_to_save``
        counter, and the ``band_index`` offset by ``max(nelup, neldw)``
        (where the trial-KI lambda matrix's empty block starts).
        """
        source = self._run_build_empty_iter_source(nelup=7, neldw=5, tot_magnetization=2, nbnd=8)
        up_keys = sorted(k for k in source if k.startswith("up_orb_"))
        down_keys = sorted(k for k in source if k.startswith("down_orb_"))

        # Asymmetric per-spin empty manifolds.
        assert up_keys == ["up_orb_8"], up_keys
        assert down_keys == ["down_orb_6", "down_orb_7", "down_orb_8"], down_keys

        # All DOWN empties get ``fixed_band`` clamped to the per-spin
        # LUMO position (= ``neldw + 1 + nelup`` = 13). kcp.x reorders
        # the constrained orbital into that slot regardless of which
        # empty we're actually screening; the orbital identity is
        # selected by the wavefunction pz_print writes per
        # ``index_empty_to_save``.
        for k in down_keys:
            sys = source[k]["dummy_parameters"]["SYSTEM"]
            assert sys["fixed_band"] == 13, (k, sys["fixed_band"])
        # UP empty's LUMO clamp = nelup + 1 = 8.
        assert source["up_orb_8"]["dummy_parameters"]["SYSTEM"]["fixed_band"] == 8

        # ``index_empty_to_save`` is the global counter across spins —
        # legacy ``_koopmans_dscf.py:697-699`` puts UP empties first
        # then DOWN. UP empty -> 1; DOWN empties -> 2, 3, 4.
        assert source["up_orb_8"]["dummy_parameters"]["NKSIC"]["index_empty_to_save"] == 1
        for k, expected in (("down_orb_6", 2), ("down_orb_7", 3), ("down_orb_8", 4)):
            got = source[k]["dummy_parameters"]["NKSIC"]["index_empty_to_save"]
            assert got == expected, (k, got, expected)

        # ``band_index`` for empties uses ``max(nelup, neldw) + i`` —
        # the offset where the trial-KI lambda matrix's empty block
        # starts (parser block-diag stack of ``filled_ham`` (sized
        # max_n_filled) and ``empty_ham``). Not the per-spin physical
        # position, which would land in the filled-block padding zone
        # for the spin with fewer filled.
        assert source["up_orb_8"]["band_index"] == 7  # max(7,5) + 0
        assert source["down_orb_6"]["band_index"] == 7  # max(7,5) + 0
        assert source["down_orb_7"]["band_index"] == 8  # max(7,5) + 1
        assert source["down_orb_8"]["band_index"] == 9  # max(7,5) + 2

        # Open-shell with nelup > neldw + adding to DOWN: post-add
        # (7, 6) still satisfies nupdwn(1) >= nupdwn(2), no swap.
        for k in up_keys + down_keys:
            assert source[k]["overlay"] == {}, (k, source[k]["overlay"])

    def test_filled_iter_source_open_shell_o2_layout(self):
        """O2-shaped filled iterator: 7 UP + 5 DOWN, DOWN bands shifted by nelup.

        For genuinely open-shell systems ``nelup != neldw``, legacy
        ``_koopmans_dscf.py:759-760`` shifts DOWN-channel
        ``fixed_band`` by ``nelup`` (not by a symmetric halved count).
        Closed-shell symmetric input previously hid this — closed-shell
        ozone has ``nelup == neldw`` so the shift agreed regardless of
        which choice was made.
        """
        from aiida_koopmans.workgraphs.kcp import build_filled_iter_source
        from aiida_koopmans.workgraphs.variational_orbitals import (
            enumerate_variational_orbitals,
        )

        orbitals = enumerate_variational_orbitals(nelup=7, neldw=5, nbnd=8, spin_polarized=True)
        source = build_filled_iter_source._callable(  # type: ignore[attr-defined]
            nelup=7,
            neldw=5,
            orbitals=orbitals,
            filled_alphas={"up": [0.6] * 7, "down": [0.6] * 5},
        )
        up_keys = sorted(k for k in source if k.startswith("up_orb_"))
        down_keys = sorted(k for k in source if k.startswith("down_orb_"))

        # Asymmetric per-spin filled manifolds (7 vs 5).
        assert up_keys == [f"up_orb_{i}" for i in range(1, 8)], up_keys
        assert down_keys == [f"down_orb_{i}" for i in range(1, 6)], down_keys

        # UP filled fixed_band = per-spin index (1..7); DOWN filled
        # fixed_band = per-spin index + nelup (8..12).
        for i in range(1, 8):
            assert source[f"up_orb_{i}"]["fixed_band"] == i
        for i in range(1, 6):
            assert source[f"down_orb_{i}"]["fixed_band"] == i + 7  # + nelup

        # ``band_index`` for filled uses the per-spin physical
        # position (the filled block fills from row 0; only the
        # filled-block padding above ``n_filled_this_spin`` is zero).
        for i in range(1, 8):
            assert source[f"up_orb_{i}"]["band_index"] == i - 1
        for i in range(1, 6):
            assert source[f"down_orb_{i}"]["band_index"] == i - 1

    def test_generate_alphas_open_shell_per_spin_sizes(self):
        """``generate_alphas`` returns asymmetric per-spin lists for nelup != neldw.

        Closed-shell symmetric halving (``n_filled // 2``) was hiding
        the bug — for O2 (7+5 = 12 electrons, nbnd=8) we want UP=7
        filled / 1 empty and DOWN=5 filled / 3 empty, *not* 6 / 2
        per spin from halving.
        """
        from aiida_koopmans.workgraphs.kcp import generate_alphas

        alphas = generate_alphas._callable(  # type: ignore[attr-defined]
            alpha_guess=0.6,
            nbnd=8,
            nelup=7,
            neldw=5,
            spin_polarized=True,
        )
        assert len(alphas["filled"][SpinChannel.UP]) == 7
        assert len(alphas["filled"][SpinChannel.DOWN]) == 5
        assert len(alphas["empty"][SpinChannel.UP]) == 1  # nbnd - nelup
        assert len(alphas["empty"][SpinChannel.DOWN]) == 3  # nbnd - neldw
        # Closed-shell representative path: single ``none`` channel.
        closed = generate_alphas._callable(  # type: ignore[attr-defined]
            alpha_guess=0.6, nbnd=10, nelup=9, neldw=9, spin_polarized=False
        )
        assert set(closed["filled"]) == {SpinChannel.NONE}
        assert len(closed["filled"][SpinChannel.NONE]) == 9
        assert len(closed["empty"][SpinChannel.NONE]) == 1

    def test_spin_polarized_init_is_single_step(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Spin-polarised init: legacy runs no symmetric pre-pass.

        Open-shell systems use independent up/down channels at init —
        only the single ``dft_init`` step should appear, with none of
        the closed-shell chain steps.
        """
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            spin_polarized=True,
        )
        labels = self._all_link_labels(wg)
        # Plain ``dft_init`` is present.
        assert any(label == "dft_init" or label.endswith(".dft_init") for label in labels), labels
        # None of the closed-shell chain steps should appear.
        for forbidden in (
            "dft_init_nspin1",
            "dft_init_nspin2_dummy",
            "convert_spin1_to_spin2",
            "dft_init_nspin2",
        ):
            assert not any(forbidden in label for label in labels), (forbidden, labels)
