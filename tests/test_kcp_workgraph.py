"""Unit tests for the kcp.x workgraph builders.

Covers the pure-function building blocks (parameter dicts, scope guards,
utility helpers). A full end-to-end WorkGraph construction test is deferred
to the Phase-5 regression harness, which will have a real SG15 pseudo
family available.
"""

from __future__ import annotations

import io
from dataclasses import replace
from typing import ClassVar

import pytest

from aiida_koopmans.functionals import Correction
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.utils.electrons import count_electrons, filled_and_empty_counts
from aiida_koopmans.variational_orbitals import VariationalOrbitalType
from aiida_koopmans.workgraphs.kcp import (
    KcpBaseInputs,
    _build_n_minus_1_parameters,
    _build_n_plus_1_parameters,
    _build_orbdep_parameters,
    _build_print_parameters,
    _ks_variational_overlay,
    _stage_wannier_seed,
    _validate_alpha_inputs,
    _validate_alpha_screening,
    _validate_scope,
    build_dft_parameters,
)

# ----------------------------------------------------------------------
# _validate_scope — every NotImplementedError path
# ----------------------------------------------------------------------


class TestValidateScope:
    def test_supported_baseline_passes(self, ozone_structure):
        _validate_scope(
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            fix_spin_contamination=False,
            structure=ozone_structure,
        )

    def test_supported_kipz_passes(self, ozone_structure):
        _validate_scope(
            correction=Correction.KIPZ,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            fix_spin_contamination=False,
            structure=ozone_structure,
        )

    @pytest.mark.parametrize("correction", [Correction.PKIPZ, Correction.NONE, Correction.ALL])
    def test_unsupported_correction_raises(self, ozone_structure, correction):
        with pytest.raises(NotImplementedError, match="correction="):
            _validate_scope(
                correction=correction,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                fix_spin_contamination=False,
                structure=ozone_structure,
            )

    def test_pz_init_raises(self, ozone_structure):
        with pytest.raises(NotImplementedError, match="init_orbitals="):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals="pz",
                fix_spin_contamination=False,
                structure=ozone_structure,
            )

    @pytest.mark.parametrize("init_orbitals", ["mlwfs", "projwfs"])
    def test_wannier_init_on_molecule_raises(self, ozone_structure, init_orbitals):
        with pytest.raises(ValueError, match="periodic structure"):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals=init_orbitals,
                fix_spin_contamination=False,
                structure=ozone_structure,
            )

    @pytest.mark.parametrize("init_orbitals", ["mlwfs", "projwfs"])
    def test_wannier_init_missing_inputs_raises(self, periodic_ozone_structure, init_orbitals):
        with pytest.raises(ValueError, match=r"\['blocks', 'kgrid'\]"):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals=init_orbitals,
                fix_spin_contamination=False,
                structure=periodic_ozone_structure,
                kpoints=object(),
                codes=object(),
            )

    def test_wannier_init_with_all_inputs_passes(self, periodic_ozone_structure):
        _validate_scope(
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            fix_spin_contamination=False,
            structure=periodic_ozone_structure,
            blocks=[object()],
            kgrid=[2, 2, 2],
            kpoints=object(),
            codes=object(),
        )

    def test_alpha_numsteps_no_longer_validated(self, ozone_structure):
        # ``alpha_numsteps`` is range-checked by the koopmans2 Pydantic
        # input model upstream; the scope guard no longer needs to look
        # at it. The recursive ``RefineScreeningParameters`` handles any
        # positive count.
        _validate_scope(
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            fix_spin_contamination=False,
            structure=ozone_structure,
        )

    def test_spin_contamination_raises(self, ozone_structure):
        with pytest.raises(NotImplementedError, match="fix_spin_contamination"):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                fix_spin_contamination=True,
                structure=ozone_structure,
            )

    def test_periodic_kohn_sham_raises(self, periodic_ozone_structure):
        with pytest.raises(NotImplementedError, match="periodic structure"):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                fix_spin_contamination=False,
                structure=periodic_ozone_structure,
            )


# ----------------------------------------------------------------------
# Per-orbital alpha injection — input-consistency and payload validators
# ----------------------------------------------------------------------


_OZONE_ALPHAS = {
    "filled": {"none": [0.6] * 9},
    "empty": {"none": [0.7]},
}


class TestValidateAlphaScreening:
    def test_closed_shell_payload_passes(self):
        _validate_alpha_screening(_OZONE_ALPHAS, nelup=9, neldw=9, nbnd=10)

    def test_spin_polarized_payload_passes(self):
        _validate_alpha_screening(
            {
                "filled": {"up": [0.6] * 7, "down": [0.6] * 5},
                "empty": {"up": [0.7], "down": [0.7] * 3},
            },
            nelup=7,
            neldw=5,
            nbnd=8,
        )

    def test_enum_keys_accepted(self):
        # ``AlphaScreening`` declares ``dict[SpinChannel, ...]``; enum keys
        # must validate the same as their post-round-trip string form.
        _validate_alpha_screening(
            {
                "filled": {SpinChannel.NONE: [0.6] * 9},
                "empty": {SpinChannel.NONE: [0.7]},
            },
            nelup=9,
            neldw=9,
            nbnd=10,
        )

    def test_unknown_channel_raises(self):
        with pytest.raises(ValueError, match="Unknown spin channel 'spinor'"):
            _validate_alpha_screening({"filled": {"spinor": [0.6]}, "empty": {"spinor": [0.7]}})

    def test_mixed_channels_raise(self):
        with pytest.raises(ValueError, match="mix"):
            _validate_alpha_screening({"filled": {"none": [0.6]}, "empty": {"up": [0.7]}})

    def test_non_numeric_entries_raise(self):
        with pytest.raises(ValueError, match="list of numbers"):
            _validate_alpha_screening({"filled": {"none": ["high"]}, "empty": {"none": []}})

    def test_wrong_filled_count_raises(self):
        with pytest.raises(ValueError, match=r"has 8 entries but .* 9 filled"):
            _validate_alpha_screening(
                {"filled": {"none": [0.6] * 8}, "empty": {"none": [0.7]}},
                nelup=9,
                neldw=9,
                nbnd=10,
            )

    def test_single_channel_open_shell_raises(self):
        with pytest.raises(ValueError, match="closed shell"):
            _validate_alpha_screening(
                {"filled": {"none": [0.6] * 7}, "empty": {"none": [0.7]}},
                nelup=7,
                neldw=5,
                nbnd=8,
            )

    def test_absent_channel_allowed_when_empty_manifold_is_empty(self):
        # nbnd == nelup: no empty orbitals, so the gather legitimately
        # produces no ``empty`` channel at all.
        _validate_alpha_screening(
            {"filled": {"none": [0.6] * 9}, "empty": {}}, nelup=9, neldw=9, nbnd=9
        )

    def test_counts_skipped_when_unknown(self):
        # Wrong lengths pass the structure-only check (counts unknown at
        # the outer build); they are caught downstream where the electron
        # counts are concrete.
        _validate_alpha_screening({"filled": {"none": [0.6] * 3}, "empty": {"none": []}})

    def test_none_channel_on_spin_polarized_run_raises(self):
        with pytest.raises(ValueError, match=r"spin_polarized=True.*'none'"):
            _validate_alpha_screening(_OZONE_ALPHAS, spin_polarized=True)

    def test_spin_channels_on_closed_shell_run_raise(self):
        with pytest.raises(ValueError, match="spin_polarized=False"):
            _validate_alpha_screening(
                {"filled": {"up": [0.6], "down": [0.6]}, "empty": {"up": [], "down": []}},
                spin_polarized=False,
            )

    def test_matching_spin_conventions_pass(self):
        _validate_alpha_screening(_OZONE_ALPHAS, spin_polarized=False)
        _validate_alpha_screening(
            {"filled": {"up": [0.6], "down": [0.6]}, "empty": {"up": [0.7], "down": [0.7]}},
            spin_polarized=True,
        )


class TestValidateAlphaInputs:
    def test_scalar_only_passes(self):
        _validate_alpha_inputs(initial_alpha=0.6, initial_alphas=None, calculate_alpha=True)

    def test_per_orbital_only_passes(self):
        _validate_alpha_inputs(
            initial_alpha=None, initial_alphas=_OZONE_ALPHAS, calculate_alpha=True
        )
        _validate_alpha_inputs(
            initial_alpha=None, initial_alphas=_OZONE_ALPHAS, calculate_alpha=False
        )

    def test_scalar_and_per_orbital_conflict(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _validate_alpha_inputs(
                initial_alpha=0.6, initial_alphas=_OZONE_ALPHAS, calculate_alpha=True
            )

    def test_skip_without_per_orbital_alphas_raises(self):
        with pytest.raises(ValueError, match="calculate_alpha=False"):
            _validate_alpha_inputs(initial_alpha=None, initial_alphas=None, calculate_alpha=False)
        with pytest.raises(ValueError, match="calculate_alpha=False"):
            _validate_alpha_inputs(initial_alpha=0.6, initial_alphas=None, calculate_alpha=False)


class TestVariationalSeedHelpers:
    def test_ks_overlay_nspin2(self):
        assert _ks_variational_overlay(2) == {
            "evc1": "evc01",
            "evc2": "evc02",
            "evc_empty1": "evc0_empty1",
            "evc_empty2": "evc0_empty2",
        }

    def test_ks_overlay_nspin1(self):
        assert _ks_variational_overlay(1) == {"evc1": "evc01", "evc_empty1": "evc0_empty1"}

    def test_stage_wannier_seed_requires_both_files(self):
        parameters = {"SYSTEM": {}}
        assert _stage_wannier_seed(parameters, None, None) is None
        assert _stage_wannier_seed(parameters, object(), None) is None
        assert "restart_from_wannier_pwscf" not in parameters["SYSTEM"]

    def test_stage_wannier_seed_switches_restart_flag(self):
        parameters = {"SYSTEM": {}}
        evc1, evc2 = object(), object()
        staged = _stage_wannier_seed(parameters, evc1, evc2)
        assert staged == {"evc_occupied1": evc1, "evc_occupied2": evc2}
        assert parameters["SYSTEM"]["restart_from_wannier_pwscf"] is True


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
        params = build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert set(params.keys()) == {"CONTROL", "SYSTEM", "ELECTRONS", "IONS", "EE"}
        assert "NKSIC" not in params
        # EE machinery always on; periodic systems compensate with 'none'.
        assert params["EE"]["which_compensation"] == "none"

    def test_dft_control_is_from_scratch(self):
        # ndr/ndw are owned by ``KcpCalculation._inject_owned_keys`` (universal
        # 50/60 across all kcp.x runs). The builder shouldn't set them.
        params = build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["CONTROL"]["restart_mode"] == "from_scratch"
        assert params["CONTROL"]["calculation"] == "cp"
        assert "ndr" not in params["CONTROL"]
        assert "ndw" not in params["CONTROL"]

    def test_dft_system_no_orbdep(self):
        params = build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["SYSTEM"]["do_orbdep"] is False
        assert params["SYSTEM"]["nelec"] == 18
        assert params["SYSTEM"]["nelup"] == 9
        assert params["SYSTEM"]["neldw"] == 9
        assert params["SYSTEM"]["nbnd"] == 10
        assert params["SYSTEM"]["ecutwfc"] == 65.0
        assert params["SYSTEM"]["ecutrho"] == 260.0

    def test_dft_outerloop_enabled(self):
        params = build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["ELECTRONS"]["do_outerloop"] is True
        assert params["ELECTRONS"]["do_outerloop_empty"] is True

    def test_conv_thr_scales_with_nelec(self):
        params = build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["ELECTRONS"]["conv_thr"] == pytest.approx(1.8e-8)

    def test_nspin_one_skips_spin_keys(self):
        base = replace(_OZONE_BASE, nspin=1, nelup=None, neldw=None)
        params = build_dft_parameters(base, nbnd=10)
        assert params["SYSTEM"]["nspin"] == 1
        assert "nelup" not in params["SYSTEM"]
        assert "neldw" not in params["SYSTEM"]

    def test_ion_radius_scales_with_ntyp(self):
        # ``ion_radius(i)`` must be emitted once per species — ozone has
        # ``ntyp=1`` so we get a single entry, not a hardcoded 1..4.
        params = build_dft_parameters(_OZONE_BASE, nbnd=10)
        assert params["IONS"]["ion_radius(1)"] == 1.0
        assert "ion_radius(2)" not in params["IONS"]
        # Three-species cell should emit three entries.
        params3 = build_dft_parameters(replace(_OZONE_BASE, ntyp=3), nbnd=10)
        assert params3["IONS"]["ion_radius(1)"] == 1.0
        assert params3["IONS"]["ion_radius(2)"] == 1.0
        assert params3["IONS"]["ion_radius(3)"] == 1.0
        assert "ion_radius(4)" not in params3["IONS"]


class TestBuildKiParameters:
    def test_has_nksic(self):
        params = _build_orbdep_parameters(_OZONE_BASE, nbnd=10, correction=Correction.KI)
        assert "NKSIC" in params
        assert params["NKSIC"]["which_orbdep"] == "nki"
        assert params["NKSIC"]["odd_nkscalfact"] is True
        assert params["NKSIC"]["odd_nkscalfact_empty"] is True
        assert params["NKSIC"]["do_bare_eigs"] is True

    def test_ki_control_is_restart(self):
        # See ``test_dft_control_is_from_scratch``: ndr/ndw live on the
        # CalcJob, not the builder.
        params = _build_orbdep_parameters(_OZONE_BASE, nbnd=10, correction=Correction.KI)
        assert params["CONTROL"]["restart_mode"] == "restart"
        assert "ndr" not in params["CONTROL"]
        assert "ndw" not in params["CONTROL"]

    def test_ki_enables_orbdep_and_disables_outerloop(self):
        params = _build_orbdep_parameters(_OZONE_BASE, nbnd=10, correction=Correction.KI)
        assert params["SYSTEM"]["do_orbdep"] is True
        assert params["ELECTRONS"]["do_outerloop"] is False
        assert params["ELECTRONS"]["do_outerloop_empty"] is False

    def test_periodic_uses_no_compensation(self):
        # The EE machinery is always on; periodic systems select
        # which_compensation='none' rather than dropping &EE.
        params = _build_orbdep_parameters(_OZONE_BASE, nbnd=10, correction=Correction.KI)
        assert params["EE"]["which_compensation"] == "none"
        assert params["SYSTEM"]["do_ee"] is True

    def test_aperiodic_emits_tcc(self):
        base = replace(_OZONE_BASE, mt_correction=True)
        params = _build_orbdep_parameters(base, nbnd=10, correction=Correction.KI)
        assert params["EE"]["which_compensation"] == "tcc"
        assert params["SYSTEM"]["do_ee"] is True

    def test_ki_disables_innerloop(self):
        # ``do_innerloop`` is True only for PZ; KI / KIPZ run no inner loop.
        params = _build_orbdep_parameters(_OZONE_BASE, nbnd=10, correction=Correction.KI)
        assert params["NKSIC"]["do_innerloop"] is False

    def test_pz_enables_innerloop(self):
        params = _build_orbdep_parameters(_OZONE_BASE, nbnd=10, correction=Correction.PZ)
        assert params["NKSIC"]["do_innerloop"] is True


class TestBuildKipzOrbdepParameters:
    """KIPZ-specific behaviour of the trial-step parameter builder.

    KIPZ piggy-backs on the same ODD machinery as KI but flips
    ``which_orbdep`` to ``nkipz`` and turns the inner CG loop on for
    the first molecular trial.
    """

    def test_kipz_which_orbdep_is_nkipz(self):
        params = _build_orbdep_parameters(_OZONE_BASE, nbnd=10, correction=Correction.KIPZ)
        assert params["NKSIC"]["which_orbdep"] == "nkipz"

    def test_kipz_first_iter_molecular_enables_innerloop(self):
        # Molecular = mt_correction=True; the fixture _OZONE_BASE is
        # mt_correction=False (periodic-ish), so make a molecular variant.
        base = replace(_OZONE_BASE, mt_correction=True)
        params = _build_orbdep_parameters(
            base, nbnd=10, correction=Correction.KIPZ, is_first_iteration=True
        )
        assert params["NKSIC"]["do_innerloop"] is True

    def test_kipz_later_iter_molecular_disables_innerloop(self):
        base = replace(_OZONE_BASE, mt_correction=True)
        params = _build_orbdep_parameters(
            base, nbnd=10, correction=Correction.KIPZ, is_first_iteration=False
        )
        assert params["NKSIC"]["do_innerloop"] is False

    def test_kipz_periodic_first_iter_disables_innerloop(self):
        # Periodic systems (mt_correction=False): no inner loop even on iter 1.
        params = _build_orbdep_parameters(
            _OZONE_BASE, nbnd=10, correction=Correction.KIPZ, is_first_iteration=True
        )
        assert params["NKSIC"]["do_innerloop"] is False


class TestBuildNMinus1Parameters:
    """``_build_n_minus_1_parameters`` — filled-orbital alpha step."""

    def test_ki_is_plain_dft(self):
        params = _build_n_minus_1_parameters(_OZONE_BASE, fixed_band=5, correction=Correction.KI)
        # KI runs plain DFT here — no orbital-dependent screening.
        assert params["SYSTEM"]["do_orbdep"] is False
        assert "which_orbdep" not in params["NKSIC"]

    def test_kipz_enables_orbdep_with_nkipz(self):
        params = _build_n_minus_1_parameters(_OZONE_BASE, fixed_band=5, correction=Correction.KIPZ)
        # KIPZ's n-1 step is alpha-dependent: do_orbdep=True + which_orbdep='nkipz'.
        # See the tripwire comment in KoopmansDSCFWorkflow.
        assert params["SYSTEM"]["do_orbdep"] is True
        assert params["NKSIC"]["which_orbdep"] == "nkipz"
        assert params["NKSIC"]["do_bare_eigs"] is True

    def test_shared_keys_for_both_corrections(self):
        # The non-functional-specific bits of n-1 (fixed_band, fixed_state,
        # restart_mode, conv_thr loosening) should be identical.
        ki = _build_n_minus_1_parameters(_OZONE_BASE, fixed_band=5, correction=Correction.KI)
        kipz = _build_n_minus_1_parameters(_OZONE_BASE, fixed_band=5, correction=Correction.KIPZ)
        for ns in ("CONTROL", "SYSTEM", "ELECTRONS"):
            for key in {"restart_mode", "fixed_band", "fixed_state", "f_cutoff"} & set(
                ki[ns].keys()
            ):
                assert ki[ns][key] == kipz[ns][key], (ns, key)


class TestBuildNPlus1Parameters:
    """``_build_n_plus_1_parameters`` — empty-orbital alpha step."""

    def test_ki_is_plain_dft(self):
        params = _build_n_plus_1_parameters(_OZONE_BASE, fixed_band=6, correction=Correction.KI)
        assert params["SYSTEM"]["do_orbdep"] is False
        assert "which_orbdep" not in params["NKSIC"]

    def test_kipz_enables_orbdep_with_nkipz(self):
        params = _build_n_plus_1_parameters(_OZONE_BASE, fixed_band=6, correction=Correction.KIPZ)
        assert params["SYSTEM"]["do_orbdep"] is True
        assert params["NKSIC"]["which_orbdep"] == "nkipz"


class TestBuildPrintParameters:
    """``_build_print_parameters`` — print step writes evcfixed_empty.dat."""

    def test_ki_uses_pz_orbdep(self):
        # KI's print step uses PZ-flavour orbdep (the ``pz_print`` step).
        params = _build_print_parameters(
            _OZONE_BASE, nbnd=10, fixed_band=6, correction=Correction.KI
        )
        assert params["NKSIC"]["which_orbdep"] == "pz"
        assert params["NKSIC"]["print_wfc_anion"] is True

    def test_kipz_uses_nkipz_orbdep(self):
        params = _build_print_parameters(
            _OZONE_BASE, nbnd=10, fixed_band=6, correction=Correction.KIPZ
        )
        assert params["NKSIC"]["which_orbdep"] == "nkipz"
        assert params["NKSIC"]["print_wfc_anion"] is True

    def test_print_step_disables_innerloop(self):
        # The print step operates on already-converged orbitals and must
        # not re-run the inner CG cycle.
        ki = _build_print_parameters(_OZONE_BASE, nbnd=10, fixed_band=6, correction=Correction.KI)
        kipz = _build_print_parameters(
            _OZONE_BASE, nbnd=10, fixed_band=6, correction=Correction.KIPZ
        )
        assert ki["NKSIC"]["do_innerloop"] is False
        assert kipz["NKSIC"]["do_innerloop"] is False


# ----------------------------------------------------------------------
# Spin-channel swap helpers
# ----------------------------------------------------------------------


class TestSwapKcpFrame:
    """Pure-function checks on ``_swap_kcp_frame``.

    The swap: swap nelup<->neldw, negate tot_magnetization (if set), and shift
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

    Reference formula:
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
        # nspin=2 here so ``SpinChannel.UP.axis == 0`` selects the up channel.
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


def _linear_sh_model(
    occ_and_emp_together: bool = True,
    correction: str = "ki",
    init_orbitals: str = "kohn-sham",
) -> dict:
    """Fit an exactly-linear self-Hartree model (``alpha = 0.4 - 0.1 * sh``)."""
    from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

    return ml_helpers.fit_screening_model(
        {
            "descriptors": [[-1.0], [-2.0], [-3.0], [-4.0]],
            "alpha_targets": [0.5, 0.6, 0.7, 0.8],
            "filled": [True, True, False, False],
            "labels": ["orb_1", "orb_2", "orb_3", "orb_4"],
        },
        "linear_regression",
        occ_and_emp_together=occ_and_emp_together,
        correction=correction,
        init_orbitals=init_orbitals,
    )


def _split_slope_sh_model() -> dict:
    """Fit occ/emp submodels with DIFFERENT slopes.

    occ: ``alpha = 0.4 - 0.1 * sh`` (0.55 at sh=-1.5); emp:
    ``alpha = -0.7 - 0.8 * sh`` (0.50 at sh=-1.5). At sh=-1.5 the two
    submodels disagree, so a filled-to-emp routing swap changes the
    result — the same-slope halves of ``_linear_sh_model`` cannot see
    that swap.
    """
    from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

    return ml_helpers.fit_screening_model(
        {
            "descriptors": [[-1.0], [-2.0], [-1.0], [-2.0]],
            "alpha_targets": [0.5, 0.6, 0.1, 0.9],
            "filled": [True, True, False, False],
            "labels": ["orb_1", "orb_2", "orb_3", "orb_4"],
        },
        "linear_regression",
        occ_and_emp_together=False,
        correction="ki",
        init_orbitals="kohn-sham",
    )


class TestKoopmansDSCFGraphBuild:
    """Inspect the task graph wired by ``KoopmansDSCFWorkflow.build`` for ozone.

    Doesn't run anything — verifies fan-out counts so a wiring regression
    surfaces without needing a real kcp.x install.
    """

    def _build_wg(
        self, *, ozone_structure, kcp_code, ozone_pseudo_family, spin_polarized=False, **overrides
    ):
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        inputs = {
            "code": kcp_code,
            "structure": ozone_structure,
            "pseudo_family": ozone_pseudo_family,
            "ecutwfc": 65.0,
            "ecutrho": 260.0,
            "nbnd": 10,
            "nspin": 2,
            "tot_magnetization": None,
            "correction": Correction.KI,
            "init_orbitals": VariationalOrbitalType.KOHN_SHAM,
            "alpha_numsteps": 1,
            "fix_spin_contamination": False,
            "initial_alpha": 0.6,
            "spin_polarized": spin_polarized,
        }
        inputs.update(overrides)
        return KoopmansDSCFWorkflow.build(**inputs)

    @staticmethod
    def _find_task(wg, name: str):
        """Return the (possibly nested) task with the given name."""

        def _walk(tasks):
            for t in tasks:
                if t.name == name:
                    return t
                found = _walk(getattr(t, "children", None) or [])
                if found is not None:
                    return found
            return None

        found = _walk(wg.tasks)
        assert found is not None, [t.name for t in wg.tasks]
        return found

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
        # Final KI is wrapped in a thin ``RunFinalKI`` @task.graph so its
        # parameter-builder arithmetic runs in a scope where ``nelec``
        # is a plain int (not a socket from ``count_electrons_task``).
        assert _has("RunFinalKI"), labels

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
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            dft_remote=dummy_remote,
        )
        sub_labels = self._all_link_labels(sub_wg)

        def _sub_has(substr: str) -> bool:
            return any(substr in label for label in sub_labels)

        assert _sub_has("generate_alphas"), sub_labels
        # Per-orbital fan-out lives inside the ``ScreeningIteration`` sub-graph.
        # ``ki_final`` no longer lives here — it's at the workflow level (it's
        # the application of the screening parameters, not part of computing
        # them).
        assert _sub_has("ScreeningIteration"), sub_labels
        assert not _sub_has("ki_final"), sub_labels

        # Build ``ScreeningIteration`` directly to verify its internals —
        # ``@task.graph`` sub-tasks are opaque from the parent graph at
        # build time, so the walker can't reach the fan-out sub-graph
        # through ``ComputeScreeningParameters`` alone.
        from aiida_koopmans.workgraphs.kcp import kcp_base_inputs

        iter_wg = ScreeningIteration.build(
            code=kcp_code,
            structure=ozone_structure,
            pseudos=pseudos,
            base=kcp_base_inputs(
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
            correction=Correction.KI,
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

        # Trial KI inside the iteration.
        assert _iter_has("ki_trial"), iter_labels
        # Grouping decision feeding the fan-out.
        assert _iter_has("assign_orbital_groups"), iter_labels
        # The per-orbital fan-out is a nested ``@task.graph`` (its body
        # is deferred until ``assign_orbital_groups``' output resolves,
        # so the scatter itself is invisible at build time — see
        # ``test_compute_orbital_screening_parameters_fanout_counts`` for the expanded
        # shape).
        assert _iter_has("compute_orbital_screening_parameters"), iter_labels
        # No Map zones remain.
        assert not any("map_zone" in s.lower() for s in iter_labels), iter_labels
        # Convergence indicator the recursive ``RefineScreeningParameters`` reads.
        assert _iter_has("max_alpha_error"), iter_labels

    def _build_per_orbital_wg(
        self,
        *,
        ozone_structure,
        kcp_code,
        ozone_pseudo_family,
        spin_polarized,
    ):
        """Build ``ComputeOrbitalScreeningParameters`` with concrete orbitals.

        With concrete inputs the deferred body executes at build time,
        so the native for-loop fan-out is fully visible in the resulting
        WorkGraph — one ``compute_alpha_<map_key>`` sub-graph per orbital.
        """
        import numpy as np
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import (
            ComputeOrbitalScreeningParameters,
            kcp_base_inputs,
        )
        from aiida_koopmans.workgraphs.variational_orbitals import (
            enumerate_variational_orbitals,
        )

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")

        orbitals = enumerate_variational_orbitals(
            nelup=9, neldw=9, nbnd=10, spin_polarized=spin_polarized
        )
        if spin_polarized:
            current_alphas = {
                "filled": {"up": [0.6] * 9, "down": [0.6] * 9},
                "empty": {"up": [0.6], "down": [0.6]},
            }
        else:
            current_alphas = {
                "filled": {"none": [0.6] * 9},
                "empty": {"none": [0.6]},
            }

        return ComputeOrbitalScreeningParameters.build(
            code=kcp_code,
            structure=ozone_structure,
            pseudos=pseudos,
            base=kcp_base_inputs(
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
            correction=Correction.KI,
            orbitals=orbitals,
            current_alphas=current_alphas,
            trial_remote=dummy_remote,
            trial_output_parameters={"energy": -100.0},
            trial_lambdas=np.zeros((2, 10, 10)),
            trial_bare_lambdas=np.zeros((2, 10, 10)),
        )

    def test_compute_orbital_screening_parameters_fanout_counts(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Closed-shell ozone: 9 filled + 1 empty screening sub-graphs.

        The for-loop fan-out is build-visible when ``orbitals`` is
        concrete — a wiring regression (wrong count, wrong labels)
        surfaces here without running anything.
        """
        wg = self._build_per_orbital_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            spin_polarized=False,
        )
        labels = self._all_link_labels(wg)
        compute_alpha_labels = {s for s in labels if s.startswith("compute_alpha_")}
        assert compute_alpha_labels == {f"compute_alpha_orb_{i}" for i in range(1, 11)}, sorted(
            compute_alpha_labels
        )
        # Gather steps packing per-orbital sockets back into an
        # ``AlphaScreening`` shape.
        assert any("expand_alphas_by_group" in s for s in labels), labels
        assert any("assemble_alpha_screening" in s for s in labels), labels
        # No Map zones remain.
        assert not any("map_zone" in s.lower() for s in labels), labels

    def test_multi_iteration_builds_refinement_loop(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``alpha_numsteps > 1`` chains a recursive ``RefineScreeningParameters``.

        For ``alpha_numsteps = 1`` the dispatcher unrolls a single
        ``ScreeningIteration`` and skips the loop entirely; for >1 it
        adds one ``refine_screening_parameters`` graph task consuming
        iter 1's outputs (each recursion level decides
        converged-vs-continue on the *previous* iteration's
        ``max_error`` inside its deferred body). This test pins that
        the loop task is actually present in the built graph — a
        regression here would silently fall back to single-iteration
        behaviour.
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
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            alpha_numsteps=2,
            dft_remote=dummy_remote,
        )
        labels = self._all_link_labels(sub_wg)

        # Exactly one refinement-loop task should be present (further
        # recursion levels are created at runtime, one per iteration).
        n_loop = sum(1 for s in labels if "refine_screening_parameters" in s)
        assert n_loop == 1, (n_loop, labels)
        # The unrolled iter_1 exists as a ``ScreeningIteration`` task.
        assert sum(1 for s in labels if s == "ScreeningIteration") >= 1, labels
        # No While zone / synthesised comparison task remains.
        assert not any("while_zone" in s.lower() for s in labels), labels
        assert not any("op_ge" in s for s in labels), labels

    def test_single_iteration_omits_refinement_loop(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``alpha_numsteps == 1`` skips the refinement loop entirely.

        The dispatcher gates ``RefineScreeningParameters`` construction on
        ``alpha_numsteps > 1`` so the single-iteration graph carries no
        superfluous recursion node.
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
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            alpha_numsteps=1,
            dft_remote=dummy_remote,
        )
        labels = self._all_link_labels(sub_wg)
        assert not any("refine_screening_parameters" in s for s in labels), labels
        assert not any("while_zone" in s.lower() for s in labels), labels

    def test_spin_polarized_screening_emits_both_channels(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``spin_polarized=True`` doubles the per-orbital fan-out.

        Builds ``ComputeOrbitalScreeningParameters`` directly with concrete
        spin-polarised orbitals: the for-loop fan-out emits
        ``compute_alpha_up_orb_N`` *and* ``compute_alpha_down_orb_N`` sub-graphs
        (rather than a single representative ``compute_alpha_orb_N`` per
        orbital), all visible at build time.
        """
        wg = self._build_per_orbital_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            spin_polarized=True,
        )
        labels = self._all_link_labels(wg)
        compute_alpha_labels = {s for s in labels if s.startswith("compute_alpha_")}
        expected = {f"compute_alpha_up_orb_{i}" for i in range(1, 11)} | {
            f"compute_alpha_down_orb_{i}" for i in range(1, 11)
        }
        assert compute_alpha_labels == expected, sorted(compute_alpha_labels)

    def test_closed_shell_init_chain_has_four_init_steps(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Closed-shell init expands into a spin-symmetric 3+1 sub-chain.

        Wires the spin-symmetric init flow:
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

        A plain function since the for-loop fan-out refactor (it runs
        inline inside ``ComputeOrbitalScreeningParameters``'s deferred body), so it is
        unit-testable directly on spin-polarised ozone-shaped input.
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
        return build_empty_iter_source(
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
        LUMO-clamp on ``fixed_band``, the global ``index_empty_to_save``
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
        # UP empties come first, then DOWN. UP empty -> 1; DOWN empties -> 2, 3, 4.
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

        For genuinely open-shell systems ``nelup != neldw``, the DOWN-channel
        ``fixed_band`` is shifted by ``nelup`` (not by a symmetric halved
        count). Closed-shell ozone has ``nelup == neldw`` so the shift agrees
        regardless of which choice is made — only open-shell exercises it.
        """
        from aiida_koopmans.workgraphs.kcp import build_filled_iter_source
        from aiida_koopmans.workgraphs.variational_orbitals import (
            enumerate_variational_orbitals,
        )

        orbitals = enumerate_variational_orbitals(nelup=7, neldw=5, nbnd=8, spin_polarized=True)
        source = build_filled_iter_source(
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

    def test_injected_alphas_skip_screening_loop(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``calculate_alpha=False`` + per-orbital alphas: no screening at all.

        The graph must contain no ``ComputeScreeningParameters`` (hence no
        trial KI and no Delta-SCF fan-out); the final KI is still present
        and the ``alphas`` output is fed by the ``injected_alphas`` echo
        task (a graph cannot echo a raw input as an output).
        """
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            initial_alpha=None,
            initial_alphas=_OZONE_ALPHAS,
            calculate_alpha=False,
        )
        labels = self._all_link_labels(wg)
        assert not any("ComputeScreeningParameters" in s for s in labels), labels
        assert any("RunFinalKI" in s for s in labels), labels
        assert any("injected_alphas" in s for s in labels), labels
        # The DFT init chain still runs (the final KI parents on it).
        assert any("dft_init" in s for s in labels), labels

        # The final KI must take over the trial KI's seeding role: on
        # the molecular route that means the KS-as-variational overlay
        # and the first-orbital-dependent-run marker. A skip branch that
        # silently drops either would still build — pin the socket
        # values.
        rfk = self._find_task(wg, "RunFinalKI")
        assert rfk.inputs.variational_orbital_overlays.value == _ks_variational_overlay(2)
        assert rfk.inputs.is_first_iteration.value is True
        assert rfk.inputs.initial_evc_occupied1.value is None

    def test_default_build_passes_the_pre_run_input_check(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Omitting the optional per-orbital alphas leaves no missing inputs.

        ``check_before_run`` runs on every real submission (both
        ``WorkGraph.run`` and ``WorkGraph.submit`` call it), so an input
        namespace it considers incomplete blocks the whole workflow —
        yet every other test here stops at ``.build()`` and cannot see
        it. This is the only guard against that class of defect.

        The optional per-orbital payload is a namespace socket whose own
        keys are required, so this also pins the dependency's contract:
        a namespace left entirely unfilled must not report its children
        as missing (``WorkGraph.find_missing_inputs``).
        """
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
        )
        missing = list(wg.find_missing_inputs(wg.inputs))
        for t in wg.tasks:
            missing.extend(wg.find_missing_inputs(t.inputs))
        assert not missing, sorted(set(missing))
        wg.check_before_run()

    def test_spin_convention_mismatch_raises_at_build(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """A closed-shell payload on a spin-polarized build fails with a named error.

        Without the convention check the mismatch would only surface as a
        bare ``KeyError`` inside the per-orbital fan-out at runtime.
        """
        with pytest.raises(ValueError, match="spin_polarized=True"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                spin_polarized=True,
                initial_alpha=None,
                initial_alphas=_OZONE_ALPHAS,
            )

    def test_injected_alphas_seed_refinement_loop(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``calculate_alpha=True`` + per-orbital alphas: loop runs, seeded.

        The screening sub-graph is still built, but its uniform-alpha
        generator must be absent — the caller's per-orbital payload feeds
        the first iteration directly.
        """
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            initial_alpha=None,
            initial_alphas=_OZONE_ALPHAS,
            calculate_alpha=True,
        )
        labels = self._all_link_labels(wg)
        assert any("ComputeScreeningParameters" in s for s in labels), labels
        assert not any("injected_alphas" in s for s in labels), labels

        # The workflow must actually hand the caller's payload to the
        # screening sub-graph — an ``initial_alphas=None`` regression at
        # this call site would silently fall back to the uniform guess.
        csp = self._find_task(wg, "ComputeScreeningParameters")
        assert csp.inputs.initial_alphas.filled.value == _OZONE_ALPHAS["filled"]
        assert csp.inputs.initial_alphas.empty.value == _OZONE_ALPHAS["empty"]

        # The generator suppression is only visible when the sub-graph's
        # deferred body runs — build it directly with concrete inputs.
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
            initial_alphas=_OZONE_ALPHAS,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            dft_remote=dummy_remote,
        )
        sub_labels = self._all_link_labels(sub_wg)
        assert not any("generate_alphas" in s for s in sub_labels), sub_labels
        assert any("ScreeningIteration" in s for s in sub_labels), sub_labels

    def test_scalar_and_per_orbital_alphas_conflict_at_build(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        with pytest.raises(ValueError, match="mutually exclusive"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                initial_alpha=0.6,
                initial_alphas=_OZONE_ALPHAS,
            )

    def test_skip_screening_without_alphas_raises_at_build(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        with pytest.raises(ValueError, match="calculate_alpha=False"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                calculate_alpha=False,
            )

    def test_run_final_ki_rejects_count_mismatch(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Injected alphas that don't match the manifolds fail loudly.

        ``RunFinalKI``'s body re-validates the payload against the
        concrete electron counts — a wrong-length channel must raise a
        named mismatch instead of writing a corrupt ``file_alpharef``.
        """
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import RunFinalKI

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")

        with pytest.raises(ValueError, match=r"has 8 entries but .* 9 filled"):
            RunFinalKI.build(
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
                correction=Correction.KI,
                alphas={"filled": {"none": [0.6] * 8}, "empty": {"none": [0.7]}},
                parent_folder=dummy_remote,
            )

    def test_spin_polarized_init_is_single_step(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Spin-polarised init: no symmetric pre-pass.

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

    def test_ml_model_routes_to_prediction(self, ozone_structure, kcp_code, ozone_pseudo_family):
        """An ml_model replaces the refinement sub-graph with the predict one."""
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_model=_linear_sh_model(),
        )
        labels = self._all_link_labels(wg)
        assert any("PredictScreeningParameters" in label for label in labels), labels
        assert not any("ComputeScreeningParameters" in label for label in labels), labels
        # The final KI still applies the (now predicted) screening parameters.
        assert any("RunFinalKI" in label for label in labels), labels

    def test_predict_subgraph_is_trial_plus_prediction(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """The predict sub-graph runs one trial KI and no Delta-SCF fan-out."""
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import PredictScreeningParameters

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")

        sub_wg = PredictScreeningParameters.build(
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
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            dft_remote=dummy_remote,
            ml_model=_linear_sh_model(),
        )
        sub_labels = self._all_link_labels(sub_wg)

        def _sub_has(substr: str) -> bool:
            return any(substr in label for label in sub_labels)

        assert _sub_has("generate_alphas"), sub_labels
        assert _sub_has("ki_trial"), sub_labels
        assert _sub_has("assign_orbital_groups"), sub_labels
        assert _sub_has("predict_alphas"), sub_labels
        # No Delta-SCF machinery anywhere in the predict route.
        for forbidden in (
            "compute_orbital_screening_parameters",
            "refine_screening_parameters",
            "ScreeningIteration",
            "dft_n_minus_1",
            "pz_print",
            "dft_n_plus_1",
        ):
            assert not _sub_has(forbidden), (forbidden, sub_labels)

    def test_ml_model_requires_calculate_alpha(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        with pytest.raises(ValueError, match="drop ml_model"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_model=_linear_sh_model(),
                calculate_alpha=False,
                initial_alpha=None,
                initial_alphas={
                    "filled": {"none": [0.6] * 9},
                    "empty": {"none": [0.6]},
                },
            )

    def test_ml_model_rejects_alpha_numsteps(self, ozone_structure, kcp_code, ozone_pseudo_family):
        with pytest.raises(ValueError, match="alpha_numsteps cannot take effect"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_model=_linear_sh_model(),
                alpha_numsteps=2,
            )


@pytest.mark.parametrize(
    ("ml_model", "ml_test", "expected"),
    [
        (None, False, False),
        ({"kind": "linear"}, False, True),
        ({"kind": "linear"}, True, False),
    ],
)
def test_model_replaces_refinement(ml_model, ml_test, expected):
    """Only a model supplied without ``ml_test`` stands in for the refinement.

    Distinguishes the two routes a model can take: replacing the Delta-SCF
    refinement, or running beside it as the thing being measured.
    """
    from aiida_koopmans.workgraphs.kcp import _model_replaces_refinement

    assert _model_replaces_refinement(ml_model=ml_model, ml_test=ml_test) is expected


# ----------------------------------------------------------------------
# Skip-mode final KI vs first-iteration trial KI: concrete kcp.x inputs
# ----------------------------------------------------------------------


def _link_source(socket):
    """Return ``(task_name, scoped_name)`` pairs feeding ``socket``, or ``None``."""
    links = getattr(socket, "_links", None) or []
    return sorted((link.from_socket._task.name, link.from_socket._scoped_name) for link in links)


def _plain(value):
    """Reduce a socket value to something comparable across two graph builds.

    Overlay / staging payloads arrive as freshly created ``orm.Dict``
    nodes, so identity and uuid differ between builds even when the
    content is identical; unwrap them. ``TaggedValue`` proxies compare
    by their wrapped value already.
    """
    from aiida import orm

    if isinstance(value, orm.Dict):
        return value.get_dict()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _socket_value(socket):
    """Return a socket's value, descending into input namespaces."""
    from aiida_workgraph.socket import TaskSocketNamespace

    if isinstance(socket, TaskSocketNamespace):
        return {s._name: _socket_value(s) for s in socket}
    return socket.value


class TestSkipModeFinalKIMatchesFirstTrialKI:
    """The skip-mode final KI must be the first-iteration trial KI in disguise.

    With ``calculate_alpha=False`` the refinement loop is gone, so the
    final KI is the first orbital-dependent run after the DFT init and
    has to inherit *every* seeding decision the trial KI would have
    made: the same parent save, the same variational-orbital seeding
    (the KS-as-variational overlay on the molecular route, the folded
    Wannier ``evc_occupied{1,2}`` staging on the periodic one) and the
    same ``is_first_iteration`` marker that drives KIPZ's inner-loop CG
    pass and the restart keys.

    Graph-shape tests cannot see any of this: dropping the overlay or
    the staging still produces a graph with the right tasks and links.
    These tests therefore compare the *concrete* kcp.x inputs — the
    namelist dicts key by key, the parent folder identity and the
    wavefunction staging — between the two routes' first trial and the
    skip-mode final run.
    """

    # A self-consistent set of concrete counts shared by both sub-graph
    # builds, so any difference in the resulting namelists comes from
    # the seeding decisions under test rather than from the inputs.
    NBND = 10
    NELEC = 18
    NELUP = NELDW = 9
    ALPHAS: ClassVar[dict] = {"filled": {"none": [0.6] * 9}, "empty": {"none": [0.7]}}

    @staticmethod
    def _pseudos(structure, pseudo_family):
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": pseudo_family})
            .one()[0]
        )
        return family.get_pseudos(structure=structure)

    def _numeric_args(self, structure, code, pseudos):
        return {
            "code": code,
            "structure": structure,
            "pseudos": pseudos,
            "ecutwfc": 65.0,
            "ecutrho": 260.0,
            "nbnd": self.NBND,
            "nspin": 2,
            "nelec": self.NELEC,
            "nelup": self.NELUP,
            "neldw": self.NELDW,
            "tot_magnetization": 0,
        }

    @staticmethod
    def _kcp_step(wg, prefix):
        """Return the single ``KcpStep`` task whose name starts with ``prefix``."""
        matches = [t for t in wg.tasks if t.name.startswith(prefix)]
        assert len(matches) == 1, (prefix, [t.name for t in wg.tasks])
        return matches[0]

    def _concrete_kcp_inputs(self, task):
        """Return the comparable subset of a ``KcpStep``'s concrete inputs."""
        return {
            "parameters": _plain(task.inputs.parameters.value),
            "overlays": _plain(_socket_value(task.inputs.variational_orbital_overlays)),
            "read_wavefunctions": _plain(_socket_value(task.inputs.read_wavefunctions)),
            "parent_folder": task.inputs.parent_folder.value.uuid,
            "alphas": _plain(_socket_value(task.inputs.alphas)),
        }

    def _build_route(self, request, route, correction, *, calculate_alpha):
        """Build a full DSCF workgraph for one route in one alpha mode."""
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        common = {
            "code": request.getfixturevalue("kcp_code"),
            "pseudo_family": request.getfixturevalue("ozone_pseudo_family"),
            "ecutwfc": 65.0,
            "ecutrho": 260.0,
            "nbnd": self.NBND,
            "nspin": 2,
            "correction": correction,
        }
        if route == "molecular":
            common |= {
                "structure": request.getfixturevalue("ozone_structure"),
                "init_orbitals": VariationalOrbitalType.KOHN_SHAM,
            }
        else:
            from tests.fixtures import ozone_projection_blocks

            common |= {
                "structure": request.getfixturevalue("periodic_ozone_structure"),
                "init_orbitals": VariationalOrbitalType.MLWFS,
                "codes": request.getfixturevalue("mlwf_codes"),
                "blocks": ozone_projection_blocks(),
                "kgrid": [2, 1, 1],
                "kpoints": request.getfixturevalue("kmesh"),
            }
        if calculate_alpha:
            return KoopmansDSCFWorkflow.build(initial_alpha=0.6, **common)
        return KoopmansDSCFWorkflow.build(
            initial_alpha=None,
            initial_alphas=self.ALPHAS,
            calculate_alpha=False,
            **common,
        )

    @pytest.mark.parametrize("correction", [Correction.KI, Correction.KIPZ])
    @pytest.mark.parametrize("route", ["molecular", "periodic"])
    def test_skip_mode_final_ki_inherits_the_trial_seeding(self, request, route, correction):
        """The skip-mode final KI's kcp.x inputs equal the first trial KI's.

        Discriminates three wiring regressions that leave the graph
        shape untouched:

        * dropping the KS-as-variational overlay on the skip path
          (``final_overlay = None``) — the overlay comparison fails;
        * dropping the Wannier ``evc_occupied`` staging and the
          ``first_orbdep_run`` marker — the staging comparison and the
          ``is_first_iteration``-sensitive namelist keys
          (``restart_mode``, KIPZ's ``do_innerloop_cg``) diverge;
        * seeding the refinement loop with ``initial_alphas=None`` —
          checked at the workflow's call site below.
        """
        from aiida import orm

        from aiida_koopmans.workgraphs.kcp import (
            ComputeScreeningParameters,
            RunFinalKI,
            ScreeningIteration,
        )

        calc_wg = self._build_route(request, route, correction, calculate_alpha=True)
        skip_wg = self._build_route(request, route, correction, calculate_alpha=False)

        csp = TestKoopmansDSCFGraphBuild._find_task(calc_wg, "ComputeScreeningParameters")
        rfk = TestKoopmansDSCFGraphBuild._find_task(skip_wg, "RunFinalKI")

        # (1) Provenance: the final KI must parent on — and stage from —
        # exactly the upstream sockets that feed the screening loop.
        assert _link_source(rfk.inputs.parent_folder) == _link_source(csp.inputs.dft_remote)
        for name in ("initial_evc_occupied1", "initial_evc_occupied2"):
            assert _link_source(rfk.inputs[name]) == _link_source(csp.inputs[name]), name

        # (2) Concrete kcp.x inputs. Both sub-graphs are expanded with
        # the same dummy stand-ins for whatever the workflow supplies
        # through a link, so the only thing that can differ is a
        # seeding decision the workflow made differently on each path.
        structure = (
            request.getfixturevalue("ozone_structure")
            if route == "molecular"
            else request.getfixturevalue("periodic_ozone_structure")
        )
        code = request.getfixturevalue("kcp_code")
        pseudos = self._pseudos(structure, request.getfixturevalue("ozone_pseudo_family"))
        numeric = self._numeric_args(structure, code, pseudos)
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")
        evc = {
            name: (
                orm.SinglefileData(io.BytesIO(b"evc"), filename=f"{name}.dat")
                if _link_source(rfk.inputs[name])
                else None
            )
            for name in ("initial_evc_occupied1", "initial_evc_occupied2")
        }

        csp_wg = ComputeScreeningParameters.build(
            **numeric,
            initial_alpha=0.6,
            correction=correction,
            init_orbitals=csp.inputs.init_orbitals.value,
            dft_remote=dummy_remote,
            **evc,
        )
        si_args = {
            sock._name: _socket_value(sock)
            for sock in TestKoopmansDSCFGraphBuild._find_task(csp_wg, "ScreeningIteration").inputs
            if sock._name not in ("_wait", "metadata", "monitors")
        }
        # The uniform-alpha generator is a task output (a socket) at this
        # point; substitute the payload so the trial's ``file_alpharef``
        # matches the injected one and cannot mask a seeding difference.
        si_args["current_alphas"] = self.ALPHAS
        trial_wg = ScreeningIteration.build(**si_args)

        # The first-orbital-dependent-run marker only reaches the
        # namelist for KIPZ (it gates the molecular inner-loop CG pass),
        # so compare it directly — otherwise the KI parametrizations
        # would not notice the skip path forgetting it.
        assert rfk.inputs.is_first_iteration.value == si_args["is_first_iteration"]

        final_wg = RunFinalKI.build(
            **numeric,
            correction=correction,
            alphas=self.ALPHAS,
            parent_folder=dummy_remote,
            # Seeding decisions taken verbatim from the skip-mode graph.
            variational_orbital_overlays=_socket_value(rfk.inputs.variational_orbital_overlays),
            initial_evc_occupied1=rfk.inputs.initial_evc_occupied1.value
            or evc["initial_evc_occupied1"],
            initial_evc_occupied2=rfk.inputs.initial_evc_occupied2.value
            or evc["initial_evc_occupied2"],
            is_first_iteration=rfk.inputs.is_first_iteration.value,
        )

        trial = self._concrete_kcp_inputs(self._kcp_step(trial_wg, "ki"))
        final = self._concrete_kcp_inputs(self._kcp_step(final_wg, "ki"))

        # Namelist comparison key by key so a failure names the culprit.
        for namelist in sorted(set(trial["parameters"]) | set(final["parameters"])):
            assert trial["parameters"].get(namelist) == final["parameters"].get(namelist), namelist
        for key in ("overlays", "read_wavefunctions", "parent_folder", "alphas"):
            assert trial[key] == final[key], key


# ----------------------------------------------------------------------
# predict_alpha_screening — plain-python callable
# ----------------------------------------------------------------------


class TestPredictAlphaScreening:
    @staticmethod
    def _call(model, descriptors, orbitals, correction="ki", init_orbitals="kohn-sham", **kwargs):
        """Predict from a per-spin self-Hartree array, via the row adapter.

        ``descriptors`` is the trial KI's ``[nspin][nbnd]`` metric; it goes
        through ``self_hartree_descriptor_rows`` so these tests exercise
        the same adapter the graph wires.
        """
        from aiida_koopmans.workgraphs.kcp import (
            predict_alpha_screening,
            self_hartree_descriptor_rows,
        )

        rows = kwargs.pop(
            "descriptor_rows",
            self_hartree_descriptor_rows._callable(  # type: ignore[attr-defined]
                metric=descriptors, orbitals=orbitals
            ),
        )
        return predict_alpha_screening._callable(  # type: ignore[attr-defined]
            model=model,
            descriptor_rows=rows,
            orbitals=orbitals,
            correction=correction,
            init_orbitals=init_orbitals,
            **kwargs,
        )

    @staticmethod
    def _orb(index, *, filled, group_id, representative, spin="none"):
        return {
            "spin": spin,
            "index": index,
            "filled": filled,
            "group_id": group_id,
            "representative": representative,
        }

    def test_predicts_every_orbital_from_its_descriptor(self):
        orbitals = [
            self._orb(1, filled=True, group_id=1, representative=True),
            self._orb(2, filled=True, group_id=2, representative=True),
            self._orb(3, filled=False, group_id=3, representative=True),
        ]
        metric = [[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]]
        result = self._call(_linear_sh_model(), metric, orbitals)
        assert result["filled"][SpinChannel.NONE] == pytest.approx([0.5, 0.6])
        assert result["empty"][SpinChannel.NONE] == pytest.approx([0.7])

    def test_group_members_inherit_the_representative_prediction(self):
        orbitals = [
            self._orb(1, filled=True, group_id=1, representative=True),
            self._orb(2, filled=True, group_id=1, representative=False),
            self._orb(3, filled=False, group_id=2, representative=True),
        ]
        # The non-representative orbital's own descriptor (-2.0) would
        # predict 0.6; the broadcast hands it the representative's 0.5.
        metric = [[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]]
        result = self._call(_linear_sh_model(), metric, orbitals)
        assert result["filled"][SpinChannel.NONE] == pytest.approx([0.5, 0.5])
        assert result["empty"][SpinChannel.NONE] == pytest.approx([0.7])

    def test_split_model_routes_filled_and_empty_to_their_submodels(self):
        model = _split_slope_sh_model()
        assert set(model["submodels"]) == {"occ", "emp"}
        # Both orbitals sit at sh=-1.5, where the two submodels disagree
        # (occ: 0.55, emp: 0.50) — a routing swap flips the assertions.
        orbitals = [
            self._orb(1, filled=True, group_id=1, representative=True),
            self._orb(2, filled=False, group_id=2, representative=True),
        ]
        metric = [[-1.5, -1.5], [-1.5, -1.5]]
        result = self._call(model, metric, orbitals)
        assert result["filled"][SpinChannel.NONE] == pytest.approx([0.55])
        assert result["empty"][SpinChannel.NONE] == pytest.approx([0.50])

    def test_model_stamp_mismatch_raises(self):
        with pytest.raises(ValueError, match="trained with correction='kipz'"):
            self._call(
                _linear_sh_model(correction="kipz"),
                [[-1.0]],
                [self._orb(1, filled=True, group_id=1, representative=True)],
            )
        with pytest.raises(ValueError, match="trained with init_orbitals='mlwfs'"):
            self._call(
                _linear_sh_model(init_orbitals="mlwfs"),
                [[-1.0]],
                [self._orb(1, filled=True, group_id=1, representative=True)],
            )

    def test_unstamped_model_raises(self):
        """A model predating the stamps (no correction field) is refused."""
        model = _linear_sh_model()
        del model["correction"]
        with pytest.raises(ValueError, match="trained with correction=None"):
            self._call(
                model, [[-1.0]], [self._orb(1, filled=True, group_id=1, representative=True)]
            )

    def test_mismatch_raises_the_typed_class_through_except_valueerror(self):
        """A rejection arrives as ``ModelMismatchError`` through ``except ValueError``."""
        from aiida_koopmans.ml import ModelMismatchError

        try:
            self._call({"descriptor": "power_spectrum"}, [], [])
        except ValueError as exc:
            assert type(exc) is ModelMismatchError
            assert exc.field == "descriptor"
        else:
            pytest.fail("ModelMismatchError was not raised")

    def test_spin_polarized_channels_read_their_own_metric_row(self):
        orbitals = [
            self._orb(1, filled=True, group_id=1, representative=True, spin="up"),
            self._orb(2, filled=False, group_id=2, representative=True, spin="up"),
            self._orb(1, filled=True, group_id=3, representative=True, spin="down"),
            self._orb(2, filled=False, group_id=4, representative=True, spin="down"),
        ]
        metric = [[-1.0, -2.0], [-3.0, -4.0]]
        result = self._call(_linear_sh_model(), metric, orbitals)
        assert result["filled"][SpinChannel.UP] == pytest.approx([0.5])
        assert result["empty"][SpinChannel.UP] == pytest.approx([0.6])
        assert result["filled"][SpinChannel.DOWN] == pytest.approx([0.7])
        assert result["empty"][SpinChannel.DOWN] == pytest.approx([0.8])

    def test_model_trained_on_another_descriptor_raises(self):
        model = _linear_sh_model()
        model["descriptor"] = "power_spectrum"
        with pytest.raises(ValueError, match="trained on `power_spectrum`"):
            self._call(
                model, [[-1.0]], [self._orb(1, filled=True, group_id=1, representative=True)]
            )


class TestPredictFromPowerSpectrumRows:
    """Prediction off multi-valued rows: the ``power_spectrum`` descriptor."""

    BASIS: ClassVar[dict] = {"n_max": 6, "l_max": 6, "r_min": 1.0, "r_max": 4.0}

    @staticmethod
    def _orb(index, *, filled, group_id, representative, spin="none"):
        return {
            "spin": spin,
            "index": index,
            "filled": filled,
            "group_id": group_id,
            "representative": representative,
        }

    def _model(self, **basis):
        """Fit a three-wide model, so a scalar row could not be substituted."""
        from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

        return ml_helpers.fit_screening_model(
            {
                "descriptors": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "alpha_targets": [0.5, 0.6, 0.7],
                "filled": [True, True, False],
                "labels": ["orb_1", "orb_2", "orb_3"],
            },
            "linear_regression",
            descriptor="power_spectrum",
            correction="ki",
            init_orbitals="mlwfs",
            radial_basis={**self.BASIS, **basis},
        )

    def _call(self, model, rows, orbitals, **overrides):
        from aiida_koopmans.ml import MLDescriptor
        from aiida_koopmans.workgraphs.kcp import predict_alpha_screening

        kwargs = {
            "correction": "ki",
            "init_orbitals": "mlwfs",
            "descriptor": MLDescriptor.POWER_SPECTRUM,
            "radial_basis": dict(self.BASIS),
        }
        kwargs.update(overrides)
        return predict_alpha_screening._callable(  # type: ignore[attr-defined]
            model=model, descriptor_rows=rows, orbitals=orbitals, **kwargs
        )

    def test_alphas_come_back_in_the_orbital_layout(self):
        orbitals = [
            self._orb(1, filled=True, group_id=1, representative=True),
            self._orb(2, filled=True, group_id=1, representative=False),
            self._orb(3, filled=False, group_id=2, representative=True),
        ]
        rows = {
            "orb_1": [1.0, 0.0, 0.0],
            "orb_2": [0.0, 1.0, 0.0],
            "orb_3": [0.0, 0.0, 1.0],
        }
        result = self._call(self._model(), rows, orbitals)
        # Two filled (the second inheriting its representative's value) and
        # one empty, in per-spin band order.
        assert result["filled"][SpinChannel.NONE] == pytest.approx([0.5, 0.5])
        assert result["empty"][SpinChannel.NONE] == pytest.approx([0.7])

    def test_a_basis_mismatch_raises_instead_of_predicting(self):
        from aiida_koopmans.ml import ModelMismatchError

        orbitals = [self._orb(1, filled=True, group_id=1, representative=True)]
        with pytest.raises(ModelMismatchError, match="radial basis"):
            self._call(self._model(r_min=0.5), {"orb_1": [1.0, 0.0, 0.0]}, orbitals)

    def test_fewer_rows_than_orbitals_names_the_counts(self):
        """A supercell run has more orbitals than primitive Wannier functions.

        The rows are per primitive Wannier function, so one of the two
        representatives has no row of its own; the message must name both
        counts rather than surfacing a bare ``KeyError``.
        """
        orbitals = [
            self._orb(1, filled=True, group_id=1, representative=True),
            self._orb(2, filled=True, group_id=2, representative=True),
        ]
        with pytest.raises(ValueError, match=r"screens 2 orbitals but 1 `power_spectrum`"):
            self._call(self._model(), {"orb_1": [1.0, 0.0, 0.0]}, orbitals)

    def test_rows_that_prefix_the_orbitals_raise_rather_than_mispredict(self):
        """Every representative finds a row, and every row is the wrong one.

        A supercell labels its orbitals ``orb_1..orb_{M*ncells}`` while the
        rows cover only the M primitive Wannier functions, so the row keys
        are a strict prefix of the orbital labels. When both group
        representatives happen to sit at an index below M, a per-label
        lookup succeeds throughout and hands each representative another
        Wannier function's power spectrum. Only the counts show it.
        """
        # One occupied and five empty primitive Wannier functions, ncells=2;
        # the empties are near-degenerate, so grouping collapses them into
        # one group whose representative is the lowest-index empty orbital.
        orbitals = [
            self._orb(1, filled=True, group_id=1, representative=False),
            self._orb(2, filled=True, group_id=1, representative=True),
            *[
                self._orb(i, filled=False, group_id=2, representative=(i == 3))
                for i in range(3, 13)
            ],
        ]
        rows = {f"orb_{i}": [float(i), 0.0, 0.0] for i in range(1, 7)}
        with pytest.raises(ValueError, match=r"screens 12 orbitals but 6 `power_spectrum`"):
            self._call(self._model(), rows, orbitals)

    def test_matching_counts_with_mismatched_labels_name_the_labels(self):
        """Equal counts but disjoint labels is a labelling fault, not a count one."""
        orbitals = [
            self._orb(1, filled=True, group_id=1, representative=True),
            self._orb(2, filled=True, group_id=2, representative=True),
        ]
        rows = {"up_orb_1": [1.0, 0.0, 0.0], "up_orb_2": [0.0, 1.0, 0.0]}
        with pytest.raises(ValueError, match=r"No `power_spectrum` descriptor row for orbital"):
            self._call(self._model(), rows, orbitals)


# ----------------------------------------------------------------------
# Predict trial vs refinement trial: identical kcp.x step inputs
# ----------------------------------------------------------------------


class TestPredictTrialMatchesComputeTrial:
    """The predict route's trial KI is the refinement route's first trial.

    Build both screening sub-graphs for identical inputs and compare the
    kwargs ``PredictScreeningParameters`` hands to ``_trial_kcp_inputs``
    with those ``ComputeScreeningParameters`` hands to
    ``ScreeningIteration`` (whose forwarding to ``_trial_kcp_inputs`` is
    a pure pass-through of the same keys). This pins the trial's
    parenting, first-iteration flag, KS overlay, Wannier-seed staging and
    overrides to the refinement route's — a predict trial that dropped
    any of them would predict off different variational orbitals and
    still pass every shape test.
    """

    FORWARDED: ClassVar[list[str]] = [
        "code",
        "structure",
        "pseudos",
        "base",
        "nbnd",
        "correction",
        "current_alphas",
        "parent_folder",
        "is_first_iteration",
        "variational_orbital_overlays",
        "initial_evc_occupied1",
        "initial_evc_occupied2",
        "ki_overrides",
        "parallelization",
    ]

    @classmethod
    def _render(cls, value):
        """Render a value comparably across two graph builds.

        ``TaggedValue`` wrappers and stored nodes carry per-instance
        uuids that are wrapper identity, not payload; scrub them.
        """
        import re

        from aiida import orm

        if type(value).__name__ in ("TaggedValue", "_TaggedScalar"):
            return re.sub(r"uuid=[0-9a-f-]+", "uuid=<>", repr(value))
        if hasattr(value, "__dataclass_fields__"):
            return {f: cls._render(getattr(value, f)) for f in sorted(value.__dataclass_fields__)}
        if hasattr(value, "_scoped_name"):
            return f"<socket {value._scoped_name}>"
        if isinstance(value, orm.Node):
            return f"<{type(value).__name__} {value.uuid}>"
        if isinstance(value, dict):
            return {k: cls._render(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        if isinstance(value, list | tuple):
            return [cls._render(v) for v in value]
        return re.sub(r"uuid=[0-9a-f-]+", "uuid=<>", repr(value))

    @pytest.mark.parametrize(
        "init_orbitals", [VariationalOrbitalType.KOHN_SHAM, VariationalOrbitalType.MLWFS]
    )
    @pytest.mark.parametrize("correction", [Correction.KI, Correction.KIPZ])
    def test_trial_inputs_match(
        self, monkeypatch, ozone_structure, kcp_code, ozone_pseudo_family, correction, init_orbitals
    ):
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs import kcp as kcp_mod

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        remote = orm.RemoteData(remote_path="/nonexistent/fake")
        evc1 = orm.SinglefileData.from_string("evc1", filename="evc_occupied1.dat").store()
        evc2 = orm.SinglefileData.from_string("evc2", filename="evc_occupied2.dat").store()

        common = {
            "code": kcp_code,
            "structure": ozone_structure,
            "pseudos": pseudos,
            "ecutwfc": 65.0,
            "ecutrho": 260.0,
            "nbnd": 10,
            "nspin": 2,
            "nelec": 18,
            "nelup": 9,
            "neldw": 9,
            "tot_magnetization": 0,
            "initial_alpha": 0.6,
            "correction": correction,
            "init_orbitals": init_orbitals,
            "dft_remote": remote,
            "self_hartree_tol": 1.5e-4,
            "initial_evc_occupied1": evc1,
            "initial_evc_occupied2": evc2,
            "overrides": {"ki": {"CONTROL": {"iprint": 7}}},
            "parallelization": {"kcp": {"ntasks": 3}},
        }

        seen_predict: dict = {}
        real_trial = kcp_mod._trial_kcp_inputs

        def spy_trial(**kwargs):
            seen_predict.update(kwargs)
            return real_trial(**kwargs)

        monkeypatch.setattr(kcp_mod, "_trial_kcp_inputs", spy_trial)
        kcp_mod.PredictScreeningParameters.build(**common, ml_model=_linear_sh_model())
        monkeypatch.undo()

        seen_compute: dict = {}
        real_iteration = kcp_mod.ScreeningIteration

        def spy_iteration(**kwargs):
            seen_compute.update(kwargs)
            return real_iteration(**kwargs)

        monkeypatch.setattr(kcp_mod, "ScreeningIteration", spy_iteration)
        kcp_mod.ComputeScreeningParameters.build(**common, alpha_numsteps=1)
        monkeypatch.undo()

        assert seen_predict, "predict never called _trial_kcp_inputs"
        assert seen_compute, "compute never called ScreeningIteration"

        diffs = {
            key: (
                self._render(seen_predict.get(key, "<ABSENT>")),
                self._render(seen_compute.get(key, "<ABSENT>")),
            )
            for key in self.FORWARDED
            if self._render(seen_predict.get(key, "<ABSENT>"))
            != self._render(seen_compute.get(key, "<ABSENT>"))
        }
        assert not diffs, diffs


class TestMlTestModeGraphBuild:
    """``ml_model`` + ``ml_test`` builds the side-by-side final KIs."""

    def _build_wg(self, *, ozone_structure, kcp_code, ozone_pseudo_family, **overrides):
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        inputs = {
            "code": kcp_code,
            "structure": ozone_structure,
            "pseudo_family": ozone_pseudo_family,
            "ecutwfc": 65.0,
            "ecutrho": 260.0,
            "nbnd": 10,
            "nspin": 2,
            "tot_magnetization": None,
            "correction": Correction.KI,
            "init_orbitals": VariationalOrbitalType.KOHN_SHAM,
            "alpha_numsteps": 1,
            "fix_spin_contamination": False,
            "initial_alpha": 0.6,
            "spin_polarized": False,
        }
        inputs.update(overrides)
        return KoopmansDSCFWorkflow.build(**inputs)

    @staticmethod
    def _labels(wg) -> list[str]:
        labels: list[str] = []

        def _walk(tasks):
            for t in tasks:
                labels.append(t.name)
                children = getattr(t, "children", None)
                if children:
                    _walk(children)

        _walk(wg.tasks)
        return labels

    def test_ml_test_keeps_refinement_and_adds_the_second_ki(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """The refinement still runs, plus prediction and a second final KI."""
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_model=_linear_sh_model(),
            ml_test=True,
        )
        labels = self._labels(wg)

        def _has(substr: str) -> bool:
            return any(substr in label for label in labels)

        # Comparison baseline: the full refinement, not the predict-only route.
        assert _has("ComputeScreeningParameters"), labels
        assert not _has("PredictScreeningParameters"), labels
        # The predicted route: model prediction + second final KI off the trial.
        assert _has("predict_alphas"), labels
        assert _has("run_final_ki_predicted"), labels
        assert _has("RunFinalKI"), labels

    def test_ml_test_allows_multiple_refinement_steps(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """``alpha_numsteps > 1`` is the comparison baseline, not an error."""
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_model=_linear_sh_model(),
            ml_test=True,
            alpha_numsteps=2,
        )
        assert any("ComputeScreeningParameters" in label for label in self._labels(wg))

    def test_ml_test_without_model_raises(self, ozone_structure, kcp_code, ozone_pseudo_family):
        """``ml_test`` without a model has nothing to predict with."""
        with pytest.raises(ValueError, match="supply ml_model"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_test=True,
            )

    def test_predicted_ki_shares_the_trial_and_applies_the_prediction(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Both final KIs hang off one trial socket; only the alphas differ.

        Socket identity is what backs the parity claim at build level: a
        second KI parented elsewhere, fed the refined alphas (every delta
        identically zero), restarted as a first iteration, or built
        without the shared overrides/nbnd passes every shape test.
        """
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_model=_linear_sh_model(),
            ml_test=True,
            overrides={"ki": {"CONTROL": {"iprint": 7}}},
        )
        by_name = {t.name: t for t in wg.tasks}
        base, predicted = by_name["RunFinalKI"], by_name["run_final_ki_predicted"]

        def source(task, port):
            links = task.inputs[port]._links
            assert len(links) == 1, (port, links)
            return links[0].from_socket

        assert source(base, "parent_folder") is source(predicted, "parent_folder")
        assert source(base, "nbnd") is source(predicted, "nbnd")
        base_alphas = source(base, "alphas")
        predicted_alphas = source(predicted, "alphas")
        assert base_alphas is not predicted_alphas
        assert predicted_alphas._task.name == "predict_alphas", predicted_alphas._task.name
        assert predicted.inputs["is_first_iteration"].value is False
        predicted_control = dict(predicted.inputs["overrides"]["CONTROL"].value)
        assert predicted_control == dict(base.inputs["overrides"]["CONTROL"].value)
        assert predicted_control["iprint"] == 7

    def test_ml_test_graph_roundtrips(self, ozone_structure, kcp_code, ozone_pseudo_family):
        """The ml_test graph survives the to_dict/from_dict round trip."""
        from tests.fixtures import assert_graph_roundtrips

        assert_graph_roundtrips(
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_model=_linear_sh_model(),
                ml_test=True,
            )
        )


class TestPowerSpectrumPredictionGraph:
    """The two prediction sites wire the decompose segment for ``power_spectrum``.

    Construction-level: nothing runs. Both sites take the same descriptor
    wiring, so both are asserted — a fix applied to one alone would leave
    ``mode: test`` predicting off self-Hartrees while claiming otherwise.
    """

    @staticmethod
    def _labels(wg) -> list[str]:
        labels: list[str] = []

        def _walk(tasks):
            for t in tasks:
                labels.append(t.name)
                children = getattr(t, "children", None)
                if children:
                    _walk(children)

        _walk(wg.tasks)
        return labels

    @staticmethod
    def _model():
        from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

        return ml_helpers.fit_screening_model(
            {
                "descriptors": [[1.0, 0.0], [0.0, 1.0]],
                "alpha_targets": [0.5, 0.7],
                "filled": [True, False],
                "labels": ["orb_1", "orb_2"],
            },
            "linear_regression",
            descriptor="power_spectrum",
            correction="ki",
            init_orbitals="mlwfs",
            radial_basis={"n_max": 4, "l_max": 4, "r_min": 0.5, "r_max": 4.0},
        )

    def _build(self, *, ozone_structure, kcp_code, ozone_pseudo_family, p2w, tmp_path, **overrides):
        from aiida import orm
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.ml import MLDescriptor
        from aiida_koopmans.workgraphs.kcp import PredictScreeningParameters
        from tests.fixtures import block_wannierization, occ_emp_merge_groups

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        inputs = {
            "code": kcp_code,
            "structure": ozone_structure,
            "pseudos": family.get_pseudos(structure=ozone_structure),
            "ecutwfc": 65.0,
            "ecutrho": 260.0,
            "nbnd": 3,
            "nspin": 2,
            "nelec": 18,
            "nelup": 2,
            "neldw": 2,
            "tot_magnetization": 0,
            "initial_alpha": 0.6,
            "correction": Correction.KI,
            "init_orbitals": VariationalOrbitalType.MLWFS,
            "dft_remote": orm.RemoteData(remote_path="/nonexistent/fake"),
            "ml_model": self._model(),
            "descriptor": MLDescriptor.POWER_SPECTRUM,
            "pw2wannier90_code": p2w,
            "nscf_remote_folder": orm.RemoteData(
                computer=p2w.computer, remote_path=str(tmp_path)
            ).store(),
            "block_wannierizations": {
                label: block_wannierization(label) for label in ("occ", "emp")
            },
            "merge_groups": occ_emp_merge_groups(),
        }
        inputs.update(overrides)
        return PredictScreeningParameters.build(**inputs)

    def test_predict_route_decomposes_instead_of_reading_self_hartrees(
        self, ozone_structure, kcp_code, ozone_pseudo_family, aiida_local_code_factory, tmp_path
    ):
        p2w = aiida_local_code_factory(
            executable="true", entry_point="koopmans.pw2wannier_decompose"
        )
        wg = self._build(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            p2w=p2w,
            tmp_path=tmp_path,
        )
        labels = self._labels(wg)
        # The trial KI still runs: it supplies the grouping metric.
        assert any("ki_trial" in label for label in labels), labels
        assert any("assign_orbital_groups" in label for label in labels), labels
        assert any("predict_alphas" in label for label in labels), labels
        # The descriptors come from the decompose segment, not the adapter:
        # follow the link the prediction actually reads its rows from.
        assert not any("self_hartree_descriptor_rows" in label for label in labels), labels
        by_name = {t.name: t for t in wg.tasks}
        rows_links = by_name["predict_alphas"].inputs["descriptor_rows"]._links
        assert len(rows_links) == 1, rows_links
        assert rows_links[0].from_socket._task.name == "descriptors_rows", labels
        assert "PowerSpectrumDescriptorWorkflow" in by_name["descriptors"].identifier
        # The slots are labelled against the run's own orbitals, so the
        # descriptor workflow itself takes nothing from the trial KI.
        labelling = by_name["descriptors_rows"]
        assert [link.from_socket._task.name for link in labelling.inputs["slots"]._links] == [
            "descriptors"
        ]
        assert [link.from_socket._task.name for link in labelling.inputs["orbitals"]._links] == [
            "assign_orbital_groups"
        ]
        assert not by_name["descriptors"].inputs._links

    def test_self_hartree_route_builds_no_decompose_segment(
        self, ozone_structure, kcp_code, ozone_pseudo_family, aiida_local_code_factory, tmp_path
    ):
        """Negative control: the live route is untouched by the new wiring."""
        p2w = aiida_local_code_factory(
            executable="true", entry_point="koopmans.pw2wannier_decompose"
        )
        wg = self._build(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            p2w=p2w,
            tmp_path=tmp_path,
            descriptor="self_hartree",
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            ml_model=_linear_sh_model(),
        )
        labels = self._labels(wg)
        assert any("self_hartree_descriptor_rows" in label for label in labels), labels
        assert not any("decompose" in label for label in labels), labels

    def test_power_spectrum_predict_graph_roundtrips(
        self, ozone_structure, kcp_code, ozone_pseudo_family, aiida_local_code_factory, tmp_path
    ):
        """The new sockets survive the to_dict/from_dict the daemon performs."""
        from tests.fixtures import assert_graph_roundtrips

        p2w = aiida_local_code_factory(
            executable="true", entry_point="koopmans.pw2wannier_decompose"
        )
        assert_graph_roundtrips(
            self._build(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                p2w=p2w,
                tmp_path=tmp_path,
            )
        )

    def test_power_spectrum_on_the_molecular_route_raises(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """The DSCF-level guard: a KS-init run wannierizes nothing to decompose."""
        from aiida_koopmans.ml import MLDescriptor
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        with pytest.raises(ValueError, match="init_orbitals"):
            KoopmansDSCFWorkflow.build(
                code=kcp_code,
                structure=ozone_structure,
                pseudo_family=ozone_pseudo_family,
                ecutwfc=65.0,
                ecutrho=260.0,
                nbnd=10,
                correction=Correction.KI,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                ml_model=self._model(),
                descriptor=MLDescriptor.POWER_SPECTRUM,
            )

    def test_power_spectrum_on_a_spin_polarized_run_raises(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """The DSCF-level guard: the descriptor is closed-shell only."""
        from aiida_koopmans.ml import MLDescriptor
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        with pytest.raises(NotImplementedError, match="spin='collinear'"):
            KoopmansDSCFWorkflow.build(
                code=kcp_code,
                structure=ozone_structure,
                pseudo_family=ozone_pseudo_family,
                ecutwfc=65.0,
                ecutrho=260.0,
                nbnd=10,
                correction=Correction.KI,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                ml_model=self._model(),
                descriptor=MLDescriptor.POWER_SPECTRUM,
                spin_polarized=True,
            )

    def test_the_dscf_forwards_every_descriptor_input_to_both_sites(
        self, ozone_structure, kcp_code, ozone_pseudo_family, monkeypatch
    ):
        """Both prediction sites receive the whole descriptor-route input set.

        The three Wannier sockets are ``None`` on this molecular route, so
        what this pins is that each site is *handed* them — a site missing
        one would fall back to its own default and silently predict off
        self-Hartrees on a Wannier run, which no molecular test can see.
        """
        import aiida_koopmans.workgraphs.kcp as kcp_mod
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow

        required = {
            "descriptor",
            "pw2wannier90_code",
            "decompose_parameters",
            "nscf_remote_folder",
            "block_wannierizations",
            "merge_groups",
        }
        seen: dict[str, set[str]] = {}
        real_predict = kcp_mod.PredictScreeningParameters
        real_twin = kcp_mod._run_predicted_final_ki

        def spy_predict(**kwargs):
            seen["PredictScreeningParameters"] = set(kwargs)
            return real_predict(**kwargs)

        def spy_twin(outputs, **kwargs):
            seen["_run_predicted_final_ki"] = set(kwargs)
            return real_twin(outputs, **kwargs)

        monkeypatch.setattr(kcp_mod, "PredictScreeningParameters", spy_predict)
        monkeypatch.setattr(kcp_mod, "_run_predicted_final_ki", spy_twin)

        common = {
            "code": kcp_code,
            "structure": ozone_structure,
            "pseudo_family": ozone_pseudo_family,
            "ecutwfc": 65.0,
            "ecutrho": 260.0,
            "nbnd": 10,
            "correction": Correction.KI,
            "init_orbitals": VariationalOrbitalType.KOHN_SHAM,
            "ml_model": _linear_sh_model(),
        }
        KoopmansDSCFWorkflow.build(**common)
        KoopmansDSCFWorkflow.build(**common, ml_test=True)

        assert required <= seen["PredictScreeningParameters"], seen
        assert required <= seen["_run_predicted_final_ki"], seen
