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
        with pytest.raises(NotImplementedError, match=r"Phase B.*alpha_numsteps=1"):
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
        # Insertion order intentionally shuffled — band index from key suffix
        # must drive the output list order.
        alphas, _ = self._run(
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
# KoopmansDSCFTask graph build — structural inspection only.
# ----------------------------------------------------------------------


class TestKoopmansDSCFGraphBuild:
    """Inspect the task graph wired by ``KoopmansDSCFTask.build`` for ozone.

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
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFTask

        return KoopmansDSCFTask.build(
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
        # plus the DFT init + the inner refinement sub-graph node.
        assert _has("resolve_pseudo_family_task"), labels
        assert _has("count_electrons_task"), labels
        assert _has("dft_init"), labels
        assert _has("KIDscfRefinementTask"), labels

        # Now build the inner refinement sub-graph independently to
        # verify the Map-zone / source-builder / gather wiring.
        from aiida import orm

        # Use the sub-graph's build entry directly. We pass plain Python
        # values for the scalar/structural inputs; ``pseudos`` and
        # ``dft_remote`` are placeholders the topology check ignores.
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import KIDscfRefinementTask, OneDSCFIteration

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        pseudos = family.get_pseudos(structure=ozone_structure)
        # Unstored placeholder — only the topology of the resulting
        # WorkGraph is inspected; the value is never dereferenced.
        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")

        sub_wg = KIDscfRefinementTask.build(
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
        # Per-orbital fan-out lives inside the ``OneDSCFIteration`` sub-graph
        # extracted by B.2; the refinement task itself only exposes
        # ``OneDSCFIteration`` + the final KI at its top level.
        assert _sub_has("OneDSCFIteration"), sub_labels
        # Final KI runs at the refinement level.
        assert _sub_has("ki_final"), sub_labels

        # Build ``OneDSCFIteration`` directly to verify its internals —
        # ``@task.graph`` sub-tasks are opaque from the parent graph at
        # build time, so the walker can't reach Map zones / source builders
        # through ``KIDscfRefinementTask`` alone.
        from aiida_koopmans.workgraphs.kcp import _kcp_base_inputs

        iter_wg = OneDSCFIteration.build(
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
