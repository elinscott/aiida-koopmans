"""Every route runs pw.x at ``CONTROL.verbosity = 'high'``.

``aiida_koopmans.owned_keywords`` classifies ``pw.CONTROL.verbosity`` as
owned, and ``koopmans`` drops it from the input file on that basis. These
build each pw.x-running graph and assert the value on every pw step it
assembles, and that a caller stating ``'low'`` does not win — upstream
merges a *default* verbosity, which a caller value would otherwise survive.
"""

from __future__ import annotations

import pytest

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _control(task, *namespace):
    """Return the ``CONTROL`` namelist of a built task's ``<namespace...>.parameters``."""
    node = task.inputs
    for key in namespace:
        node = node[key]
    return node["parameters"].value.get_dict()["CONTROL"]


def _low(step: str) -> dict:
    """Return an overrides dict asking ``step`` for the lowest pw.x verbosity."""
    return {step: {"pw": {"parameters": {"CONTROL": {"verbosity": "low"}}}}}


# ----------------------------------------------------------------------
# force_pw_verbosity — pure dict function
# ----------------------------------------------------------------------


class TestForcePwVerbosity:
    def test_sets_the_value_and_keeps_the_rest(self, aiida_profile):
        from aiida import orm

        from aiida_koopmans.workgraphs import force_pw_verbosity

        pw_inputs = {"parameters": orm.Dict({"CONTROL": {"calculation": "nscf"}})}
        force_pw_verbosity(pw_inputs)
        control = pw_inputs["parameters"].get_dict()["CONTROL"]
        assert control["verbosity"] == "high"
        assert control["calculation"] == "nscf"

    def test_replaces_a_stated_value(self, aiida_profile):
        from aiida import orm

        from aiida_koopmans.workgraphs import force_pw_verbosity

        pw_inputs = {"parameters": orm.Dict({"CONTROL": {"verbosity": "low"}})}
        force_pw_verbosity(pw_inputs)
        assert pw_inputs["parameters"].get_dict()["CONTROL"]["verbosity"] == "high"


# ----------------------------------------------------------------------
# One graph per pw.x-running route
# ----------------------------------------------------------------------


class TestRunScfNscf:
    """The scf + nscf pair the DFPT chain and the block Wannierisation share."""

    def _build(self, fake_cutoffs_family, silicon_structure, kmesh, pw_code, overrides=None):
        from aiida_koopmans.workgraphs.pw import RunScfNscf

        return RunScfNscf.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kmesh,
            overrides=overrides,
        )

    def test_both_steps_run_at_high_verbosity(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        wg = self._build(fake_cutoffs_family, silicon_structure, kmesh, pw_code)
        assert _control(wg.tasks["scf"], "pw")["verbosity"] == "high"
        assert _control(wg.tasks["nscf"], "pw")["verbosity"] == "high"

    @pytest.mark.parametrize("step", ["scf", "nscf"])
    def test_a_stated_value_does_not_win(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code, step
    ):
        wg = self._build(
            fake_cutoffs_family, silicon_structure, kmesh, pw_code, overrides=_low(step)
        )
        assert _control(wg.tasks[step], "pw")["verbosity"] == "high"

    def test_negative_control_without_forcing_a_stated_value_wins(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code, monkeypatch
    ):
        """Neutralizing the forcing lets ``verbosity = 'low'`` reach the task.

        Holds the build infrastructure constant and removes only the forcing,
        so the assertions above discriminate: upstream neither blocks the
        keyword nor overrules a caller value.
        """
        from aiida_koopmans.workgraphs import pw as pw_module

        monkeypatch.setattr(pw_module, "force_pw_verbosity", lambda pw_inputs: None)
        wg = pw_module.RunScfNscf.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kmesh,
            overrides=_low("nscf"),
        )
        assert _control(wg.tasks["nscf"], "pw")["verbosity"] == "low"


class TestRunPwBands:
    """The ``dft_bands`` route: scf + bands inside ``PwBandsWorkChain``."""

    def _build(self, fake_cutoffs_family, silicon_structure, kmesh, pw_code, overrides=None):
        from aiida_koopmans.workgraphs.pw import RunPwBands

        return RunPwBands.build(
            codes={"pw": pw_code},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kmesh,
            overrides=overrides,
        )

    def test_both_steps_run_at_high_verbosity(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        wg = self._build(fake_cutoffs_family, silicon_structure, kmesh, pw_code)
        task = wg.tasks["PwBandsWorkChain"]
        assert _control(task, "scf", "pw")["verbosity"] == "high"
        assert _control(task, "bands", "pw")["verbosity"] == "high"

    @pytest.mark.parametrize("step", ["scf", "bands"])
    def test_a_stated_value_does_not_win(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code, step
    ):
        wg = self._build(
            fake_cutoffs_family, silicon_structure, kmesh, pw_code, overrides=_low(step)
        )
        assert _control(wg.tasks["PwBandsWorkChain"], step, "pw")["verbosity"] == "high"


class TestWannierize:
    """The whole-manifold Wannierisation: scf + nscf inside ``Wannier90WorkChain``."""

    def _build(self, fake_cutoffs_family, silicon_structure, wannier_codes, overrides=None):
        from aiida_koopmans.workgraphs.wannier90 import Wannierize

        return Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            overrides=overrides,
        )

    def test_both_steps_run_at_high_verbosity(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        wg = self._build(fake_cutoffs_family, silicon_structure, wannier_codes)
        task = wg.tasks["Wannier90WorkChain"]
        assert _control(task, "scf", "pw")["verbosity"] == "high"
        assert _control(task, "nscf", "pw")["verbosity"] == "high"

    @pytest.mark.parametrize("step", ["scf", "nscf"])
    def test_a_stated_value_does_not_win(
        self, fake_cutoffs_family, silicon_structure, wannier_codes, step
    ):
        wg = self._build(
            fake_cutoffs_family, silicon_structure, wannier_codes, overrides=_low(step)
        )
        assert _control(wg.tasks["Wannier90WorkChain"], step, "pw")["verbosity"] == "high"


class TestRunPdos:
    """The projected-DOS route: scf + nscf inside ``PdosWorkChain``."""

    def _build(self, fake_cutoffs_family, silicon_structure, run_pdos_codes, overrides=None):
        from aiida_koopmans.workgraphs.pdos import RunPdos

        return RunPdos.build(
            codes=run_pdos_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            overrides=overrides,
        )

    def test_both_steps_run_at_high_verbosity(
        self, fake_cutoffs_family, silicon_structure, run_pdos_codes
    ):
        wg = self._build(fake_cutoffs_family, silicon_structure, run_pdos_codes)
        task = wg.tasks["PdosWorkChain"]
        assert _control(task, "scf", "pw")["verbosity"] == "high"
        assert _control(task, "nscf", "pw")["verbosity"] == "high"

    @pytest.mark.parametrize("step", ["scf", "nscf"])
    def test_a_stated_value_does_not_win(
        self, fake_cutoffs_family, silicon_structure, run_pdos_codes, step
    ):
        wg = self._build(
            fake_cutoffs_family, silicon_structure, run_pdos_codes, overrides=_low(step)
        )
        assert _control(wg.tasks["PdosWorkChain"], step, "pw")["verbosity"] == "high"


class TestDielectricTask:
    """The ``dft_eps`` route: the scf the ph.x response is taken about."""

    def _build(self, fake_cutoffs_family, silicon_structure, ph_codes, overrides=None):
        from aiida_koopmans.workgraphs.ph import DielectricTask

        return DielectricTask.build(
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            overrides=overrides,
        )

    def test_the_scf_runs_at_high_verbosity(self, fake_cutoffs_family, silicon_structure, ph_codes):
        wg = self._build(fake_cutoffs_family, silicon_structure, ph_codes)
        assert _control(wg.tasks["scf"], "pw")["verbosity"] == "high"

    def test_a_stated_value_does_not_win(self, fake_cutoffs_family, silicon_structure, ph_codes):
        wg = self._build(fake_cutoffs_family, silicon_structure, ph_codes, overrides=_low("scf"))
        assert _control(wg.tasks["scf"], "pw")["verbosity"] == "high"


class TestBandsStep:
    """The pw.x reference bands a Wannierisation runs alongside its own."""

    def test_high_verbosity_on_the_bands_step(
        self, fake_cutoffs_family, silicon_structure, wannier_codes, labelled_kpath
    ):
        from aiida_koopmans.workgraphs.wannier90 import Wannierize

        wg = Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=labelled_kpath,
        )
        assert _control(wg.tasks["bands"], "pw")["verbosity"] == "high"
