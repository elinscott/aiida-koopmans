"""Tests for the per-step ``CONTROL.calculation`` enforcement backstop.

The pure-function tests exercise :func:`enforce_step_calculation` directly.
The graph-build tests construct ``RunScfNscf`` / ``RunPwBands`` / ``Wannierize``
and assert each step owner stamps its own calculation mode on the built task,
and that a genuinely conflicting override raises.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.workgraphs import enforce_step_calculation

# ----------------------------------------------------------------------
# enforce_step_calculation — pure dict function
# ----------------------------------------------------------------------


class TestEnforceStepCalculation:
    def test_stamps_when_absent(self):
        params: dict = {"SYSTEM": {"nbnd": 20}}
        out = enforce_step_calculation(params, "nscf", "nscf")
        assert out["CONTROL"]["calculation"] == "nscf"
        # Mutates in place and returns the same object.
        assert out is params
        assert out["SYSTEM"]["nbnd"] == 20

    def test_accepts_matching_explicit_value(self):
        params = {"CONTROL": {"calculation": "bands", "verbosity": "high"}}
        out = enforce_step_calculation(params, "bands", "bands")
        assert out["CONTROL"]["calculation"] == "bands"
        assert out["CONTROL"]["verbosity"] == "high"

    def test_conflicting_value_raises_naming_step_and_values(self):
        params = {"CONTROL": {"calculation": "scf"}}
        with pytest.raises(ValueError, match=r"'nscf' step.*nscf.*scf"):
            enforce_step_calculation(params, "nscf", "nscf")


# ----------------------------------------------------------------------
# Graph-build integration: each step owner stamps its own calculation
# ----------------------------------------------------------------------


def _calc(task, *namespace):
    """Return CONTROL.calculation of a built task's ``<namespace...>.parameters``."""
    node = task.inputs
    for key in namespace:
        node = node[key]
    return node["parameters"].value.get_dict()["CONTROL"]["calculation"]


class TestRunScfNscfEnforcement:
    def test_k2_shaped_nscf_override_lands_as_nscf(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        """A k2-shaped nscf override (no calculation key) resolves to 'nscf'."""
        from aiida_koopmans.workgraphs.pw import RunScfNscf

        wg = RunScfNscf.build(
            code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kmesh,
            overrides={"nscf": {"pw": {"parameters": {"SYSTEM": {"nbnd": 20}}}}},
        )
        assert _calc(wg.tasks["scf"], "pw") == "scf"
        assert _calc(wg.tasks["nscf"], "pw") == "nscf"

    def test_conflicting_nscf_override_raises(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        from aiida_koopmans.workgraphs.pw import RunScfNscf

        with pytest.raises(ValueError, match=r"'nscf' step"):
            RunScfNscf.build(
                code=pw_code,
                structure=silicon_structure,
                pseudo_family=fake_cutoffs_family.label,
                nscf_kpoints=kmesh,
                overrides={"nscf": {"pw": {"parameters": {"CONTROL": {"calculation": "bands"}}}}},
            )

    def test_negative_control_without_enforcement_a_scf_override_wins(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code, monkeypatch
    ):
        """Neutralizing the backstop lets a k2-shaped ``calculation='scf'`` leak win.

        Holds the build infrastructure constant and removes only the
        enforcement: an nscf override carrying ``calculation='scf'`` (exactly
        what today's koopmans main injects into every pw override) then reaches
        the nscf task unchanged, which is the defect this PR backstops.
        """
        from aiida_koopmans.workgraphs import pw as pw_module

        monkeypatch.setattr(
            pw_module, "enforce_step_calculation", lambda params, step, expected: params
        )
        wg = pw_module.RunScfNscf.build(
            code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kmesh,
            overrides={"nscf": {"pw": {"parameters": {"CONTROL": {"calculation": "scf"}}}}},
        )
        assert _calc(wg.tasks["nscf"], "pw") == "scf"


class TestRunPwBandsEnforcement:
    def test_k2_shaped_bands_override_lands_as_bands(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        """A k2-shaped bands override (no calculation key) resolves to 'bands'."""
        from aiida_koopmans.workgraphs.pw import RunPwBands

        wg = RunPwBands.build(
            code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kmesh,
            overrides={"bands": {"pw": {"parameters": {"SYSTEM": {"nbnd": 20}}}}},
        )
        task = wg.tasks["PwBandsWorkChain"]
        assert _calc(task, "scf", "pw") == "scf"
        assert _calc(task, "bands", "pw") == "bands"

    def test_conflicting_bands_override_raises(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        from aiida_koopmans.workgraphs.pw import RunPwBands

        with pytest.raises(ValueError, match=r"'bands' step"):
            RunPwBands.build(
                code=pw_code,
                structure=silicon_structure,
                pseudo_family=fake_cutoffs_family.label,
                bands_kpoints=kmesh,
                overrides={"bands": {"pw": {"parameters": {"CONTROL": {"calculation": "scf"}}}}},
            )


class TestWannierizeEnforcement:
    """Exercise the nscf enforcement shared by ``Wannierize`` / ``OptimizeWannierization``.

    Both route through ``_finalize_wannier_builder``, which is where the nscf
    step stamps its calculation. Building the whole ``Wannierize`` graph is
    avoided here because its output linking trips an unrelated ``NotRequired``
    output-socket mismatch (``disentanglement_data`` / ``spread_data``) for a
    run without disentanglement — orthogonal to this backstop.
    """

    def _builder(self, wannier_codes, structure, pseudo_family, nscf_override):
        from aiida_wannier90_workflows.workflows import Wannier90WorkChain

        return Wannier90WorkChain.get_builder_from_protocol(
            codes=wannier_codes,
            structure=structure,
            pseudo_family=pseudo_family,
            overrides={"nscf": nscf_override},
        )

    def test_k2_shaped_nscf_override_lands_as_nscf(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        """A k2-shaped nscf override (no calculation key) resolves to 'nscf'."""
        from aiida_koopmans.workgraphs.wannier90 import _finalize_wannier_builder

        builder = self._builder(
            wannier_codes,
            silicon_structure,
            fake_cutoffs_family.label,
            {"pw": {"parameters": {"SYSTEM": {"nbnd": 20}}}},
        )
        data = _finalize_wannier_builder(
            builder,
            kpoint_path=None,
            bands_kpoints=None,
            projector_rotation=None,
            set_bands_kpoints=True,
        )
        assert data["nscf"]["pw"]["parameters"].get_dict()["CONTROL"]["calculation"] == "nscf"

    def test_conflicting_nscf_override_raises(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        from aiida_koopmans.workgraphs.wannier90 import _finalize_wannier_builder

        builder = self._builder(
            wannier_codes,
            silicon_structure,
            fake_cutoffs_family.label,
            {"pw": {"parameters": {"CONTROL": {"calculation": "scf"}}}},
        )
        with pytest.raises(ValueError, match=r"'nscf' step"):
            _finalize_wannier_builder(
                builder,
                kpoint_path=None,
                bands_kpoints=None,
                projector_rotation=None,
                set_bands_kpoints=True,
            )
