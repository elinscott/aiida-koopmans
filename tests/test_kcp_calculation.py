"""Unit tests for ``KcpCalculation`` input rendering.

Exercise the pure-function helpers on the class (``_normalize_parameters``,
``_render_namelists``, ``_render_atomic_species``, ``_render_atomic_positions``,
``_render_cell_parameters``, ``_write_alpha_file``) without booting a full
AiiDA daemon — each helper is a classmethod / staticmethod.

Uses the ozone geometry from ``koopmans/tutorials/tutorial_1/ozone.json`` as
the realistic test input.
"""

from __future__ import annotations

import io

import pytest

from aiida_koopmans.calculations.kcp import KcpCalculation


# ----------------------------------------------------------------------
# _normalize_parameters
# ----------------------------------------------------------------------


class TestNormalizeParameters:
    def test_uppercases_namelists_and_lowercases_keys(self):
        raw = {"control": {"Calculation": "cp"}, "System": {"ECUTWFC": 65.0}}
        got = KcpCalculation._normalize_parameters(raw)
        assert got == {"CONTROL": {"calculation": "cp"}, "SYSTEM": {"ecutwfc": 65.0}}

    def test_rejects_blocked_control_keys(self):
        for blocked in ("outdir", "pseudo_dir", "prefix"):
            with pytest.raises(ValueError, match="set by the CalcJob"):
                KcpCalculation._normalize_parameters({"CONTROL": {blocked: "nope"}})

    def test_rejects_blocked_system_keys(self):
        for blocked in ("nat", "ntyp", "ibrav"):
            with pytest.raises(ValueError, match="set by the CalcJob"):
                KcpCalculation._normalize_parameters({"SYSTEM": {blocked: 1}})

    def test_rejects_non_dict_namelist(self):
        with pytest.raises(ValueError, match="must map to a dict"):
            KcpCalculation._normalize_parameters({"CONTROL": "cp"})


# ----------------------------------------------------------------------
# _render_namelists
# ----------------------------------------------------------------------


class TestRenderNamelists:
    def test_canonical_order(self):
        params = {
            "NKSIC": {"do_innerloop": True},
            "CONTROL": {"calculation": "cp"},
            "SYSTEM": {"nspin": 2},
        }
        text = KcpCalculation._render_namelists(params)
        order = [text.index(f"&{name}") for name in ("CONTROL", "SYSTEM", "NKSIC")]
        assert order == sorted(order), f"namelists emitted out of order:\n{text}"
        assert "&CONTROL" in text
        assert "&SYSTEM" in text
        assert "&NKSIC" in text
        # Every namelist closes with '/'
        assert text.count("/\n") == 3

    def test_fortran_boolean_and_quoted_string(self):
        params = {"CONTROL": {"calculation": "cp", "verbosity": "low"}, "NKSIC": {"do_innerloop": True}}
        text = KcpCalculation._render_namelists(params)
        assert "calculation = 'cp'" in text
        assert "verbosity = 'low'" in text
        assert "do_innerloop = .true." in text

    def test_fortran_float_and_int(self):
        params = {"SYSTEM": {"ecutwfc": 65.0, "nbnd": 10}}
        text = KcpCalculation._render_namelists(params)
        # Reals get scientific notation with 'd' exponent
        assert "ecutwfc = " in text
        assert "d" in text.split("ecutwfc = ", 1)[1].split("\n", 1)[0]
        assert "nbnd = 10" in text

    def test_unexpected_namelist_emitted_at_end(self):
        params = {"CONTROL": {"calculation": "cp"}, "CUSTOM": {"x": 1}}
        text = KcpCalculation._render_namelists(params)
        assert text.index("&CONTROL") < text.index("&CUSTOM")

    def test_empty_namelists_skipped(self):
        params = {"CONTROL": {"calculation": "cp"}, "SYSTEM": {}}
        text = KcpCalculation._render_namelists(params)
        assert "&SYSTEM" not in text


# ----------------------------------------------------------------------
# _render_atomic_species / _render_atomic_positions / _render_cell_parameters
# ----------------------------------------------------------------------


def test_render_atomic_species(ozone_structure, ozone_pseudos):
    text = KcpCalculation._render_atomic_species(ozone_structure, ozone_pseudos)
    assert text.startswith("ATOMIC_SPECIES\n")
    assert "O" in text
    assert "O.upf" in text
    # Atomic mass of O is ~15.999
    import re

    mass_match = re.search(r"^\s+O\s+(\d+\.\d+)\s+O\.upf", text, flags=re.MULTILINE)
    assert mass_match is not None
    assert 15.0 < float(mass_match.group(1)) < 17.0


def test_render_atomic_positions(ozone_structure):
    text = KcpCalculation._render_atomic_positions(ozone_structure)
    assert text.startswith("ATOMIC_POSITIONS angstrom\n")
    for site in ozone_structure.sites:
        for coord in site.position:
            assert f"{coord:.10f}" in text
    body = text.splitlines()[1:]
    assert len(body) == len(ozone_structure.sites)


def test_render_cell_parameters(ozone_structure):
    text = KcpCalculation._render_cell_parameters(ozone_structure)
    assert text.startswith("CELL_PARAMETERS angstrom\n")
    for vec in ozone_structure.cell:
        for coord in vec:
            assert f"{coord:.10f}" in text
    body = text.splitlines()[1:]
    assert len(body) == 3


# ----------------------------------------------------------------------
# _write_alpha_file
# ----------------------------------------------------------------------


class FakeFolder:
    """Minimal stand-in for the sandbox folder that ``prepare_for_submission`` receives."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def open(self, name, mode="r", encoding=None):  # noqa: ARG002
        buf = io.StringIO()
        parent = self

        class _Handle:
            def __enter__(_self):  # noqa: N805
                return buf

            def __exit__(_self, *exc):  # noqa: N805
                parent.files[name] = buf.getvalue()
                return False

        return _Handle()


def test_write_alpha_file_format_filled():
    folder = FakeFolder()
    KcpCalculation._write_alpha_file(folder, [0.7, 0.7, 0.7], "file_alpharef.txt")
    content = folder.files["file_alpharef.txt"]
    lines = content.splitlines()
    assert lines[0] == "3"
    assert lines[1] == "1 0.7 1.0"
    assert lines[2] == "2 0.7 1.0"
    assert lines[3] == "3 0.7 1.0"


def test_write_alpha_file_empty_list_emits_header_only():
    folder = FakeFolder()
    KcpCalculation._write_alpha_file(folder, [], "file_alpharef_empty.txt")
    content = folder.files["file_alpharef_empty.txt"]
    assert content == "0\n"


# ----------------------------------------------------------------------
# End-to-end rendering: ozone DFT step input file
# ----------------------------------------------------------------------


def test_full_ozone_input_rendering_has_expected_sections(ozone_structure, ozone_pseudos):
    """Render a plausible ozone DFT-init input and check the overall structure."""
    # After normalisation + injection, these are the params the CalcJob would use.
    params = KcpCalculation._normalize_parameters(
        {
            "CONTROL": {"calculation": "cp", "verbosity": "low", "iprint": 1},
            "SYSTEM": {"ecutwfc": 65.0, "ecutrho": 260.0, "nbnd": 10, "nspin": 2},
            "ELECTRONS": {"electron_dynamics": "cg", "maxiter": 300, "do_outerloop": True},
            "NKSIC": {"do_innerloop": True, "do_orbdep": False},
        }
    )
    # Mimic what prepare_for_submission would inject
    params["CONTROL"]["outdir"] = "./out/"
    params["CONTROL"]["pseudo_dir"] = "./pseudo/"
    params["CONTROL"]["prefix"] = "aiida"
    params["SYSTEM"]["ibrav"] = 0
    params["SYSTEM"]["nat"] = 3
    params["SYSTEM"]["ntyp"] = 1

    content = (
        KcpCalculation._render_namelists(params)
        + KcpCalculation._render_atomic_species(ozone_structure, ozone_pseudos)
        + KcpCalculation._render_atomic_positions(ozone_structure)
        + KcpCalculation._render_cell_parameters(ozone_structure)
    )

    # Namelists present and in order
    assert content.index("&CONTROL") < content.index("&SYSTEM")
    assert content.index("&SYSTEM") < content.index("&ELECTRONS")
    assert content.index("&ELECTRONS") < content.index("&NKSIC")

    # Injected CalcJob-owned keys present
    assert "outdir = './out/'" in content
    assert "pseudo_dir = './pseudo/'" in content
    assert "prefix = 'aiida'" in content
    assert "ibrav = 0" in content
    assert "nat = 3" in content
    assert "ntyp = 1" in content

    # Cards after namelists
    assert content.index("/\n") < content.index("ATOMIC_SPECIES\n")
    assert content.index("ATOMIC_SPECIES") < content.index("ATOMIC_POSITIONS")
    assert content.index("ATOMIC_POSITIONS") < content.index("CELL_PARAMETERS")

    # Ozone positions faithfully rendered
    assert "7.0869000000" in content
    assert "8.1738000000" in content
    assert "6.0000000000" in content
