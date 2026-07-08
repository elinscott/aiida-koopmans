"""Unit tests for the shared CalcJob base helpers.

Exercise the pure staticmethods on ``KoopmansStdoutCalculation``
(``render_namelist``, ``_write_alpha_file``) directly — no AiiDA daemon
needed. The subclass plugins inherit these, so their own suites cover the
wiring; these tests pin the golden output of the shared implementation.
"""

from __future__ import annotations

import io

from aiida_koopmans.calculations.base import KoopmansStdoutCalculation


class TestRenderNamelist:
    def test_golden_output(self):
        text = KoopmansStdoutCalculation.render_namelist(
            "INPUTPP", {"seedname": "wannier90", "gamma_trick": True, "print_rho": False}
        )
        assert text == (
            "&INPUTPP\n  seedname = 'wannier90'\n  gamma_trick = .true.\n  print_rho = .false.\n/\n"
        )

    def test_empty_namelist_is_header_and_close(self):
        assert KoopmansStdoutCalculation.render_namelist("SCREEN", {}) == "&SCREEN\n/\n"


class FakeFolder:
    """Minimal stand-in for the sandbox folder ``prepare_for_submission`` receives."""

    def __init__(self):
        self.files: dict[str, str] = {}

    def open(self, name, mode="r", encoding=None):
        buf = io.StringIO()
        parent = self

        class _Handle:
            def __enter__(_self):  # noqa: N805
                return buf

            def __exit__(_self, *exc):  # noqa: N805
                parent.files[name] = buf.getvalue()
                return False

        return _Handle()


class TestWriteAlphaFile:
    def test_filled(self):
        folder = FakeFolder()
        KoopmansStdoutCalculation._write_alpha_file(folder, [0.7, 0.7, 0.7], "file_alpharef.txt")
        assert folder.files["file_alpharef.txt"] == "3\n1 0.7 1.0\n2 0.7 1.0\n3 0.7 1.0\n"

    def test_empty_list_emits_header_only(self):
        folder = FakeFolder()
        KoopmansStdoutCalculation._write_alpha_file(folder, [], "file_alpharef_empty.txt")
        assert folder.files["file_alpharef_empty.txt"] == "0\n"
