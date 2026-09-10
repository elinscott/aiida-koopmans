"""Unit tests for the shared CalcJob base helpers.

Exercise the pure staticmethods on ``KoopmansStdoutCalculation``
(``render_namelist``, ``_write_alpha_file``) directly — no AiiDA daemon
needed. The subclass plugins inherit these, so their own suites cover the
wiring; these tests pin the golden output of the shared implementation.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from aiida_koopmans.calculations.base import KoopmansCalculation, KoopmansStdoutCalculation


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


def _computer(default_mpiprocs=None, label="test-computer"):
    """Minimal stand-in for ``orm.Computer`` -- only the two attributes used."""
    return SimpleNamespace(label=label, get_default_mpiprocs_per_machine=lambda: default_mpiprocs)


class TestResolveTotalMpiprocs:
    """``_resolve_total_mpiprocs`` backs the wann2kcp.x / merge_evc.x single-rank checks.

    A remote computer's ``resources`` often carries only ``num_machines`` and
    relies on the computer's ``default_mpiprocs_per_machine`` for the rest --
    the same resolution AiiDA's own scheduler validation applies at
    submission. Assuming ``num_mpiprocs_per_machine=1`` in that case (the
    pre-fix behaviour) undercounts and lets an actually-parallel job pass the
    single-rank check.
    """

    def test_tot_num_mpiprocs_wins_outright(self):
        resources = {"tot_num_mpiprocs": 8, "num_machines": 1}
        assert KoopmansCalculation._resolve_total_mpiprocs(resources, _computer(4)) == 8

    def test_explicit_num_mpiprocs_per_machine(self):
        resources = {"num_machines": 2, "num_mpiprocs_per_machine": 3}
        assert KoopmansCalculation._resolve_total_mpiprocs(resources, _computer(99)) == 6

    def test_falls_back_to_computer_default(self):
        resources = {"num_machines": 4}
        assert KoopmansCalculation._resolve_total_mpiprocs(resources, _computer(8)) == 32

    def test_num_machines_defaults_to_one(self):
        resources = {}
        assert KoopmansCalculation._resolve_total_mpiprocs(resources, _computer(1)) == 1

    def test_underdetermined_raises(self):
        resources = {"num_machines": 4}
        with pytest.raises(ValueError, match="cannot determine the MPI process count"):
            KoopmansCalculation._resolve_total_mpiprocs(resources, _computer(None))

    def test_no_computer_and_no_explicit_count_raises(self):
        resources = {"num_machines": 4}
        with pytest.raises(ValueError, match="cannot determine the MPI process count"):
            KoopmansCalculation._resolve_total_mpiprocs(resources, None)
