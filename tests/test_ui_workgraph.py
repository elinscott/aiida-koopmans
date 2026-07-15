"""Tests for the unfold-and-interpolate workgraph in ``workgraphs/ui.py``.

Everything in this graph is a pure-python ``@task``, so unlike the
QE-backed workgraph tests these can execute the graph end-to-end
(``wg.run()``) against the silicon fixtures in ``tests/data/ui/`` and
compare the outputs with the reference data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from aiida import orm

DATA_DIR = Path(__file__).parent / "data" / "ui"


@pytest.fixture(scope="module")
def si_reference() -> dict:
    """Load the silicon reference data."""
    with open(DATA_DIR / "si_ui_reference.json") as handle:
        return json.load(handle)


@pytest.fixture
def si_ui_inputs(aiida_profile, si_reference):
    """Assemble the ORM inputs for a silicon unfold-and-interpolate run."""
    structure = orm.StructureData(cell=si_reference["cell"])
    structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si")
    structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si")

    kpath = orm.KpointsData()
    kpath.set_kpoints(np.array(si_reference["kpath_kpts"]))
    kpath.labels = [(0, "GAMMA"), (len(si_reference["kpath_kpts"]) - 1, "L")]

    return {
        "kc_ham_file": orm.SinglefileData(DATA_DIR / "kc_ham.dat"),
        "wannier90_wout": orm.SinglefileData(DATA_DIR / "wann.wout"),
        "structure": structure,
        "kpath": kpath,
        "kgrid": list(si_reference["kgrid"]),
        "do_map": True,
        "use_ws_distance": True,
        "dft_ham_file": orm.SinglefileData(DATA_DIR / "dft_ham.dat"),
        "dft_smooth_ham_file": orm.SinglefileData(DATA_DIR / "smooth_dft_ham.dat"),
        "plotting": {"degauss": 0.05, "nstep": 1000, "Emin": -10, "Emax": 4},
    }


class TestBuild:
    """Graph construction (no execution)."""

    def test_build_with_dos(self, si_ui_inputs):
        """do_dos=True wires both tasks."""
        from aiida_koopmans.workgraphs.ui import UnfoldAndInterpolateTask

        wg = UnfoldAndInterpolateTask.build(**si_ui_inputs, do_dos=True)
        names = wg.get_task_names()
        assert "interpolate_bands" in names
        assert "compute_dos_from_bands" in names

    def test_build_without_dos(self, si_ui_inputs):
        """do_dos=False leaves only the interpolation task."""
        from aiida_koopmans.workgraphs.ui import UnfoldAndInterpolateTask

        wg = UnfoldAndInterpolateTask.build(**si_ui_inputs, do_dos=False)
        names = wg.get_task_names()
        assert "interpolate_bands" in names
        assert "compute_dos_from_bands" not in names

    def test_build_without_smooth_hamiltonians(self, si_ui_inputs):
        """The DFT Hamiltonian inputs are genuinely optional."""
        from aiida_koopmans.workgraphs.ui import UnfoldAndInterpolateTask

        si_ui_inputs.pop("dft_ham_file")
        si_ui_inputs.pop("dft_smooth_ham_file")
        si_ui_inputs["kc_ham_file"] = orm.SinglefileData(DATA_DIR / "dft_ham.dat")
        si_ui_inputs["do_map"] = False
        wg = UnfoldAndInterpolateTask.build(**si_ui_inputs, do_dos=False)
        assert "interpolate_bands" in wg.get_task_names()


class TestRun:
    """End-to-end execution against the silicon reference."""

    def test_bands_and_dos_match_reference(self, si_ui_inputs, si_reference):
        """The interpolated bands and DOS reproduce the reference numbers."""
        from aiida_koopmans.workgraphs.ui import UnfoldAndInterpolateTask

        wg = UnfoldAndInterpolateTask.build(**si_ui_inputs, do_dos=True)
        wg.run()

        bands = wg.tasks.interpolate_bands.outputs.result.value
        assert np.allclose(bands.get_list(), si_reference["energies"], atol=1e-10)

        dos = wg.tasks.compute_dos_from_bands.outputs
        assert np.allclose(dos.energies.value.get_list(), si_reference["dos_energies"], atol=1e-10)
        assert np.allclose(dos.dos.value.get_list(), si_reference["dos_values"], atol=1e-8)
