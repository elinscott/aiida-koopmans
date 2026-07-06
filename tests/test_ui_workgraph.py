"""Tests for the unfold-and-interpolate workgraph in ``workgraphs/ui.py``.

Everything in this graph is a pure-python ``@task.calcfunction``, so unlike
the QE-backed workgraph tests these can execute the graph end-to-end
(``wg.run()``) against the silicon fixtures in ``tests/data/ui/`` and
compare the stored ``BandsData`` / ``XyData`` with the legacy reference.
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
    """Load the legacy-generated silicon reference data."""
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
        """do_dos=True wires both calcfunctions."""
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
    """End-to-end execution against the legacy silicon reference."""

    def test_bands_and_dos_match_legacy(self, si_ui_inputs, si_reference):
        """The stored BandsData and XyData reproduce the legacy numbers."""
        from aiida_koopmans.workgraphs.ui import UnfoldAndInterpolateTask

        wg = UnfoldAndInterpolateTask.build(**si_ui_inputs, do_dos=True)
        wg.run()

        bands_node = wg.tasks.interpolate_bands.outputs.result.value
        assert isinstance(bands_node, orm.BandsData)
        assert np.allclose(bands_node.get_bands(), si_reference["energies"], atol=1e-10)
        assert np.allclose(bands_node.get_kpoints(), si_reference["kpath_kpts"])

        dos_node = wg.tasks.compute_dos_from_bands.outputs.result.value
        assert isinstance(dos_node, orm.XyData)
        assert np.allclose(dos_node.get_x()[1], si_reference["dos_energies"], atol=1e-10)
        assert np.allclose(dos_node.get_y()[0][1], si_reference["dos_values"], atol=1e-8)
