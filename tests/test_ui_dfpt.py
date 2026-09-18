"""The DFPT route's band structure: kcw.x's Hamiltonian through the shared interpolator.

The smooth-interpolation maths is tested in ``test_ui_smooth_helpers.py``
(analytically) and ``test_ui_helpers.py`` (against the reference
implementation's numbers). What is specific to the DFPT route, and tested
here, is that kcw.x's own ``*.kcw_hr_*.dat`` files enter that machinery
correctly: the right R-vectors, the right units, the right centres. The
discriminating check is a live run's own output — kcw.x interpolates the
same Hamiltonian internally, so its printed bands are an independent
answer to the same question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiida_koopmans.workgraphs.ui import helpers as ui_helpers

DATA_DIR = Path(__file__).parent / "data" / "ui" / "dfpt"

#: The occupied and empty manifolds' band slices in the concatenated
#: eigenvalue table the live silicon run produced (four Wannier functions
#: each).
MANIFOLD_BANDS = {"occ": slice(0, 4), "emp": slice(4, 8)}


def _interpolate(si_kcw_reference: dict, manifold: str, **dft) -> np.ndarray:
    """Interpolate one manifold of the live run, off kcw.x's own Hamiltonian."""
    return ui_helpers.unfold_and_interpolate(
        hr_content=(DATA_DIR / f"kcw_hr_{manifold}.dat").read_text(),
        centers=np.array(si_kcw_reference["centres"][manifold]),
        cell=np.array(si_kcw_reference["cell"]),
        kgrid=tuple(si_kcw_reference["kgrid"]),
        kpath_kpts=np.array(si_kcw_reference["kpath_kpts"]),
        **dft,
    )


class TestKcwHamiltoniansInterpolateAsKcwDoes:
    """kcw.x's real-space Hamiltonian, read back, gives kcw.x's own bands."""

    @pytest.mark.parametrize("manifold", ["occ", "emp"])
    def test_a_manifold_reproduces_the_eigenvalues_kcw_printed(
        self, si_kcw_reference: dict, manifold: str
    ):
        """The interpolation reproduces kcw.x's ``ham`` bands to its printed precision.

        kcw.x wrote both the Hamiltonian and the band structure in the same
        run, so this discriminates every convention the file crosses on its
        way into the shared interpolator at once: R-vector ordering, energy
        units (eV, not Ry), the Wigner-Seitz phase convention, the Wannier
        centres' order, and the Monkhorst-Pack grid the Hamiltonian lives
        on. The controls below break each of those and watch the agreement
        go. kcw.x prints four decimals, so agreement below 1e-4 eV is
        agreement to the last digit it reports.
        """
        interpolated = _interpolate(si_kcw_reference, manifold)
        printed = np.array(si_kcw_reference["kcw_band_energies"])[:, MANIFOLD_BANDS[manifold]]

        assert interpolated.shape == printed.shape
        assert np.abs(interpolated - printed).max() < 1e-4

    def test_dropping_the_wigner_seitz_phase_does_not_give_those_eigenvalues(
        self, si_kcw_reference: dict
    ):
        """Negative control: the agreement above is not something any convention gives.

        On a 2x2x2 grid the Wigner-Seitz image selection is most of the
        interpolation, so interpolating the same file without it moves the
        bands by eV. That kcw.x's own numbers come back instead is what
        pins the phase convention shared with it.
        """
        without = _interpolate(si_kcw_reference, "occ", use_ws_distance=False)
        printed = np.array(si_kcw_reference["kcw_band_energies"])[:, MANIFOLD_BANDS["occ"]]

        assert np.abs(without - printed).max() > 0.1

    @pytest.mark.parametrize("manifold", ["occ", "emp"])
    def test_permuting_the_centres_does_not_give_those_eigenvalues(
        self, si_kcw_reference: dict, manifold: str
    ):
        """Negative control: the centres reach the phase in the Hamiltonian's band order.

        The centres enter only as differences, so a rigid shift of all of
        them cancels and says nothing. Permuting them is what pins the
        pairing with the Hamiltonian's rows, and it moves the bands by eV.
        """
        centres = np.roll(np.array(si_kcw_reference["centres"][manifold]), 1, axis=0)
        permuted = ui_helpers.unfold_and_interpolate(
            hr_content=(DATA_DIR / f"kcw_hr_{manifold}.dat").read_text(),
            centers=centres,
            cell=np.array(si_kcw_reference["cell"]),
            kgrid=tuple(si_kcw_reference["kgrid"]),
            kpath_kpts=np.array(si_kcw_reference["kpath_kpts"]),
        )
        printed = np.array(si_kcw_reference["kcw_band_energies"])[:, MANIFOLD_BANDS[manifold]]

        assert np.abs(permuted - printed).max() > 0.1

    def test_the_wrong_monkhorst_pack_grid_does_not_give_those_eigenvalues(
        self, si_kcw_reference: dict
    ):
        """Negative control: ``kgrid`` selects the R-vectors, and the wrong one shows.

        The grid is what turns the file's R-vectors into the primitive-cell
        set the Fourier sum runs over, so reading the same file on a 1x1x1
        grid keeps only the home cell and moves the bands by eV.
        """
        wrong = ui_helpers.unfold_and_interpolate(
            hr_content=(DATA_DIR / "kcw_hr_occ.dat").read_text(),
            centers=np.array(si_kcw_reference["centres"]["occ"]),
            cell=np.array(si_kcw_reference["cell"]),
            kgrid=(1, 1, 1),
            kpath_kpts=np.array(si_kcw_reference["kpath_kpts"]),
        )
        printed = np.array(si_kcw_reference["kcw_band_energies"])[:, MANIFOLD_BANDS["occ"]]

        assert np.abs(wrong - printed).max() > 0.1

    def test_the_two_manifolds_hamiltonians_are_not_interchangeable(self, si_kcw_reference: dict):
        """Negative control: the occupied and empty files are told apart by name alone.

        Silicon's occupied and empty Wannier functions sit on the same bond
        centres, so nothing else in this fixture would catch the two files
        being swapped; the eigenvalues differ by 15 eV.
        """
        swapped = ui_helpers.unfold_and_interpolate(
            hr_content=(DATA_DIR / "kcw_hr_emp.dat").read_text(),
            centers=np.array(si_kcw_reference["centres"]["occ"]),
            cell=np.array(si_kcw_reference["cell"]),
            kgrid=tuple(si_kcw_reference["kgrid"]),
            kpath_kpts=np.array(si_kcw_reference["kpath_kpts"]),
        )
        printed = np.array(si_kcw_reference["kcw_band_energies"])[:, MANIFOLD_BANDS["occ"]]

        assert np.abs(swapped - printed).max() > 1.0


class TestTheSmoothCorrectionReachesTheKcwHamiltonian:
    """The DFPT route's Hamiltonian takes the smooth correction like any other."""

    def test_a_denser_hamiltonian_changes_the_interpolated_bands(self, si_kcw_reference: dict):
        """Both DFT Hamiltonians together move the bands off kcw.x's own answer.

        Establishes that the correction is wired through for a kcw.x
        Hamiltonian at all — with the coarse term alone the helper leaves
        the bands untouched (``test_ui_smooth_helpers.py``), so a
        difference here can only come from the dense term.
        """
        coarse = (DATA_DIR / "kcw_hr_occ.dat").read_text()
        dense = (DATA_DIR / "kcw_hr_emp.dat").read_text()
        plain = _interpolate(si_kcw_reference, "occ")
        corrected = _interpolate(
            si_kcw_reference, "occ", dft_ham_content=coarse, dft_smooth_ham_content=dense
        )

        assert np.abs(corrected - plain).max() > 0.1

    def test_a_coarse_hamiltonian_equal_to_the_koopmans_one_leaves_the_dense_bands(
        self, si_kcw_reference: dict
    ):
        """``H_KI = H_coarse`` removes the Koopmans term, leaving the dense sum alone.

        The expected value is the weighted Fourier transform of the dense
        file written out directly, so this pins the dense term's own
        R-vectors and degeneracy weights as they are read out of a real
        file rather than a synthetic one.
        """
        coarse = (DATA_DIR / "kcw_hr_occ.dat").read_text()
        dense = (DATA_DIR / "kcw_hr_emp.dat").read_text()
        bands = _interpolate(
            si_kcw_reference, "occ", dft_ham_content=coarse, dft_smooth_ham_content=dense
        )

        hr_smooth, rvect, weights = ui_helpers.load_smooth_hr(dense, num_wann=4)
        phases = np.exp(2j * np.pi * np.array(si_kcw_reference["kpath_kpts"]) @ rvect.T)
        expected = np.linalg.eigvalsh(
            np.einsum("kr,rij->kij", phases, hr_smooth / weights[:, None, None])
        )

        assert np.allclose(bands, expected, atol=1e-10)


class TestDfptBandStructureTaskWiring:
    """The graph that runs the above inside a workflow."""

    @staticmethod
    def _build(*, emp: bool, smooth: bool = True):
        from aiida import orm

        from aiida_koopmans.workgraphs.ui.dfpt import DfptBandStructureTask
        from tests.fixtures import block_wannierization

        kpath = orm.KpointsData()
        kpath.set_kpoints([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]])
        structure = orm.StructureData(cell=[[0, 2.7, 2.7], [2.7, 0, 2.7], [2.7, 2.7, 0]])
        structure.append_atom(position=(0, 0, 0), symbols="Si")
        labels = ["occ"] + (["emp"] if emp else [])
        return DfptBandStructureTask.build(
            structure=structure.store(),
            koopmans_ham_retrieved=orm.FolderData().store(),
            block_wannierizations={label: block_wannierization(label) for label in labels},
            smooth_block_wannierizations={
                label: block_wannierization(f"{label}_smooth") for label in labels
            },
            occ_labels=["occ"],
            emp_labels=["emp"] if emp else None,
            kgrid=[2, 2, 2],
            kpath=kpath.store(),
        )

    def test_both_manifolds_are_interpolated_and_merged(self, aiida_profile):
        """Each manifold gets its own extraction and interpolation, then one merge."""
        wg = self._build(emp=True)
        names = [t.name for t in wg.tasks]

        for manifold in ("occ", "emp"):
            assert f"extract_{manifold}_hamiltonian" in names
            assert f"interpolate_{manifold}" in names
        assert names.count("merge_manifold_energies") == 1
        assert "build_band_structure" in names

    def test_both_dft_hamiltonians_reach_each_interpolation(self, aiida_profile):
        """The coarse and the dense wannierization both feed every manifold.

        Half the correction is not a correction: the helper ignores a
        coarse Hamiltonian arriving without a dense one, so a wiring that
        dropped one would silently interpolate the Koopmans Hamiltonian
        alone.
        """
        wg = self._build(emp=True)
        by_name = {t.name: t for t in wg.tasks}

        for manifold in ("occ", "emp"):
            inputs = by_name[f"interpolate_{manifold}"].inputs
            assert inputs["dft_ham_file"]._links
            assert inputs["dft_smooth_ham_file"]._links

    def test_an_occupied_only_run_merges_the_one_manifold(self, aiida_profile):
        """No empty projections means no empty interpolation, and still a band structure."""
        wg = self._build(emp=False)
        names = [t.name for t in wg.tasks]

        assert "interpolate_occ" in names
        assert "interpolate_emp" not in names
        assert not by_name_inputs(wg, "merge_manifold_energies", "empty")._links
        assert "build_band_structure" in names


def by_name_inputs(wg, task_name: str, socket: str):
    """Return one task's named input socket, by task name."""
    return {t.name: t for t in wg.tasks}[task_name].inputs[socket]
