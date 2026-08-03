"""Tests for the pure-python unfold-and-interpolate helpers in ``workgraphs/ui/helpers.py``.

The silicon data in ``tests/data/ui/`` comes from the reference (ASE-based)
``koopmans`` package's test suite; ``si_ui_reference.json`` holds the band
energies and DOS its unfold-and-interpolate implementation produces on
those files. The implementations agree bit-for-bit, so the comparisons use
tight tolerances.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiida_koopmans.workgraphs.ui import helpers as ui_helpers

DATA_DIR = Path(__file__).parent / "data" / "ui"


@pytest.fixture(scope="module")
def si_wout_content() -> str:
    """Return the silicon Wannier90 ``.wout`` contents."""
    return (DATA_DIR / "wann.wout").read_text()


# ----------------------------------------------------------------------
# Small unit tests
# ----------------------------------------------------------------------


class TestLatticeUtilities:
    """Lattice-vector generation and crystal/cartesian conversion."""

    def test_latt_vect_enumerates_grid(self):
        """R-vectors enumerate the grid in row-major order."""
        rvec = ui_helpers.latt_vect(2, 1, 2)
        assert rvec.tolist() == [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]

    def test_crys_to_cart_roundtrip(self):
        """typ=+1 against the direct cell and typ=-1 against its reciprocal invert each other."""
        rng = np.random.default_rng(0)
        cell = np.eye(3) + 0.1 * rng.random((3, 3))
        vec = rng.random((5, 3))
        cart = ui_helpers.crys_to_cart(vec, cell, +1)
        back = ui_helpers.crys_to_cart(cart, ui_helpers.reciprocal_cell(cell), -1)
        assert np.allclose(back, vec)

    def test_crys_to_cart_rejects_bad_typ(self):
        """Any typ other than ±1 is an error."""
        with pytest.raises(ValueError, match="must be either"):
            ui_helpers.crys_to_cart(np.zeros(3), np.eye(3), 0)


class TestParsers:
    """Wannier90-format file parsers."""

    def test_parse_hr_gamma_only(self):
        """A single-R (kcp.x-style) Hamiltonian parses with a trivial R-vector."""
        content = (DATA_DIR / "kc_ham.dat").read_text()
        hr, rvect, _weights, nrpts = ui_helpers.parse_hr_file_contents(content)
        assert nrpts == 1
        assert rvect.tolist() == [[0, 0, 0]]
        assert len(hr) == 32**2

    def test_parse_hr_with_kpoints(self):
        """A k-point Hamiltonian exposes its Wigner-Seitz R-vectors and weights."""
        content = (DATA_DIR / "dft_ham.dat").read_text()
        hr, rvect, weights, nrpts = ui_helpers.parse_hr_file_contents(content)
        assert nrpts == 19
        assert rvect.shape == (19, 3)
        assert len(weights) == 19
        assert len(hr) == 19 * 4**2

    def test_parse_hr_rejects_unknown_format(self):
        """Contents without the Wannier90 header are rejected."""
        with pytest.raises(ValueError, match="not recognized"):
            ui_helpers.parse_hr_file_contents("garbage\n1\n1\n")

    def test_parse_hr_rejects_legacy_xml_format(self):
        """The old xml-based Hamiltonian file format is explicitly unsupported."""
        with pytest.raises(ValueError, match="no longer supported"):
            ui_helpers.parse_hr_file_contents('<?xml version="1.0"?>\n1\n1\n')

    def test_parse_wout(self, si_wout_content, si_reference):
        """Centres match the reference parse of the same file."""
        centers = ui_helpers.parse_wout_centers(si_wout_content)
        assert np.allclose(centers, si_reference["centers"])

    def test_parse_wout_without_final_state_raises(self):
        """A truncated .wout without a Final State block is an error."""
        with pytest.raises(ValueError, match="Final State"):
            ui_helpers.parse_wout_centers("no final state here\n")


class TestInferWannierCounts:
    """num_wann / num_wann_sc inference from the centre count."""

    def test_primitive_cell_input(self):
        """The centres describe one primitive cell."""
        assert ui_helpers.infer_wannier_counts(4, (2, 2, 2)) == (4, 32)


class TestExtractHr:
    """Restriction of a Wannier90 Hamiltonian onto the primitive-cell R-vectors."""

    def test_rejects_a_supercell_missing_a_required_r_vector(self):
        """A grid whose R-vectors don't cover every primitive-cell translation is refused."""
        hr = np.zeros((1, 1, 1), dtype=complex)
        rvect = np.array([[1, 0, 0]])  # never satisfies the R=0-cell fold test
        with pytest.raises(ValueError, match="Wrong number"):
            ui_helpers.extract_hr(hr, rvect, 1, 1, 1)


class TestLoaders:
    """Element-count guards in the Hamiltonian loaders."""

    def test_load_primary_hr_kpoint_wrong_wann_count_raises(self):
        content = (DATA_DIR / "dft_ham.dat").read_text()
        with pytest.raises(ValueError, match="Wrong number of matrix elements"):
            ui_helpers.load_primary_hr(content, num_wann=3, num_wann_sc=32, kgrid=(2, 2, 2))

    def test_load_coarse_hr_gamma_only_raises(self):
        """A Gamma-only coarse Hamiltonian (supercell DFT run) is rejected."""
        content = (DATA_DIR / "kc_ham.dat").read_text()
        with pytest.raises(ValueError, match="Wrong number of matrix elements for hr_coarse"):
            ui_helpers.load_coarse_hr(content, num_wann=4, num_wann_sc=32, kgrid=(2, 2, 2))

    def test_load_coarse_hr_kpoint_wrong_count_raises(self):
        content = (DATA_DIR / "dft_ham.dat").read_text()
        with pytest.raises(ValueError, match="Wrong number of matrix elements for hr_coarse"):
            ui_helpers.load_coarse_hr(content, num_wann=3, num_wann_sc=32, kgrid=(2, 2, 2))

    def test_load_smooth_hr_wrong_count_raises(self):
        content = (DATA_DIR / "smooth_dft_ham.dat").read_text()
        with pytest.raises(ValueError, match="Wrong number of matrix elements for hr_smooth"):
            ui_helpers.load_smooth_hr(content, num_wann=3)


class TestCalcBands:
    """The Fourier-transform core, exercised directly with a trivial 1-WF cell."""

    def _minimal_kwargs(self):
        return {
            "hr": np.array([[1.0 + 0.0j]]),
            "centers": np.zeros((1, 3)),
            "kpts": np.zeros((1, 3)),
            "rvec": np.array([[0, 0, 0]]),
            "kgrid": (1, 1, 1),
            "acell": np.eye(3),
            "num_wann": 1,
            "num_wann_sc": 1,
            "use_ws_distance": False,
        }

    def test_hr_smooth_without_rvectors_raises(self):
        kwargs = self._minimal_kwargs()
        kwargs["hr_smooth"] = np.array([[[1.0 + 0.0j]]])
        with pytest.raises(ValueError, match="hr_smooth requires"):
            ui_helpers.calc_bands(**kwargs)


class TestComputeDos:
    """Gaussian-smearing DOS."""

    def test_single_gaussian_normalisation(self):
        """One eigenvalue in one spin channel integrates to 2 (spin degeneracy)."""
        grid, dos = ui_helpers.compute_dos(
            np.array([[[0.0]]]), width=0.1, emin=-2.0, emax=2.0, npts=4001
        )
        assert np.isclose(np.trapezoid(dos, grid), 2.0, atol=1e-6)

    def test_two_spin_channels_are_summed(self):
        """With two spin channels the total DOS is the plain sum (no doubling)."""
        e = np.array([[[0.0]], [[0.0]]])
        grid, dos = ui_helpers.compute_dos(e, width=0.1, emin=-2.0, emax=2.0, npts=4001)
        assert np.isclose(np.trapezoid(dos, grid), 2.0, atol=1e-6)

    def test_requires_spin_axis(self):
        """A (k, n)-shaped array without the spin axis is rejected."""
        with pytest.raises(ValueError, match="spin"):
            ui_helpers.compute_dos(np.zeros((3, 4)))

    def test_energy_window_defaults_to_the_data_range(self):
        """Without explicit emin/emax the window is derived from the eigenvalues."""
        grid, _dos = ui_helpers.compute_dos(np.array([[[0.0, 1.0]]]), width=0.1, npts=101)
        assert grid[0] == pytest.approx(-0.5)
        assert grid[-1] == pytest.approx(1.5)


# ----------------------------------------------------------------------
# Silicon regression against the reference data
# ----------------------------------------------------------------------


class TestSiliconRegression:
    """Reproduce the reference-implementation numbers exactly."""

    def test_smooth_interpolation(self, si_reference, si_wout_content):
        """Full path: Γ-only KI Hamiltonian, smooth interpolation."""
        centers = ui_helpers.parse_wout_centers(si_wout_content)
        energies = ui_helpers.unfold_and_interpolate(
            hr_content=(DATA_DIR / "kc_ham.dat").read_text(),
            centers=centers,
            cell=np.array(si_reference["cell"]),
            kgrid=tuple(si_reference["kgrid"]),
            kpath_kpts=np.array(si_reference["kpath_kpts"]),
            use_ws_distance=True,
            dft_ham_content=(DATA_DIR / "dft_ham.dat").read_text(),
            dft_smooth_ham_content=(DATA_DIR / "smooth_dft_ham.dat").read_text(),
        )
        assert np.allclose(energies, si_reference["energies"], atol=1e-10)

    @pytest.mark.parametrize("use_ws_distance", [True, False], ids=["ws", "nows"])
    def test_plain_interpolation(self, si_reference, si_wout_content, use_ws_distance):
        """Plain path: k-point DFT Hamiltonian, no mapping, no smoothing."""
        centers = ui_helpers.parse_wout_centers(si_wout_content)
        energies = ui_helpers.unfold_and_interpolate(
            hr_content=(DATA_DIR / "dft_ham.dat").read_text(),
            centers=centers,
            cell=np.array(si_reference["cell"]),
            kgrid=tuple(si_reference["kgrid"]),
            kpath_kpts=np.array(si_reference["kpath_kpts"]),
            use_ws_distance=use_ws_distance,
        )
        key = "plain_ws_energies" if use_ws_distance else "plain_nows_energies"
        assert np.allclose(energies, si_reference[key], atol=1e-10)

    def test_dos_matches_reference(self, si_reference, si_wout_content):
        """The DOS of the smooth-interpolated bands matches the reference."""
        centers = ui_helpers.parse_wout_centers(si_wout_content)
        energies = ui_helpers.unfold_and_interpolate(
            hr_content=(DATA_DIR / "kc_ham.dat").read_text(),
            centers=centers,
            cell=np.array(si_reference["cell"]),
            kgrid=tuple(si_reference["kgrid"]),
            kpath_kpts=np.array(si_reference["kpath_kpts"]),
            use_ws_distance=True,
            dft_ham_content=(DATA_DIR / "dft_ham.dat").read_text(),
            dft_smooth_ham_content=(DATA_DIR / "smooth_dft_ham.dat").read_text(),
        )
        grid, dos = ui_helpers.compute_dos(
            energies[np.newaxis], width=0.05, emin=-10, emax=4, npts=1001
        )
        assert np.allclose(grid, si_reference["dos_energies"], atol=1e-10)
        assert np.allclose(dos, si_reference["dos_values"], atol=1e-8)

    def test_smooth_requires_coarse_hamiltonian(self, si_reference, si_wout_content):
        """Supplying only the smooth Hamiltonian is an error."""
        centers = ui_helpers.parse_wout_centers(si_wout_content)
        with pytest.raises(ValueError, match="coarse DFT Hamiltonian"):
            ui_helpers.unfold_and_interpolate(
                hr_content=(DATA_DIR / "kc_ham.dat").read_text(),
                centers=centers,
                cell=np.array(si_reference["cell"]),
                kgrid=tuple(si_reference["kgrid"]),
                kpath_kpts=np.array(si_reference["kpath_kpts"]),
                dft_smooth_ham_content=(DATA_DIR / "smooth_dft_ham.dat").read_text(),
            )

    def test_wrong_grid_raises(self, si_reference, si_wout_content):
        """A k-grid inconsistent with the Hamiltonian size fails the element-count check."""
        centers = ui_helpers.parse_wout_centers(si_wout_content)
        with pytest.raises(ValueError, match="Wrong number of matrix elements"):
            ui_helpers.unfold_and_interpolate(
                hr_content=(DATA_DIR / "kc_ham.dat").read_text(),
                centers=centers,
                cell=np.array(si_reference["cell"]),
                kgrid=(2, 2, 1),
                kpath_kpts=np.array(si_reference["kpath_kpts"]),
            )
