"""Analytic tests of the smooth-interpolation correction in ``workgraphs/ui/helpers.py``.

``tests/test_ui_helpers.py`` pins the same code against the reference
(ASE-based) ``koopmans`` implementation's numbers on real silicon data:
that says the two agree, not what they compute. These tests take the
formula apart instead, on a synthetic two-Wannier-function cell whose every
term is known in closed form::

    H(k) = Σ_R φ(k,R)·φ_corr(k,R)·[H_KI(R) - H_c(R)]
           + Σ_Rs e^{2πik·Rs} H_s(Rs) / w_Rs

so a sign, a transpose or a missing degeneracy weight fails one of them
without any reference file to compare against.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiida_koopmans.workgraphs.ui import helpers as ui_helpers

DATA_DIR = Path(__file__).parent / "data" / "ui"


class SmoothFixture:
    """A synthetic cell small enough to write the whole Fourier sum out by hand.

    Two Wannier functions in a 2x1x1 supercell, so ``H_KI`` and ``H_c``
    live on two R-vectors and the dense Hamiltonian on its own
    Wigner-Seitz set with non-unit degeneracy weights.
    """

    num_wann = 2
    kgrid = (2, 1, 1)

    def __init__(self, seed: int = 3) -> None:
        rng = np.random.default_rng(seed)
        self.num_wann_sc = self.num_wann * int(np.prod(self.kgrid))
        self.rvec = ui_helpers.latt_vect(*self.kgrid)
        self.acell = np.eye(3)
        self.centers = np.concatenate(
            [rng.random((self.num_wann, 3)) * 0.3 + rvect for rvect in self.rvec]
        )
        self.kpts = np.array([[0.0, 0.0, 0.0], [0.13, 0.0, 0.0], [0.5, 0.0, 0.0]])

        def _matrix(*shape: int) -> np.ndarray:
            return rng.random(shape) + 1j * rng.random(shape) - (0.5 + 0.5j)

        self.hr = _matrix(self.num_wann_sc, self.num_wann)
        self.hr_coarse = _matrix(self.num_wann_sc, self.num_wann)
        # A Wigner-Seitz set with a degenerate R-vector, so a dropped
        # weight cannot pass unnoticed.
        self.rvect_smooth = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [2, 0, 0]])
        self.weights_smooth = np.array([2, 2, 2, 4])
        self.hr_smooth = _matrix(len(self.rvect_smooth), self.num_wann, self.num_wann)

    @property
    def geometry(self) -> dict:
        """The positional arguments every :func:`calc_bands` call here shares."""
        return {
            "centers": self.centers,
            "kpts": self.kpts,
            "rvec": self.rvec,
            "kgrid": self.kgrid,
            "acell": self.acell,
            "num_wann": self.num_wann,
            "num_wann_sc": self.num_wann_sc,
            "use_ws_distance": True,
        }

    @property
    def smooth(self) -> dict:
        """The dense-Hamiltonian arguments of :func:`calc_bands`."""
        return {
            "hr_smooth": self.hr_smooth,
            "rvect_smooth": self.rvect_smooth,
            "weights_smooth": self.weights_smooth,
        }

    def bands(self, hr: np.ndarray, **overrides) -> np.ndarray:
        """Interpolate ``hr`` with the smooth correction on."""
        kwargs = {**self.geometry, "hr_coarse": self.hr_coarse, **self.smooth, **overrides}
        return ui_helpers.calc_bands(hr, **kwargs)


@pytest.fixture
def cell() -> SmoothFixture:
    """Return the synthetic two-Wannier-function cell."""
    return SmoothFixture()


class TestTheCorrectionReplacesTheDftHamiltonian:
    """What the method claims: the coarse DFT part out, the dense one in."""

    def test_a_koopmans_hamiltonian_equal_to_the_coarse_one_leaves_only_the_dense_sum(
        self, cell: SmoothFixture
    ):
        """``H_KI = H_c`` cancels the first term, so the bands are the dense FT alone.

        The expected value is written from the formula, not from the
        helper: a weighted Fourier transform of ``H_s`` in three lines.
        """
        bands = cell.bands(cell.hr_coarse)

        phases = np.exp(2j * np.pi * cell.kpts @ cell.rvect_smooth.T)
        weighted = cell.hr_smooth / cell.weights_smooth[:, None, None]
        expected = np.linalg.eigvalsh(np.einsum("kr,rij->kij", phases, weighted))

        assert np.allclose(bands, expected, atol=1e-12)

    def test_a_constant_added_at_the_home_cell_shifts_every_band_by_its_weighted_value(
        self, cell: SmoothFixture
    ):
        """``c·I`` at ``R_s = 0`` enters every k-point undamped, divided by ``w_0``.

        The degeneracy weight is the discriminating part: an
        implementation that forgot to divide by it shifts the bands by
        ``c`` instead.
        """
        reference = cell.bands(cell.hr)

        shift = 0.75
        home = np.argmax(np.all(cell.rvect_smooth == 0, axis=1))
        shifted_smooth = cell.hr_smooth.copy()
        shifted_smooth[home] += shift * np.eye(cell.num_wann)
        bands = cell.bands(cell.hr, hr_smooth=shifted_smooth)

        assert np.allclose(bands, reference + shift / cell.weights_smooth[home], atol=1e-12)

    def test_a_vanishing_dense_hamiltonian_interpolates_the_difference_alone(
        self, cell: SmoothFixture
    ):
        """``H_s ≡ 0`` leaves the Wigner-Seitz-corrected interpolation of ``H_KI - H_c``.

        Discriminates the real-space subtraction from the k-space
        addition: the same difference interpolated with no smooth
        machinery at all must give the same bands.
        """
        bands = cell.bands(cell.hr, hr_smooth=np.zeros_like(cell.hr_smooth))
        expected = ui_helpers.calc_bands(cell.hr - cell.hr_coarse, **cell.geometry)

        assert np.allclose(bands, expected, atol=1e-12)


class TestTheCoarseHamiltonianNeverActsAlone:
    """Both DFT Hamiltonians, or neither: half the correction is not a correction."""

    @staticmethod
    def _interpolate(si_reference, centers, **dft):
        return ui_helpers.unfold_and_interpolate(
            hr_content=(DATA_DIR / "kc_ham.dat").read_text(),
            centers=centers,
            cell=np.array(si_reference["cell"]),
            kgrid=tuple(si_reference["kgrid"]),
            kpath_kpts=np.array(si_reference["kpath_kpts"]),
            **dft,
        )

    def test_a_coarse_hamiltonian_without_a_dense_one_changes_nothing(self, si_reference):
        """Negative control: the coarse Hamiltonian alone is inert, not subtracted.

        Subtracting it without adding the dense one back would move every
        band, and the reference numbers ``test_ui_helpers`` pins would
        still pass — they never exercise this combination.
        """
        centers = ui_helpers.parse_wout_centers((DATA_DIR / "wann.wout").read_text())
        with_coarse = self._interpolate(
            si_reference, centers, dft_ham_content=(DATA_DIR / "dft_ham.dat").read_text()
        )
        without = self._interpolate(si_reference, centers)

        assert np.array_equal(with_coarse, without)
