"""Unit tests for the pure-Python Wannier90 file merge helpers.

Round-trips the parse / generate pairs and asserts the merge math against
hand-computed block-diagonal / concatenated / identity-extended results.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_koopmans.wannier_merge import (
    extend_wannier_u_dis_file_content,
    generate_wannier_centres_file_contents,
    generate_wannier_hr_file_contents,
    generate_wannier_u_file_contents,
    merge_wannier_centres_file_contents,
    merge_wannier_hr_file_contents,
    merge_wannier_u_file_contents,
    parse_wannier_centres_file_contents,
    parse_wannier_hr_file_contents,
    parse_wannier_u_file_contents,
    parse_wannier_u_file_shape,
)

RVECT = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
WEIGHTS = [1, 2, 2]
KPTS = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])

ATOM_LINES = [
    "Si       0.00000000      0.00000000      0.00000000",
    "Si       1.35750000      1.35750000      1.35750000",
]


def _random_complex(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape) + 1j * rng.random(shape)


# ----------------------------------------------------------------------
# parse / generate round-trips
# ----------------------------------------------------------------------


class TestRoundTrips:
    def test_hr(self):
        ham = _random_complex((3, 2, 2), seed=1)
        content = generate_wannier_hr_file_contents(ham, RVECT, WEIGHTS)
        ham_back, rvect_back, weights_back = parse_wannier_hr_file_contents(content)
        # The file stores 6 decimal places (up to sqrt(2)/2 ulp error in modulus).
        np.testing.assert_allclose(ham_back, ham, atol=1e-6)
        np.testing.assert_array_equal(rvect_back, RVECT)
        assert weights_back == WEIGHTS

    def test_hr_many_rpoints_weight_wrapping(self):
        """The degeneracy weights wrap at 15 per line."""
        nrpts = 17
        rvect = np.array([[i, 0, 0] for i in range(nrpts)])
        weights = list(range(1, nrpts + 1))
        ham = _random_complex((nrpts, 1, 1), seed=2)
        content = generate_wannier_hr_file_contents(ham, rvect, weights)
        _, rvect_back, weights_back = parse_wannier_hr_file_contents(content)
        assert weights_back == weights
        np.testing.assert_array_equal(rvect_back, rvect)

    def test_hr_unrecognized_header_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            parse_wannier_hr_file_contents("<?xml version>\nstuff\n")

    def test_u(self):
        umat = _random_complex((2, 3, 3), seed=3)
        content = generate_wannier_u_file_contents(umat, KPTS)
        umat_back, kpts_back = parse_wannier_u_file_contents(content)
        # The file stores 10 decimal places.
        np.testing.assert_allclose(umat_back, umat, atol=5e-11)
        np.testing.assert_allclose(kpts_back, KPTS)
        assert parse_wannier_u_file_shape(content) == (2, 3, 3)

    def test_u_rectangular(self):
        """u_dis matrices are rectangular (num_wann x num_bands)."""
        umat = _random_complex((2, 2, 5), seed=4)
        content = generate_wannier_u_file_contents(umat, KPTS)
        umat_back, _ = parse_wannier_u_file_contents(content)
        np.testing.assert_allclose(umat_back, umat, atol=5e-11)
        assert parse_wannier_u_file_shape(content) == (2, 2, 5)

    def test_centres(self):
        centres = [[0.25, 0.5, 0.75], [1.0, 2.0, 3.0]]
        content = generate_wannier_centres_file_contents(centres, ATOM_LINES)
        centres_back, atom_lines_back = parse_wannier_centres_file_contents(content)
        np.testing.assert_allclose(centres_back, centres)
        assert atom_lines_back == ATOM_LINES
        # xyz header: total entry count.
        assert content.split("\n")[0].strip() == "4"


# ----------------------------------------------------------------------
# merges (hand-computed expectations)
# ----------------------------------------------------------------------


class TestMergeHr:
    def test_block_diagonal(self):
        ham_a = _random_complex((3, 2, 2), seed=5)
        ham_b = _random_complex((3, 1, 1), seed=6)
        merged = merge_wannier_hr_file_contents(
            [
                generate_wannier_hr_file_contents(ham_a, RVECT, WEIGHTS),
                generate_wannier_hr_file_contents(ham_b, RVECT, WEIGHTS),
            ]
        )
        ham, rvect, weights = parse_wannier_hr_file_contents(merged)
        assert ham.shape == (3, 3, 3)
        np.testing.assert_array_equal(rvect, RVECT)
        assert weights == WEIGHTS
        np.testing.assert_allclose(ham[:, :2, :2], ham_a, atol=1e-6)
        np.testing.assert_allclose(ham[:, 2:, 2:], ham_b, atol=1e-6)
        # Off-diagonal blocks (couplings between different blocks) are zero.
        np.testing.assert_array_equal(ham[:, :2, 2:], 0)
        np.testing.assert_array_equal(ham[:, 2:, :2], 0)

    def test_differing_weights_raise(self):
        ham = _random_complex((3, 1, 1), seed=7)
        contents = [
            generate_wannier_hr_file_contents(ham, RVECT, WEIGHTS),
            generate_wannier_hr_file_contents(ham, RVECT, [1, 1, 1]),
        ]
        with pytest.raises(ValueError, match="differing weights"):
            merge_wannier_hr_file_contents(contents)

    def test_differing_rvectors_raise(self):
        """A single differing R-vector fires the consistency check."""
        ham = _random_complex((3, 1, 1), seed=8)
        rvect_other = RVECT.copy()
        rvect_other[2] = [0, 1, 0]
        contents = [
            generate_wannier_hr_file_contents(ham, RVECT, WEIGHTS),
            generate_wannier_hr_file_contents(ham, rvect_other, WEIGHTS),
        ]
        with pytest.raises(ValueError, match="R-vectors"):
            merge_wannier_hr_file_contents(contents)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No hr file contents"):
            merge_wannier_hr_file_contents([])


class TestMergeU:
    def test_block_diagonal(self):
        u_a = _random_complex((2, 2, 2), seed=9)
        u_b = _random_complex((2, 3, 3), seed=10)
        merged = merge_wannier_u_file_contents(
            [
                generate_wannier_u_file_contents(u_a, KPTS),
                generate_wannier_u_file_contents(u_b, KPTS),
            ]
        )
        umat, kpts = parse_wannier_u_file_contents(merged)
        assert umat.shape == (2, 5, 5)
        np.testing.assert_allclose(kpts, KPTS)
        np.testing.assert_allclose(umat[:, :2, :2], u_a, atol=5e-11)
        np.testing.assert_allclose(umat[:, 2:, 2:], u_b, atol=5e-11)
        np.testing.assert_array_equal(umat[:, :2, 2:], 0)
        np.testing.assert_array_equal(umat[:, 2:, :2], 0)

    def test_differing_kpoints_raise(self):
        umat = _random_complex((2, 1, 1), seed=11)
        other_kpts = KPTS + 0.25
        contents = [
            generate_wannier_u_file_contents(umat, KPTS),
            generate_wannier_u_file_contents(umat, other_kpts),
        ]
        with pytest.raises(ValueError, match="k-points"):
            merge_wannier_u_file_contents(contents)


class TestMergeCentres:
    def test_concatenation_keeps_atoms_once(self):
        centres_a = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
        centres_b = [[2.0, 2.0, 2.0]]
        merged = merge_wannier_centres_file_contents(
            [
                generate_wannier_centres_file_contents(centres_a, ATOM_LINES),
                generate_wannier_centres_file_contents(centres_b, ATOM_LINES),
            ]
        )
        centres, atom_lines = parse_wannier_centres_file_contents(merged)
        np.testing.assert_allclose(centres, centres_a + centres_b)
        assert atom_lines == ATOM_LINES

    def test_differing_atoms_raise(self):
        contents = [
            generate_wannier_centres_file_contents([[0.0, 0.0, 0.0]], ATOM_LINES),
            generate_wannier_centres_file_contents([[0.0, 0.0, 0.0]], ATOM_LINES[:1]),
        ]
        with pytest.raises(ValueError, match="atomic entries"):
            merge_wannier_centres_file_contents(contents)


class TestExtendUDis:
    def test_identity_plus_bottom_right_block(self):
        """The manifold-wide u_dis: identity for earlier blocks, u_dis last.

        Manifold: 2 + 2 Wannier functions over 6 bands; only the last block
        is disentangled, with a 2 x 4 u_dis (its 2 WFs over its 2 + 2 extra
        bands). The merged 4 x 6 matrix maps bands 1-2 identically onto WFs
        1-2 and applies the last block's u_dis to bands 3-6 / WFs 3-4.
        """
        udis_last = _random_complex((2, 2, 4), seed=12)
        extended = extend_wannier_u_dis_file_content(
            generate_wannier_u_file_contents(udis_last, KPTS), nbnd=6, nwann=4
        )
        umat, kpts = parse_wannier_u_file_contents(extended)
        assert umat.shape == (2, 4, 6)
        np.testing.assert_allclose(kpts, KPTS)
        expected = np.zeros((2, 4, 6), dtype=complex)
        expected[:, :4, :4] = np.identity(4)
        expected[:, 2:, 2:] = udis_last
        np.testing.assert_allclose(umat, expected, atol=5e-11)

    def test_single_block_extension_is_identity_free(self):
        """When the block spans the whole manifold, extension reproduces it."""
        udis = _random_complex((2, 3, 5), seed=13)
        extended = extend_wannier_u_dis_file_content(
            generate_wannier_u_file_contents(udis, KPTS), nbnd=5, nwann=3
        )
        umat, _ = parse_wannier_u_file_contents(extended)
        np.testing.assert_allclose(umat, udis, atol=5e-11)
