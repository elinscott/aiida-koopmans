"""Unit tests for the pure-Python Wannier90 file merge helpers.

Round-trips the parse / generate pairs and asserts the merge math against
hand-computed block-diagonal / concatenated / identity-extended results.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiida_koopmans.workgraphs.utils.wannier_merge import (
    extend_wannier_u_dis_file_content,
    generate_wannier_centres_file_contents,
    generate_wannier_hr_file_contents,
    generate_wannier_u_file_contents,
    merge_wannier_centres_file_contents,
    merge_wannier_hr_file_contents,
    merge_wannier_split_u_dis_file_contents,
    merge_wannier_u_file_contents,
    parse_wannier_amn_file_contents,
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


class TestGenerateHr:
    def test_rejects_a_ham_shape_inconsistent_with_rvect_and_weights(self):
        ham = _random_complex((3, 2, 2), seed=12)  # nrpts=3 in ham, but only 2 weights given
        with pytest.raises(ValueError, match="expected"):
            generate_wannier_hr_file_contents(ham, RVECT[:2], WEIGHTS[:2])


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

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No U matrix file contents"):
            merge_wannier_u_file_contents([])


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

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No centres file contents"):
            merge_wannier_centres_file_contents([])

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


# ---------------------------------------------------------------------------
# Split-manifold disentanglement matrix
# ---------------------------------------------------------------------------


def _amn_file_contents(mat: np.ndarray) -> str:
    """Write a ``(nkpts, num_bands, num_wann)`` matrix as a Wannier90 ``.amn``."""
    nk, nbands, nwann = mat.shape
    lines = ["synthetic split rotation", f"{nbands:12d}{nk:12d}{nwann:12d}"]
    for ik in range(nk):
        for iw in range(nwann):
            for ib in range(nbands):
                value = mat[ik, ib, iw]
                lines.append(
                    f"{ib + 1:5d}{iw + 1:5d}{ik + 1:5d}{value.real:18.12f}{value.imag:18.12f}"
                )
    return "\n".join(lines) + "\n"


def _random_unitary(rng, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _staged_gauge(u_dis: np.ndarray, u_block: np.ndarray) -> np.ndarray:
    """Rebuild the bands-to-Wannier gauge kcw.x forms from the staged pair.

    Both files are stored as ``(nkpts, num_wann, num_bands)``; the
    disentanglement matrix maps bands onto the manifold and the block
    gauge rotates within it.
    """
    bands_to_manifold = u_dis.transpose(0, 2, 1)
    within_manifold = u_block.conj().transpose(0, 2, 1)
    return np.einsum("kbn,knm->kbm", bands_to_manifold, within_manifold)


class TestSplitUDis:
    """A split manifold's gauge must still rebuild its own Hamiltonian.

    Synthesizes a parent manifold whose bands are split into groups, runs
    the products through the writers this module ships, and asks whether
    the staged pair (``_u_dis.mat`` from the split rotations, block-diagonal
    ``_u.mat`` from the groups) reproduces the merged Hamiltonian. The
    variants that drop or scramble the split rotation must not.
    """

    NBANDS, GROUPS, NK = 6, (2, 4), 3

    def _fixture(self, seed: int = 7):
        rng = np.random.default_rng(seed)
        nk, nbands = self.NK, self.NBANDS
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.5]])[:nk]
        eps = np.sort(rng.normal(scale=5.0, size=(nk, nbands)), axis=1)
        # The split rotation carries the parent's gauge but does not mix
        # bands across the groups, which are separated in energy: it is
        # unitary within each group's own band range. Anything else would
        # leave the merged Hamiltonian non-block-diagonal, which is the
        # property the split exists to produce.
        split = np.zeros((nk, nbands, nbands), dtype=complex)
        off = 0
        for width in self.GROUPS:
            block = np.stack([_random_unitary(rng, width) for _ in range(nk)])
            split[:, off : off + width, off : off + width] = block
            off += width
        gauges = [np.stack([_random_unitary(rng, n) for _ in range(nk)]) for n in self.GROUPS]
        return kpts, eps, split, gauges

    @staticmethod
    def _merged_hamiltonian(eps, split, gauges, groups):
        """H in the final split basis, group by group, block-diagonal."""
        nk = eps.shape[0]
        total = sum(groups)
        merged = np.zeros((nk, total, total), dtype=complex)
        off = 0
        for width, gauge in zip(groups, gauges, strict=True):
            columns = split[:, :, off : off + width]
            rotated = np.einsum("kbn,knm->kbm", columns, gauge.conj().transpose(0, 2, 1))
            block = np.einsum("kbm,kb,kbn->kmn", rotated.conj(), eps.astype(complex), rotated)
            merged[:, off : off + width, off : off + width] = block
            off += width
        return merged

    def test_split_rotation_reproduces_the_merged_hamiltonian(self):
        kpts, eps, split, gauges = self._fixture()
        target = self._merged_hamiltonian(eps, split, gauges, self.GROUPS)

        off = 0
        contents = []
        for width in self.GROUPS:
            contents.append(_amn_file_contents(split[:, :, off : off + width]))
            off += width
        u_dis, _ = parse_wannier_u_file_contents(
            merge_wannier_split_u_dis_file_contents(contents, kpts)
        )
        u_block, _ = parse_wannier_u_file_contents(
            merge_wannier_u_file_contents(
                [generate_wannier_u_file_contents(g, kpts) for g in gauges]
            )
        )
        # The staged pair, read back through this module's own parser.
        gauge = _staged_gauge(u_dis, u_block)
        rebuilt = np.einsum("kbm,kb,kbn->kmn", gauge.conj(), eps.astype(complex), gauge)
        np.testing.assert_allclose(rebuilt, target, atol=1e-9)

    def test_dropping_the_split_rotation_fails(self):
        """The block-diagonal gauge alone is not the manifold's gauge."""
        _, eps, split, gauges = self._fixture()
        target = self._merged_hamiltonian(eps, split, gauges, self.GROUPS)
        nk, total = eps.shape[0], sum(self.GROUPS)
        blockdiag = np.zeros((nk, total, total), dtype=complex)
        off = 0
        for width, gauge in zip(self.GROUPS, gauges, strict=True):
            blockdiag[:, off : off + width, off : off + width] = gauge
            off += width
        rebuilt = np.einsum("kbm,kb,kbn->kmn", blockdiag.conj(), eps.astype(complex), blockdiag)
        assert np.abs(rebuilt - target).max() > 1e-3

    def test_mis_ordered_groups_fail(self):
        """Concatenating the split rotations out of band order is detected."""
        kpts, eps, split, gauges = self._fixture()
        target = self._merged_hamiltonian(eps, split, gauges, self.GROUPS)
        off = 0
        columns = []
        for width in self.GROUPS:
            columns.append(split[:, :, off : off + width])
            off += width
        u_dis, _ = parse_wannier_u_file_contents(
            merge_wannier_split_u_dis_file_contents(
                [_amn_file_contents(c) for c in reversed(columns)], kpts
            )
        )
        u_block, _ = parse_wannier_u_file_contents(
            merge_wannier_u_file_contents(
                [generate_wannier_u_file_contents(g, kpts) for g in gauges]
            )
        )
        gauge = _staged_gauge(u_dis, u_block)
        rebuilt = np.einsum("kbm,kb,kbn->kmn", gauge.conj(), eps.astype(complex), gauge)
        assert np.abs(rebuilt - target).max() > 1e-3

    def test_rectangular_split_rotations_round_trip(self):
        """A split rotation is rectangular; the square check must be off."""
        kpts, _, split, _ = self._fixture()
        contents = [_amn_file_contents(split[:, :, :2]), _amn_file_contents(split[:, :, 2:])]
        merged, _ = parse_wannier_u_file_contents(
            merge_wannier_split_u_dis_file_contents(contents, kpts)
        )
        assert merged.shape == (self.NK, self.NBANDS, self.NBANDS)
        with pytest.raises(ValueError, match="not square"):
            parse_wannier_amn_file_contents(contents[0])
