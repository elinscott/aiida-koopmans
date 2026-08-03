"""Pure-Python unfolding-and-interpolation helpers.

Implements the unfolding-and-interpolation method of De Gennaro, Colonna,
Linscott and Marzari, Phys. Rev. B 106, 035106 (2022): a supercell Wannier
Hamiltonian is mapped onto primitive-cell R-vectors and its bands are
interpolated along a k-path by Fourier transform, optionally with the
smooth-interpolation correction from a denser-grid DFT Hamiltonian.
Every function takes and returns plain Python / numpy data, so the tasks
in :mod:`aiida_koopmans.workgraphs.ui` stay thin wrappers.
"""

from __future__ import annotations

import re
from math import pi, sqrt
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

# ----------------------------------------------------------------------
# Lattice / coordinate utilities
# ----------------------------------------------------------------------


def latt_vect(nr1: int, nr2: int, nr3: int) -> NDArray[np.int_]:
    """Generate lattice vectors {R} of the primitive cell commensurate to the supercell.

    The R-vectors are given in crystal units.
    """
    return np.array([[i, j, k] for i in range(nr1) for j in range(nr2) for k in range(nr3)])


def crys_to_cart(
    vec: NDArray[np.float64], trmat: NDArray[np.float64], typ: int
) -> NDArray[np.float64]:
    """Transform ``vec`` from crystal to cartesian (in alat units), or vice versa.

    Follows the transformation as done in Quantum ESPRESSO:
    ``typ=+1`` for crystal-to-cartesian, ``typ=-1`` for cartesian-to-crystal.
    For a real-space vector ``trmat`` must be ``at`` if ``typ=+1`` and
    ``bvec`` if ``typ=-1``; for a k-space vector, ``bg`` if ``typ=+1`` and
    ``avec`` if ``typ=-1``.
    """
    if typ == +1:
        return np.dot(vec, trmat)
    if typ == -1:
        return np.dot(vec, np.transpose(trmat))
    raise ValueError(f"`typ = {typ}` in `crys_to_cart` must be either +1 or -1")


def reciprocal_cell(cell: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the reciprocal lattice of ``cell`` without the 2π factor.

    Rows are the reciprocal lattice vectors with the 2π factor omitted,
    i.e. the ``bvec`` transformation matrix for :func:`crys_to_cart`.
    """
    return np.linalg.inv(cell).transpose().astype(np.float64, copy=False)


def extract_hr(
    hr: NDArray[np.complex128], rvect: NDArray[np.int_], nr1: int, nr2: int, nr3: int
) -> NDArray[np.complex128]:
    """Select the Wannier Hamiltonian only on the primitive-cell R-vectors.

    The Hamiltonian coming from a Wannier90 calculation with k-points is
    defined on the Wigner-Seitz lattice vectors; matrix elements on
    R-vectors exceeding the boundaries of the original supercell are
    ignored.
    """
    rvec = latt_vect(nr1, nr2, nr3)
    rgrid = [nr1, nr2, nr3]
    hr_new: list[NDArray[np.complex128]] = []

    for r in rvec:
        for ir, rvec_i in enumerate(rvect):
            if all(x < 1 for x in rvec_i / rgrid):
                folded = rvec_i % rgrid
                if all(folded == r):
                    hr_new.append(hr[ir, :, :])
                    break

    if len(hr_new) != np.prod(rgrid):
        raise ValueError(f"Wrong number ({len(hr_new)}) of R-vectors in `extract_hr`")

    return np.array(hr_new, dtype=np.complex128)


# ----------------------------------------------------------------------
# File parsers
# ----------------------------------------------------------------------


class HrFileContents(NamedTuple):
    """Parsed contents of a Wannier90-format ``*_hr.dat`` Hamiltonian file.

    ``hr`` is the flat list of matrix elements (not reshaped, because
    different Hamiltonians are reshaped differently downstream).
    """

    hr: NDArray[np.complex128]
    rvect: NDArray[np.int_]
    weights: list[int]
    nrpts: int


def parse_hr_file_contents(content: str) -> HrFileContents:
    """Parse the contents of a Wannier90-format Hamiltonian file.

    kcp.x Hamiltonian files use the same format (with a single
    R-vector).
    """
    lines = content.rstrip("\n").split("\n")
    if "written on" in lines[0].lower():
        pass
    elif "xml version" in lines[0]:
        raise ValueError("The format of Hamiltonian file contents is no longer supported")
    else:
        raise ValueError("The format of the Hamiltonian file contents is not recognized")

    nrpts = int(lines[2].split()[0])
    single_r = nrpts == 1

    num_wann = 0 if single_r else int(lines[1].split()[0])

    lines_to_skip = 3 + nrpts // 15
    if nrpts % 15 > 0:
        lines_to_skip += 1

    weights = [int(x) for line in lines[3:lines_to_skip] for x in line.split()]

    hr: list[complex] = []
    rvect: list[list[int]] = []
    for i, line in enumerate(lines[lines_to_skip:]):
        parts = line.split()
        hr.append(float(parts[5]) + 1j * float(parts[6]))
        if not single_r and i % num_wann**2 == 0:
            rvect.append([int(x) for x in parts[0:3]])

    rvect_np = np.array([[0, 0, 0]]) if single_r else np.array(rvect, dtype=int)

    return HrFileContents(np.array(hr, dtype=complex), rvect_np, weights, nrpts)


def parse_wout_centers(content: str) -> NDArray[np.float64]:
    """Extract the final-state Wannier centres from a ``.wout`` file, in Å (cartesian)."""
    lines = content.split("\n")
    centers: list[list[float]] = []
    for i, line in enumerate(lines):
        if "Final State" not in line:
            continue
        j = 1
        while i + j < len(lines) and "WF centre and spread" in lines[i + j]:
            parts = re.sub("[(),]", " ", lines[i + j]).split()
            centers.append([float(x) for x in parts[5:8]])
            j += 1
    if not centers:
        raise ValueError("No `Final State` Wannier centres found in the .wout contents")
    return np.array(centers, dtype=float)


# ----------------------------------------------------------------------
# Hamiltonian loaders
# ----------------------------------------------------------------------


def load_primary_hr(
    content: str, num_wann: int, num_wann_sc: int, kgrid: tuple[int, int, int]
) -> NDArray[np.complex128]:
    """Load the (Koopmans) Hamiltonian being interpolated.

    Γ-only files (``nrpts == 1``, e.g. from kcp.x on the supercell) come
    back as ``(num_wann_sc, num_wann_sc)``; k-point files are restricted to
    the primitive-cell R-vectors and reshaped to ``(num_wann_sc, num_wann)``.
    """
    hr, rvect, _, nrpts = parse_hr_file_contents(content)
    if nrpts == 1:
        if len(hr) != num_wann_sc**2:
            raise ValueError(
                f"Wrong number of matrix elements ({len(hr)}) for the input hamiltonian"
            )
        return hr.reshape(num_wann_sc, num_wann_sc)
    if len(hr) != nrpts * num_wann**2:
        raise ValueError(f"Wrong number of matrix elements ({len(hr)}) for the input hamiltonian")
    hr_grid = extract_hr(hr.reshape(nrpts, num_wann, num_wann), rvect, *kgrid)
    return hr_grid.reshape(num_wann_sc, num_wann)


def load_coarse_hr(
    content: str, num_wann: int, num_wann_sc: int, kgrid: tuple[int, int, int]
) -> NDArray[np.complex128]:
    """Load the coarse DFT Hamiltonian used by the smooth-interpolation method.

    Returns a ``(num_wann_sc, num_wann)`` matrix.
    """
    hr, rvect, _, nrpts = parse_hr_file_contents(content)
    if len(hr) != nrpts * num_wann**2:
        raise ValueError(f"Wrong number of matrix elements for hr_coarse {len(hr)}")
    hr_grid = extract_hr(hr.reshape(nrpts, num_wann, num_wann), rvect, *kgrid)
    return hr_grid.reshape(num_wann_sc, num_wann)


def load_smooth_hr(
    content: str, num_wann: int
) -> tuple[NDArray[np.complex128], NDArray[np.int_], NDArray[np.int_]]:
    """Load the smooth (dense-grid) DFT Hamiltonian.

    Returns ``(hr_smooth, rvect, weights)`` with ``hr_smooth`` shaped
    ``(nrpts, num_wann, num_wann)`` — it stays on its own Wigner-Seitz
    R-vectors, whose degeneracy ``weights`` divide the Fourier sum.
    """
    hr, rvect, weights, nrpts = parse_hr_file_contents(content)
    if len(hr) != nrpts * num_wann**2:
        raise ValueError(f"Wrong number of matrix elements for hr_smooth {len(hr)}")
    return hr.reshape(nrpts, num_wann, num_wann), rvect, np.array(weights, dtype=int)


# ----------------------------------------------------------------------
# Interpolation core
# ----------------------------------------------------------------------


def correct_phase(
    centers: NDArray[np.float64],
    kpts: NDArray[np.float64],
    rvec: NDArray[np.int_],
    kgrid: tuple[int, int, int],
    acell: NDArray[np.float64],
    num_wann: int,
    num_wann_sc: int,
    use_ws_distance: bool,
) -> NDArray[np.complex128]:
    """Determine the phase factor entering the Fourier transform of H(R).

    The correction consists of finding the right distance — i.e. the right
    R-vector — considering also the Born-von-Kármán boundary conditions.
    If ``use_ws_distance`` the intracell distance between Wannier functions
    is accounted for as in the Wannier90 code; otherwise only intercell
    distances enter. All vectors must be in crystal units.
    """
    if use_ws_distance:
        wf_dist = np.concatenate([centers] * num_wann) - np.concatenate(
            [[c] * num_wann_sc for c in centers[:num_wann]]
        )
    else:
        wf_dist = np.array(
            np.concatenate([[rvect] * num_wann for rvect in rvec]).tolist() * num_wann
        )

    # Supercell lattice vectors (the 27 neighbouring supercell translations)
    tvec = np.array(
        [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
    ) * np.asarray(kgrid)

    phase = np.zeros((len(kpts), len(wf_dist)), dtype=complex)
    for i, dist in enumerate(wf_dist):
        distance = crys_to_cart(dist + tvec, acell, +1)
        norms = np.linalg.norm(distance, axis=1)
        t_index = np.where(norms - norms.min() < 1.0e-3)[0]
        phase[:, i] = np.exp(2j * pi * np.dot(kpts, tvec[t_index].transpose())).sum(axis=1) / len(
            t_index
        )

    phase_reshaped = phase.reshape(len(kpts), num_wann, len(rvec), num_wann)
    return np.transpose(phase_reshaped, axes=(0, 2, 3, 1))


def calc_bands(
    hr: NDArray[np.complex128],
    centers: NDArray[np.float64],
    kpts: NDArray[np.float64],
    rvec: NDArray[np.int_],
    kgrid: tuple[int, int, int],
    acell: NDArray[np.float64],
    num_wann: int,
    num_wann_sc: int,
    use_ws_distance: bool = True,
    hr_coarse: NDArray[np.complex128] | None = None,
    hr_smooth: NDArray[np.complex128] | None = None,
    rvect_smooth: NDArray[np.int_] | None = None,
    weights_smooth: NDArray[np.int_] | None = None,
) -> NDArray[np.float64]:
    """Interpolate the electronic bands along ``kpts`` by Fourier transforming H(R).

    When the smooth-interpolation Hamiltonians are given, the coarse DFT
    part is removed from ``hr`` in real space and the dense-grid DFT part
    is added back in k-space. Returns the eigenvalues as a
    ``(len(kpts), num_wann)`` array.
    """
    hr = hr[:, :num_wann]
    if hr_coarse is not None:
        hr = hr - hr_coarse
    hr = hr.reshape(len(rvec), num_wann, num_wann)

    # phi has shape Nkpath x NR; phi_corr has shape Nkpath x NR x num_wann x num_wann
    phi = np.exp(2j * pi * np.dot(kpts, rvec.transpose()))
    phi_corr = correct_phase(
        centers, kpts, rvec, kgrid, acell, num_wann, num_wann_sc, use_ws_distance
    )

    hk = np.transpose(
        np.sum(phi * np.transpose(hr * phi_corr, axes=(2, 3, 0, 1)), axis=3), axes=(2, 0, 1)
    )
    if hr_smooth is not None:
        if rvect_smooth is None or weights_smooth is None:
            raise ValueError("hr_smooth requires its R-vectors and weights")
        phi = np.exp(2j * pi * np.dot(kpts, rvect_smooth.transpose()))
        hr_smooth_w = np.transpose(hr_smooth, axes=(2, 1, 0)) / weights_smooth
        hk += np.dot(phi, np.transpose(hr_smooth_w, axes=(1, 2, 0)))

    return np.linalg.eigvalsh(hk).astype(np.float64, copy=False)


# ----------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------


def infer_wannier_counts(n_centers: int, kgrid: tuple[int, int, int]) -> tuple[int, int]:
    """Return ``(num_wann, num_wann_sc)`` from the number of parsed centres.

    The centres describe one primitive cell's worth of Wannier functions.
    """
    return n_centers, n_centers * int(np.prod(kgrid))


def unfold_and_interpolate(
    hr_content: str,
    centers: NDArray[np.float64],
    cell: NDArray[np.float64],
    kgrid: tuple[int, int, int],
    kpath_kpts: NDArray[np.float64],
    use_ws_distance: bool = True,
    dft_ham_content: str | None = None,
    dft_smooth_ham_content: str | None = None,
) -> NDArray[np.float64]:
    """Unfold a supercell Wannier Hamiltonian and interpolate its bands along a k-path.

    ``hr_content`` (and the optional coarse/smooth DFT Hamiltonian contents,
    which switch on the smooth-interpolation method when both are given)
    are Wannier90-format ``*_hr.dat`` file contents; ``centers`` are the
    Wannier centres in Å (cartesian, as printed in the ``.wout``);
    ``cell`` is the primitive cell in Å; ``kpath_kpts`` are the band-path
    k-points in primitive-cell crystal units.

    Returns the interpolated eigenvalues as a
    ``(len(kpath_kpts), num_wann)`` array (eV, same units as the input
    Hamiltonians).
    """
    num_wann, num_wann_sc = infer_wannier_counts(len(centers), kgrid)

    # Generate centres for the non-primitive-cell (R≠0) WFs, in
    # primitive-cell crystal units.
    rvec = latt_vect(*kgrid)
    alat = float(np.linalg.norm(cell[0]))
    acell = np.asarray(cell, dtype=float) / alat
    centers_crys = crys_to_cart(np.asarray(centers, dtype=float) / alat, reciprocal_cell(acell), -1)
    centers_all = np.concatenate([centers_crys + rvect for rvect in rvec])

    hr = load_primary_hr(hr_content, num_wann, num_wann_sc, kgrid)

    hr_coarse = hr_smooth = rvect_smooth = weights_smooth = None
    if dft_smooth_ham_content is not None:
        if dft_ham_content is None:
            raise ValueError(
                "Smooth interpolation requires both the coarse DFT Hamiltonian and the "
                "smooth (dense-grid) DFT Hamiltonian"
            )
        hr_coarse = load_coarse_hr(dft_ham_content, num_wann, num_wann_sc, kgrid)
        hr_smooth, rvect_smooth, weights_smooth = load_smooth_hr(dft_smooth_ham_content, num_wann)

    return calc_bands(
        hr,
        centers_all,
        np.asarray(kpath_kpts, dtype=float),
        rvec,
        kgrid,
        acell,
        num_wann,
        num_wann_sc,
        use_ws_distance=use_ws_distance,
        hr_coarse=hr_coarse,
        hr_smooth=hr_smooth,
        rvect_smooth=rvect_smooth,
        weights_smooth=weights_smooth,
    )


# ----------------------------------------------------------------------
# DOS
# ----------------------------------------------------------------------


def compute_dos(
    energies_skn: NDArray[np.float64],
    width: float = 0.05,
    emin: float | None = None,
    emax: float | None = None,
    npts: int = 1001,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute a Gaussian-smearing total DOS from band energies.

    Unit k-point weights, an explicit energy window, and the *total* DOS
    (both spins summed for two spin channels; doubled for one), following
    ASE's ``ase.dft.dos.DOS``. ``energies_skn`` is indexed by
    (spin, k-point, band).

    Returns ``(energy_grid, dos)``.
    """
    e_skn = np.asarray(energies_skn, dtype=float)
    if e_skn.ndim != 3:
        raise ValueError("energies_skn must be indexed by (spin, k-point, band)")
    if emin is None:
        emin = e_skn.min() - 5 * width
    if emax is None:
        emax = e_skn.max() + 5 * width
    grid = np.linspace(emin, emax, npts)

    def spin_dos(e_kn: NDArray[np.float64]) -> NDArray[np.float64]:
        """Accumulate unit-weight Gaussians centred on one spin channel's eigenvalues."""
        dos = np.zeros(npts)
        for e in e_kn.flatten():
            dos += np.exp(-(((grid - e) / width) ** 2)) / (sqrt(pi) * width)
        return dos

    nspins = e_skn.shape[0]
    if nspins == 2:
        total = spin_dos(e_skn[0]) + spin_dos(e_skn[1])
    else:
        total = 2 * spin_dos(e_skn[0])
    return grid, total
