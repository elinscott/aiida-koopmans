"""Pure-Python merging of per-block Wannier90 product files.

Every function here takes and returns plain strings / numpy arrays, so the
``@task.calcfunction`` in :mod:`aiida_koopmans.workgraphs.dfpt` that stages
them stays a thin wrapper.

kcw.x consumes *one* set of Wannier files per manifold (``<seed>_u.mat``,
``_hr.dat``, ``_centres.xyz``, plus the empty manifold's ``_u_dis.mat``),
so when a manifold is Wannierised as several independent blocks the
per-block products must be combined:

* ``hr`` (real-space Hamiltonian): block-diagonal at every R-vector. All
  blocks must share the same R-vector list and degeneracy weights, which
  is guaranteed when they come off the same k-mesh (the Wigner-Seitz R
  grid depends only on the lattice and the mesh, not on the projections).
* ``u`` (Bloch→Wannier rotations): block-diagonal at every k-point. The
  manifold's bands are ordered block by block, so the row (Wannier) and
  column (band) offsets both advance by each block's ``num_wann``.
* ``centres``: Wannier centres concatenated in block order, the atom
  coordinates appended once.
* ``u_dis`` (disentanglement): only the *last* block of a manifold carries
  bands beyond its Wannier count (the band layout the manifold derivation
  fixes), so the merged matrix is an identity for the preceding blocks
  with the last block's rectangular ``u_dis`` in the bottom-right corner.

The generate functions reproduce the fixed-width formats kcw.x parses.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np


def _timestamp() -> str:
    """Return the file-header timestamp (UTC, seconds precision)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# hr (real-space Hamiltonian) files
# ---------------------------------------------------------------------------


def parse_wannier_hr_file_contents(content: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Parse a Wannier90 ``_hr.dat`` file.

    Returns ``(ham, rvect, weights)`` where ``ham`` is complex with shape
    ``(nrpts, num_wann, num_wann)``, ``rvect`` is an integer ``(nrpts, 3)``
    array of R-vectors, and ``weights`` the ``nrpts`` degeneracy weights.
    """
    lines = content.rstrip("\n").split("\n")
    if "written on" not in lines[0].lower():
        raise ValueError("The format of the Hamiltonian file contents is not recognized.")

    num_wann = int(lines[1].split()[0])
    nrpts = int(lines[2].split()[0])

    # The degeneracy weights are written 15 per line.
    weight_lines = nrpts // 15 + (1 if nrpts % 15 else 0)
    lines_to_skip = 3 + weight_lines
    weights = [int(x) for line in lines[3:lines_to_skip] for x in line.split()]

    ham = np.empty(nrpts * num_wann * num_wann, dtype=complex)
    rvect = np.empty((nrpts, 3), dtype=int)
    for i, line in enumerate(lines[lines_to_skip:]):
        fields = line.split()
        ham[i] = float(fields[5]) + 1j * float(fields[6])
        if i % num_wann**2 == 0:
            rvect[i // num_wann**2] = [int(x) for x in fields[0:3]]

    return ham.reshape(nrpts, num_wann, num_wann), rvect, weights


def generate_wannier_hr_file_contents(
    ham: np.ndarray, rvect: np.ndarray | Sequence[Sequence[int]], weights: Sequence[int]
) -> str:
    """Generate the contents of a Wannier90 ``_hr.dat`` file."""
    nrpts = len(weights)
    num_wann = np.size(ham, -1)
    expected_shape = (nrpts, num_wann, num_wann)
    if ham.shape != expected_shape:
        raise ValueError(f"`ham` has shape {ham.shape}; expected {expected_shape}")

    flines = [f" Written on {_timestamp()}"]
    flines.append(f"{num_wann:12d}")
    flines.append(f"{nrpts:12d}")

    ints_per_line = 15
    for pos in range(0, len(weights), ints_per_line):
        flines.append("".join([f"{x:5d}" for x in weights[pos : pos + ints_per_line]]))

    for r, ham_block in zip(rvect, ham, strict=True):
        flines += [
            f"{r[0]:5d}{r[1]:5d}{r[2]:5d}{j + 1:5d}{i + 1:5d}{val.real:12.6f}{val.imag:12.6f}"
            for i, row in enumerate(ham_block)
            for j, val in enumerate(row)
        ]

    return "\n".join(flines) + "\n"


def merge_wannier_hr_file_contents(contents: Sequence[str]) -> str:
    """Merge per-block ``_hr.dat`` contents into one block-diagonal Hamiltonian.

    Every block must share the same R-vectors and degeneracy weights; the
    merged Hamiltonian at each R-point is the direct sum of the per-block
    Hamiltonians, in the order given (== band order within the manifold).
    """
    if not contents:
        raise ValueError("No hr file contents provided to merge.")
    parsed = [parse_wannier_hr_file_contents(content) for content in contents]
    ham_list = [ham for ham, _, _ in parsed]
    _, rvect_out, weights_out = parsed[0]
    for _, rvect, weights in parsed[1:]:
        if weights != weights_out:
            raise ValueError("Cannot merge HR file contents that have differing weights.")
        if not np.array_equal(rvect, rvect_out):
            raise ValueError("Cannot merge HR file contents with differing sets of R-vectors.")

    nrpts = len(weights_out)
    num_wann_tot = sum(ham.shape[-1] for ham in ham_list)
    ham_out = np.zeros((nrpts, num_wann_tot, num_wann_tot), dtype=complex)
    start = 0
    for ham in ham_list:
        end = start + ham.shape[-1]
        ham_out[:, start:end, start:end] = ham
        start = end

    return generate_wannier_hr_file_contents(ham_out, rvect_out, weights_out)


# ---------------------------------------------------------------------------
# u / u_dis (rotation / disentanglement matrix) files
# ---------------------------------------------------------------------------


def parse_wannier_u_file_shape(content: str) -> tuple[int, int, int]:
    """Read the ``(nkpts, num_wann, num_bands)`` header of a ``_u[_dis].mat`` file."""
    nk, m, n = (int(x) for x in content.split("\n")[1].split())
    return nk, m, n


def parse_wannier_u_file_contents(content: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Wannier90 ``_u.mat`` / ``_u_dis.mat`` file.

    Returns ``(umat, kpts)`` with ``umat`` of complex shape
    ``(nkpts, num_wann, num_bands)`` and ``kpts`` of shape ``(nkpts, 3)``.
    The matrix rows index Wannier functions and the columns bands, matching
    the Fortran write order of the file (band index fastest).
    """
    lines = content.split("\n")
    nk, m, n = parse_wannier_u_file_shape(content)

    kpts = np.empty((nk, 3), dtype=float)
    umat = np.empty((nk, m, n), dtype=complex)
    block = 2 + m * n  # blank line + k-point line + m*n value lines
    for ik in range(nk):
        offset = 3 + ik * block  # index of this k-point's coordinate line
        kpts[ik] = [float(x) for x in lines[offset].split()]
        values = [
            complex(*[float(x) for x in line.split()])
            for line in lines[offset + 1 : offset + 1 + m * n]
        ]
        umat[ik] = np.reshape(values, (m, n))

    return umat, kpts


def generate_wannier_u_file_contents(umat: np.ndarray, kpts: np.ndarray) -> str:
    """Generate the contents of a Wannier90 ``_u[_dis].mat`` file."""
    flines = [f" Written on {_timestamp()}"]
    flines.append("".join([f"{x:12d}" for x in umat.shape]))

    for kpt, umatk in zip(kpts, umat, strict=True):
        flines.append("")
        flines.append("".join([f"{k:15.10f}" for k in kpt]))
        flines += [f"{c.real:15.10f}{c.imag:15.10f}" for c in umatk.flatten()]

    return "\n".join(flines) + "\n"


def merge_wannier_u_file_contents(contents: Sequence[str]) -> str:
    """Merge per-block ``_u.mat`` contents into one block-diagonal rotation.

    Every block must share the same k-points; the merged matrix at each
    k-point is the direct sum of the per-block rotations (both the Wannier
    and the band offset advance by each block's ``num_wann``, since the
    manifold's bands are ordered block by block).
    """
    if not contents:
        raise ValueError("No U matrix file contents provided to merge.")
    parsed = [parse_wannier_u_file_contents(content) for content in contents]
    u_list = [umat for umat, _ in parsed]
    kpts_out = parsed[0][1]
    for _, kpts in parsed[1:]:
        if not (len(kpts) == len(kpts_out) and np.allclose(kpts, kpts_out)):
            raise ValueError("Cannot merge U matrix file contents with differing sets of k-points.")

    num_wann_tot = sum(u.shape[1] for u in u_list)
    num_bands_tot = sum(u.shape[2] for u in u_list)
    u_merged = np.zeros((len(kpts_out), num_wann_tot, num_bands_tot), dtype=complex)

    i_start = 0
    j_start = 0
    for u in u_list:
        i_end = i_start + u.shape[1]
        j_end = j_start + u.shape[2]
        u_merged[:, i_start:i_end, j_start:j_end] = u
        i_start = i_end
        j_start = j_end

    return generate_wannier_u_file_contents(u_merged, kpts_out)


def extend_wannier_u_dis_file_content(content: str, nbnd: int, nwann: int) -> str:
    """Extend one block's ``_u_dis.mat`` to cover its whole (merged) manifold.

    Only the last block of a manifold is disentangled (it alone carries
    bands beyond its Wannier count), so the manifold-wide ``nwann x nbnd``
    disentanglement matrix is an identity mapping band *i* → Wannier *i*
    for the preceding blocks, with this block's rectangular ``u_dis`` in
    the bottom-right corner (overwriting the identity entries it overlaps).

    Args:
        content: the last block's ``_u_dis.mat`` contents.
        nbnd: total number of bands of the merged manifold (e.g. all empty
            bands of the calculation for the empty manifold).
        nwann: total number of Wannier functions of the merged manifold.
    """
    udis_mat, kpts = parse_wannier_u_file_contents(content)

    udis_mat_large = np.zeros((len(kpts), nwann, nbnd), dtype=complex)
    udis_mat_large[:, :nwann, :nwann] = np.identity(nwann)
    udis_mat_large[:, -udis_mat.shape[1] :, -udis_mat.shape[2] :] = udis_mat

    return generate_wannier_u_file_contents(udis_mat_large, kpts)


# ---------------------------------------------------------------------------
# centres (xyz) files
# ---------------------------------------------------------------------------


def parse_wannier_centres_file_contents(content: str) -> tuple[list[list[float]], list[str]]:
    """Parse a Wannier90 ``_centres.xyz`` file.

    Returns ``(centres, atom_lines)``: the Wannier centre coordinates and
    the atom entries kept as verbatim lines (the atoms are re-emitted
    unchanged on merge, so there is no need to interpret them).
    """
    centres: list[list[float]] = []
    atom_lines: list[str] = []
    for line in content.split("\n")[2:]:
        if not line.strip():
            continue
        if line.startswith("X    "):
            centres.append([float(x) for x in line.split()[1:]])
        else:
            atom_lines.append(line)
    return centres, atom_lines


def generate_wannier_centres_file_contents(
    centres: Sequence[Sequence[float]], atom_lines: Sequence[str]
) -> str:
    """Generate the contents of a Wannier90 ``_centres.xyz`` file."""
    flines = [
        f"{len(centres) + len(atom_lines):6d}",
        f" Wannier centres, written by koopmans on {_timestamp()}",
    ]
    for centre in centres:
        flines.append("X    " + "".join([f"{x:16.8f}" for x in centre]))
    flines += list(atom_lines)

    return "\n".join(flines) + "\n"


def merge_wannier_centres_file_contents(contents: Sequence[str]) -> str:
    """Merge per-block ``_centres.xyz`` contents: centres concatenated, atoms once.

    All blocks describe the same structure, so their atom entries must
    coincide; the atoms of the first file are re-emitted verbatim.
    """
    if not contents:
        raise ValueError("No centres file contents provided to merge.")
    parsed = [parse_wannier_centres_file_contents(content) for content in contents]
    atom_lines_out = parsed[0][1]
    for _, atom_lines in parsed[1:]:
        if atom_lines != atom_lines_out:
            raise ValueError(
                "Cannot merge Wannier centres file contents with differing atomic entries."
            )
    centres = [centre for block_centres, _ in parsed for centre in block_centres]

    return generate_wannier_centres_file_contents(centres, atom_lines_out)
