"""Unfold-and-interpolate workgraph: Wannier-based band-structure interpolation.

A supercell (or coarse-grid) Wannier Hamiltonian is unfolded onto the
primitive cell and its eigenvalues are interpolated along a k-path, with an
optional smooth-interpolation correction from a denser-grid DFT Hamiltonian
and an optional Gaussian-smearing DOS.

All the maths lives in :mod:`aiida_koopmans.ui_helpers` (pure numpy); the
tasks here only unpack ORM nodes into plain arrays, so the interpolated
bands carry provenance back to the input Hamiltonian / ``.wout`` files.

Scope notes:

* One (occupied or empty) x (spin) block per graph. The per-(filling, spin)
  fan-out and band merging of a full singlepoint band structure belong to
  the DSCF/DFPT band-structure integration, not here.
* ``wf_phases.dat`` phase renormalisation is not exposed.
  :func:`aiida_koopmans.ui_helpers.calc_bands` accepts phases, but no
  working renormalisation expression exists for realistic shapes
  (``num_wann_sc`` phases against a ``(num_wann, num_wann, nR)`` array).
* The smooth-interpolation correction consumes a pre-computed denser-grid
  DFT Hamiltonian (``dft_smooth_ham_file``); wannierizing that denser grid
  is the caller's job.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from aiida import orm
from aiida_workgraph import task

from aiida_koopmans import ui_helpers
from aiida_koopmans.utils import KOOPMANS_NODE_DESERIALIZERS


class DensityOfStates(TypedDict):
    """A Gaussian-smearing total DOS on a uniform energy grid (both in eV)."""

    energies: list[float]
    dos: list[float]


class _RequiredOutputs(TypedDict):
    """Required outputs of :func:`UnfoldAndInterpolateTask`: the interpolated bands."""

    band_energies: list[list[float]]


class UnfoldAndInterpolateOutputs(_RequiredOutputs, total=False):
    """Outputs of :func:`UnfoldAndInterpolateTask`.

    * ``band_energies`` — the interpolated eigenvalues along the input
      k-path, one ``(n_kpoints, n_bands)`` table in eV.
    * ``dos`` — Gaussian-smearing total DOS of those bands (only when
      ``do_dos``; declared ``total=False`` rather than ``NotRequired`` so
      the graph-output socket keeps an annotation the socket type-checker
      can match).
    """

    dos: DensityOfStates


@task(deserializers=KOOPMANS_NODE_DESERIALIZERS)
def interpolate_bands(
    kc_ham_file: orm.SinglefileData,
    wannier90_wout: orm.SinglefileData,
    structure: orm.StructureData,
    kpath: orm.KpointsData,
    kgrid: list[int],
    do_map: bool = False,
    use_ws_distance: bool = True,
    w90_input_sc: bool = False,
    dft_ham_file: orm.SinglefileData | None = None,
    dft_smooth_ham_file: orm.SinglefileData | None = None,
) -> list[list[float]]:
    """Unfold the Wannier Hamiltonian and interpolate its bands along ``kpath``.

    ``wannier90_wout`` supplies the Wannier centres and spreads (its
    ``Final State`` block). Passing both DFT Hamiltonians switches on the
    smooth-interpolation method. Returns the ``(n_kpoints, n_bands)``
    eigenvalue table in eV.
    """
    centers, spreads = ui_helpers.parse_wout_centers_and_spreads(wannier90_wout.get_content("r"))
    k1, k2, k3 = (int(n) for n in kgrid)

    energies = ui_helpers.unfold_and_interpolate(
        hr_content=kc_ham_file.get_content("r"),
        centers=centers,
        spreads=spreads,
        cell=np.array(structure.cell),
        kgrid=(k1, k2, k3),
        kpath_kpts=kpath.get_kpoints(),
        w90_input_sc=bool(w90_input_sc),
        do_map=bool(do_map),
        use_ws_distance=bool(use_ws_distance),
        dft_ham_content=dft_ham_file.get_content("r") if dft_ham_file is not None else None,
        dft_smooth_ham_content=(
            dft_smooth_ham_file.get_content("r") if dft_smooth_ham_file is not None else None
        ),
    )
    return [[float(e) for e in row] for row in energies]


@task
def compute_dos_from_bands(
    band_energies: list[list[float]],
    plotting: dict,
) -> DensityOfStates:
    """Compute the Gaussian-smearing total DOS of an interpolated band structure.

    ``plotting`` carries ``degauss`` (smearing width, eV), ``nstep`` (the
    energy grid has ``nstep + 1`` points), and optional ``Emin`` / ``Emax``
    window bounds (default: the eigenvalue range padded by five smearing
    widths).
    """
    bands = np.asarray(band_energies, dtype=float)
    e_skn = bands if bands.ndim == 3 else bands[np.newaxis]

    grid, dos = ui_helpers.compute_dos(
        e_skn,
        width=float(plotting.get("degauss", 0.05)),
        emin=plotting.get("Emin"),
        emax=plotting.get("Emax"),
        npts=int(plotting.get("nstep", 1000)) + 1,
    )
    return DensityOfStates(energies=grid.tolist(), dos=dos.tolist())


@task.graph
def UnfoldAndInterpolateTask(
    kc_ham_file: orm.SinglefileData,
    wannier90_wout: orm.SinglefileData,
    structure: orm.StructureData,
    kpath: orm.KpointsData,
    kgrid: list[int],
    do_map: bool = False,
    use_ws_distance: bool = True,
    w90_input_sc: bool = False,
    do_dos: bool = True,
    dft_ham_file: orm.SinglefileData | None = None,
    dft_smooth_ham_file: orm.SinglefileData | None = None,
    plotting: dict | None = None,
) -> UnfoldAndInterpolateOutputs:
    """Interpolate a band structure from a Wannier Hamiltonian, optionally with a DOS.

    ``kc_ham_file`` is the Hamiltonian to interpolate, ``wannier90_wout``
    the Wannier90 output providing centres and spreads, ``kgrid`` the
    Monkhorst-Pack grid the supercell corresponds to, and ``kpath`` the
    primitive-cell band path (crystal coordinates). Supplying both
    ``dft_ham_file`` and ``dft_smooth_ham_file`` activates the
    smooth-interpolation method. ``plotting`` (``degauss`` / ``nstep`` /
    ``Emin`` / ``Emax``) shapes the optional DOS.
    """
    interpolation_kwargs = {}
    if dft_ham_file is not None:
        interpolation_kwargs["dft_ham_file"] = dft_ham_file
    if dft_smooth_ham_file is not None:
        interpolation_kwargs["dft_smooth_ham_file"] = dft_smooth_ham_file

    bands = interpolate_bands(
        kc_ham_file=kc_ham_file,
        wannier90_wout=wannier90_wout,
        structure=structure,
        kpath=kpath,
        kgrid=[int(n) for n in kgrid],
        do_map=bool(do_map),
        use_ws_distance=bool(use_ws_distance),
        w90_input_sc=bool(w90_input_sc),
        **interpolation_kwargs,
    ).result

    outputs = UnfoldAndInterpolateOutputs(band_energies=bands)

    if do_dos:
        # The DensityOfStates return fans out into one socket per key; the
        # whole namespace becomes the ``dos`` output.
        outputs["dos"] = compute_dos_from_bands(
            band_energies=bands,
            plotting=dict(plotting) if plotting is not None else {},
        )

    return outputs
