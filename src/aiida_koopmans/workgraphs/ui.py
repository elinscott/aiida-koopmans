"""Unfold-and-interpolate workgraph: Wannier-based band-structure interpolation.

Port of the legacy standalone ``task='ui'`` workflow (the single-process
``SingleUnfoldAndInterpolateWorkflow`` driving one
``UnfoldAndInterpolateProcess``, ``koopmans/processes/ui/_process.py``): a
supercell (or coarse-grid) Wannier Hamiltonian is unfolded onto the
primitive cell and its eigenvalues are interpolated along a k-path, with an
optional smooth-interpolation correction from a denser-grid DFT
Hamiltonian and an optional Gaussian-smearing DOS.

All the maths lives in :mod:`aiida_koopmans.ui_helpers` (pure numpy); the
``@task.calcfunction``s here only translate between ORM nodes and plain
arrays, so the interpolated bands carry provenance back to the input
Hamiltonian / ``.wout`` files.

Scope notes:

* One (occupied or empty) x (spin) block per graph — exactly what the
  standalone legacy task ran. The per-block fan-out + band merging that the
  legacy ``UnfoldAndInterpolateWorkflow`` performs inside a DSCF/DFPT
  singlepoint (occ/emp x spin runs concatenated into one band structure)
  belongs to the singlepoint integration and is not wired here yet.
* The legacy ``wf_phases.dat`` phase renormalisation is not exposed: it
  only applies to supercell-Wannier90 inputs and the legacy expression
  cannot broadcast for any realistic shape (``num_wann_sc`` phases against
  an ``(num_wann, num_wann, nR)`` array), so no working behaviour exists to
  port. :func:`aiida_koopmans.ui_helpers.calc_bands` still accepts phases
  for a future fix.
* On-the-fly smooth Wannierization (legacy re-ran a ``WannierizeWorkflow``
  on the denser grid when no smooth Hamiltonian was supplied) is likewise
  part of the singlepoint integration; the standalone task always consumed
  a pre-computed file, which is what the ``dft_smooth_ham_file`` input
  takes.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from aiida import orm
from aiida_workgraph import task

from aiida_koopmans import ui_helpers


class _RequiredOutputs(TypedDict):
    """Required outputs of :func:`UnfoldAndInterpolateTask`: the interpolated bands."""

    band_structure: orm.BandsData


class UnfoldAndInterpolateOutputs(_RequiredOutputs, total=False):
    """Outputs of :func:`UnfoldAndInterpolateTask`.

    * ``band_structure`` — the interpolated bands along the input k-path.
    * ``dos`` — Gaussian-smearing total DOS of those bands (only when
      ``do_dos``; declared ``total=False`` rather than ``NotRequired`` so
      the graph-output socket keeps the plain ``XyData`` annotation the
      socket type-checker can match).
    """

    dos: orm.XyData


@task.calcfunction
def interpolate_bands(
    kc_ham_file: orm.SinglefileData,
    wannier90_wout: orm.SinglefileData,
    structure: orm.StructureData,
    kpath: orm.KpointsData,
    kgrid: orm.List,
    parameters: orm.Dict,
    dft_ham_file: orm.SinglefileData | None = None,
    dft_smooth_ham_file: orm.SinglefileData | None = None,
) -> orm.BandsData:
    """Unfold the Wannier Hamiltonian and interpolate its bands along ``kpath``.

    ``wannier90_wout`` supplies the Wannier centres and spreads (its
    ``Final State`` block); ``parameters`` carries the boolean knobs
    ``do_map`` / ``use_ws_distance`` / ``w90_input_sc``. Passing both DFT
    Hamiltonians switches on the smooth-interpolation method. Single-output
    convention: consumers wire the bands via ``interpolate_bands(...).result``.
    """
    params = parameters.get_dict()
    centers, spreads = ui_helpers.parse_wout_centers_and_spreads(wannier90_wout.get_content("r"))
    k1, k2, k3 = (int(n) for n in kgrid.get_list())

    energies = ui_helpers.unfold_and_interpolate(
        hr_content=kc_ham_file.get_content("r"),
        centers=centers,
        spreads=spreads,
        cell=np.array(structure.cell),
        kgrid=(k1, k2, k3),
        kpath_kpts=kpath.get_kpoints(),
        w90_input_sc=bool(params.get("w90_input_sc", False)),
        do_map=bool(params.get("do_map", False)),
        use_ws_distance=bool(params.get("use_ws_distance", True)),
        dft_ham_content=dft_ham_file.get_content("r") if dft_ham_file is not None else None,
        dft_smooth_ham_content=(
            dft_smooth_ham_file.get_content("r") if dft_smooth_ham_file is not None else None
        ),
    )

    bands = orm.BandsData()
    bands.set_kpointsdata(kpath)
    bands.set_bands(energies, units="eV")
    return bands


@task.calcfunction
def compute_dos_from_bands(
    band_structure: orm.BandsData,
    plotting: orm.Dict,
) -> orm.XyData:
    """Compute the Gaussian-smearing total DOS of an interpolated band structure.

    ``plotting`` mirrors the legacy plot settings: ``degauss`` (smearing
    width, eV), ``nstep`` (the energy grid has ``nstep + 1`` points), and
    optional ``Emin`` / ``Emax`` window bounds (default: the eigenvalue
    range padded by five smearing widths).
    """
    params = plotting.get_dict()
    bands = band_structure.get_bands()
    e_skn = bands if bands.ndim == 3 else bands[np.newaxis]

    grid, dos = ui_helpers.compute_dos(
        e_skn,
        width=float(params.get("degauss", 0.05)),
        emin=params.get("Emin"),
        emax=params.get("Emax"),
        npts=int(params.get("nstep", 1000)) + 1,
    )

    xy = orm.XyData()
    xy.set_x(grid, "energy", "eV")
    xy.set_y(dos, "dos", "states/eV")
    return xy


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

    Inputs mirror the legacy ``ui`` calculator block: ``kc_ham_file`` is
    the Hamiltonian to interpolate, ``wannier90_wout`` the Wannier90 output
    providing centres and spreads, ``kgrid`` the Monkhorst-Pack grid the
    supercell corresponds to, and ``kpath`` the primitive-cell band path
    (crystal coordinates). Supplying both ``dft_ham_file`` and
    ``dft_smooth_ham_file`` activates the smooth-interpolation method.
    ``plotting`` (``degauss`` / ``nstep`` / ``Emin`` / ``Emax``) shapes the
    optional DOS.
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
        parameters={
            "do_map": bool(do_map),
            "use_ws_distance": bool(use_ws_distance),
            "w90_input_sc": bool(w90_input_sc),
        },
        **interpolation_kwargs,
    ).result

    outputs = UnfoldAndInterpolateOutputs(band_structure=bands)

    if do_dos:
        outputs["dos"] = compute_dos_from_bands(
            band_structure=bands,
            plotting=dict(plotting) if plotting is not None else {},
        ).result

    return outputs
