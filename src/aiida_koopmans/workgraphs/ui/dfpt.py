"""Smooth-interpolated band structure of a DFPT (kcw.x) singlepoint.

kcw.x's ``ham`` step interpolates the whole Koopmans Hamiltonian from the
Monkhorst-Pack grid the Wannier functions were built on, so on a coarse
grid the DFT part of that Hamiltonian interpolates badly. The
smooth-interpolation method replaces it: the coarse DFT Hamiltonian is
removed from the Koopmans one in real space, and the same quantity from a
Wannierization on a denser mesh is added back in k-space.

:func:`DfptBandStructureTask` runs that for one spin channel, off the
``*.kcw_hr_occ.dat`` / ``*.kcw_hr_emp.dat`` files the ham step retrieves
under ``HAM.write_hr``. The interpolation itself is the machinery the ΔSCF
route uses (:mod:`aiida_koopmans.workgraphs.ui.manifolds`); only the
Hamiltonian's provenance differs.
"""

# No ``from __future__ import annotations``: stringified annotations hide
# ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the socket type-checker reads.

from typing import Annotated, TypedDict

from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans.calculations.kcw import kcw_hamiltonian_filename
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlockOutputs
from aiida_koopmans.workgraphs.ui.manifolds import (
    build_band_structure,
    extract_koopmans_hamiltonian,
    interpolate_manifold,
    merge_manifold_energies,
)


class DfptBandStructureOutputs(TypedDict):
    """Outputs of :func:`DfptBandStructureTask`.

    * ``band_structure`` — the smooth-interpolated Koopmans bands along the
      input k-path, occupied then empty, on pw.x's absolute energy scale
      (kcw.x's own scale; nothing is shifted).
    * ``reference`` — the valence-band maximum in eV, for plot alignment.
    """

    band_structure: orm.BandsData
    reference: float


@task.graph
def DfptBandStructureTask(
    structure: orm.StructureData,
    koopmans_ham_retrieved: orm.FolderData,
    block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)],
    smooth_block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)],
    occ_labels: list,
    kgrid: list[int],
    kpath: orm.KpointsData,
    emp_labels: list | None = None,
    use_ws_distance: bool = True,
) -> DfptBandStructureOutputs:
    """Interpolate one kcw.x channel's Koopmans bands with the smooth correction.

    One interpolation per manifold, off the Hamiltonian the ham step
    printed for it, with that manifold's Wannier centres; the results are
    concatenated occupied-then-empty.

    Args:
        structure: the primitive cell the Wannierizations ran on.
        koopmans_ham_retrieved: the ham step's retrieved folder, holding the
            ``*.kcw_hr_occ.dat`` / ``*.kcw_hr_emp.dat`` files.
        block_wannierizations: the per-block wannierization outputs on the
            kcw.x mesh, keyed by block label.
        smooth_block_wannierizations: the same blocks Wannierized on the
            denser mesh, keyed by the same labels.
        occ_labels: the occupied manifold's block labels, in band order.
        kgrid: the Monkhorst-Pack grid the Koopmans Hamiltonian lives on
            (kcw.x's ``CONTROL.mp1-3``).
        kpath: the band path, in crystal coordinates.
        emp_labels: the empty manifold's block labels, in band order; omit
            for an occupied-only run.
        use_ws_distance: whether the Wigner-Seitz distance between Wannier
            centres enters the interpolation phase, as in wannier90. Pass
            what the ham step ran with (``HAM.use_ws_distance``), so the two
            interpolations of the same Hamiltonian agree.
    """
    energies = {}
    for filled, labels in ((True, occ_labels), (False, emp_labels)):
        if labels is None:
            continue
        label = "occ" if filled else "emp"
        hamiltonian = extract_koopmans_hamiltonian(
            retrieved=koopmans_ham_retrieved,
            filename=kcw_hamiltonian_filename(filled=filled),
            metadata={"call_link_label": f"extract_{label}_hamiltonian"},
        ).result
        energies[filled] = interpolate_manifold(
            hamiltonian,
            [str(block_label) for block_label in labels],
            label=label,
            block_wannierizations=block_wannierizations,
            smooth_block_wannierizations=smooth_block_wannierizations,
            structure=structure,
            kpath=kpath,
            kgrid=kgrid,
            use_ws_distance=use_ws_distance,
        )

    merged = merge_manifold_energies(
        occupied=energies[True],
        empty=energies.get(False),
        metadata={"call_link_label": "merge_manifold_energies"},
    )
    return DfptBandStructureOutputs(
        band_structure=build_band_structure(
            kpath=kpath,
            energies=merged["energies"],
            reference=merged["reference"],
            metadata={"call_link_label": "build_band_structure"},
        ).result,
        reference=merged["reference"],
    )
