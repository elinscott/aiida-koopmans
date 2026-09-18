"""Band structure of a periodic ΔSCF singlepoint, by unfold-and-interpolate.

A ΔSCF run computes on a Γ-point supercell, so its eigenvalues are folded.
:func:`DscfBandStructureTask` recovers the primitive-cell picture: the final
KI prints one Koopmans Hamiltonian per (filling, spin) manifold, each is
unfolded and interpolated along the primitive-cell k-path with that
manifold's Wannier centres, and the results are concatenated into one band
structure whose reference energy is the valence-band maximum.

Manifold membership and band order come from the ``merge_groups`` partition
the initialisation wannierization emitted, never from the block labels.

Passing a second, denser-mesh wannierization of the same blocks switches on
the smooth-interpolation correction: each manifold's DFT Hamiltonian is
removed from the Koopmans one in real space and its denser-mesh counterpart
added back in k-space.
"""

# No ``from __future__ import annotations``: stringified annotations hide
# ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the socket type-checker reads.

from typing import Annotated, NotRequired, TypedDict

from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlockOutputs
from aiida_koopmans.workgraphs.kcp_files import kcp_hamiltonian_filename
from aiida_koopmans.workgraphs.ui import (
    DensityOfStates,
    compute_dos_from_bands,
)
from aiida_koopmans.workgraphs.ui.manifolds import (
    build_band_structure,
    extract_koopmans_hamiltonian,
    interpolate_manifold,
    merge_manifold_energies,
)


class DscfBandStructureOutputs(TypedDict):
    """Outputs of :func:`DscfBandStructureTask`.

    * ``band_structure`` — the interpolated Koopmans bands along the input
      k-path, occupied then empty within each spin channel, on pw.x's
      absolute energy scale (same convention as the DFPT route's kcw.x
      bands), ``offset`` having been added to every eigenvalue; an
      ``offset`` of 0.0 leaves them on kcp.x's own scale.
    * ``reference`` — the valence-band maximum in eV, for plot alignment.
    * ``dos`` — the bands' Gaussian-smearing total DOS, present only when
      ``do_dos``.
    """

    band_structure: orm.BandsData
    reference: float
    dos: NotRequired[DensityOfStates]


def _select_manifold(merge_groups: list, *, filled: bool, spin: SpinChannel) -> list:
    """Return one ``(filled, spin)`` manifold's block labels, in band order.

    Raises if the run has no projection manifold for the combination: every
    band structure needs an occupied and an empty manifold in each spin
    channel it interpolates.
    """
    matches = [
        group["blocks"]
        for group in merge_groups
        if bool(group["filled"]) == filled and SpinChannel(group["spin"]) == spin
    ]
    if not matches:
        raise ValueError(
            f"Interpolating a band structure needs an occupied and an empty projection "
            f"manifold in every spin channel; the run has none for filled={filled}, "
            f"spin={spin.value!r}. Add projections covering the empty bands (and both "
            "spin channels, if polarized)."
        )
    [blocks] = matches
    return [block["label"] for block in blocks]


@task.graph
def DscfBandStructureTask(
    structure: orm.StructureData,
    merge_groups: list,
    block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)],
    koopmans_ham_retrieved: orm.FolderData,
    kgrid: list[int],
    kpath: orm.KpointsData,
    smooth_block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)] | None = None,
    spin_polarized: bool = False,
    use_ws_distance: bool = True,
    do_dos: bool = True,
    plotting: dict | None = None,
    offset: float = 0.0,
) -> DscfBandStructureOutputs:
    """Interpolate the Koopmans band structure of a periodic ΔSCF singlepoint.

    One interpolation per (filling, spin) manifold, off the Hamiltonian the
    final KI printed for it, with that manifold's Wannier centres; the
    results are concatenated occupied-then-empty within a channel and
    stacked across channels.

    Args:
        structure: the primitive cell the wannierization ran on, which the
            supercell Hamiltonian is unfolded back onto.
        merge_groups: the ``(filled, spin, blocks)`` partition the
            initialisation wannierization emitted, which fixes both
            manifold membership and band order.
        block_wannierizations: the per-block wannierization outputs, keyed
            by block label.
        smooth_block_wannierizations: the same blocks Wannierized on a
            denser mesh, keyed by the same labels. Present asks for the
            smooth-interpolation correction: each manifold's DFT
            Hamiltonian is subtracted in real space and its dense
            counterpart added back in k-space. Absent interpolates the
            Koopmans Hamiltonian alone.
        koopmans_ham_retrieved: the final KI's retrieved folder, holding
            the ``ham_occ_?.dat`` / ``ham_emp_?.dat`` files.
        kgrid: the Monkhorst-Pack grid, which is also the supercell's
            repeat count along each lattice vector.
        kpath: the primitive-cell band path, in crystal coordinates.
        plotting: DOS shaping — ``degauss``, ``nstep``, ``Emin``, ``Emax``.
        offset: the shift from kcp.x's absolute energy scale to pw.x's
            (:func:`~aiida_koopmans.workgraphs.mlwf_init.check_wannier_initialization`'s
            ``offset`` output), added to every returned eigenvalue and to
            the reference. 0.0 keeps kcp.x's own scale.
    """
    spins = [SpinChannel.UP, SpinChannel.DOWN] if spin_polarized else [SpinChannel.NONE]

    energies_by_manifold = {}
    for spin in spins:
        for filled in (True, False):
            manifold = "occ" if filled else "emp"
            label = manifold if spin == SpinChannel.NONE else f"{manifold}_{spin.value}"
            hamiltonian = extract_koopmans_hamiltonian(
                retrieved=koopmans_ham_retrieved,
                filename=kcp_hamiltonian_filename(
                    filled=filled,
                    # kcp.x indexes its printed files 1 = up (and the single
                    # channel of an unpolarized run), 2 = down.
                    spin_index=2 if spin == SpinChannel.DOWN else 1,
                ),
                metadata={"call_link_label": f"extract_{label}_hamiltonian"},
            ).result
            energies_by_manifold[filled, spin] = interpolate_manifold(
                hamiltonian,
                _select_manifold(merge_groups, filled=filled, spin=spin),
                label=label,
                block_wannierizations=block_wannierizations,
                smooth_block_wannierizations=smooth_block_wannierizations,
                structure=structure,
                kpath=kpath,
                kgrid=kgrid,
                use_ws_distance=use_ws_distance,
            )

    first = SpinChannel.UP if spin_polarized else SpinChannel.NONE
    merged = merge_manifold_energies(
        occupied=energies_by_manifold[True, first],
        empty=energies_by_manifold[False, first],
        occupied_down=energies_by_manifold.get((True, SpinChannel.DOWN)),
        empty_down=energies_by_manifold.get((False, SpinChannel.DOWN)),
        offset=offset,
        metadata={"call_link_label": "merge_manifold_energies"},
    )

    outputs = DscfBandStructureOutputs(
        band_structure=build_band_structure(
            kpath=kpath,
            energies=merged["energies"],
            reference=merged["reference"],
            metadata={"call_link_label": "build_band_structure"},
        ).result,
        reference=merged["reference"],
    )
    if do_dos:
        outputs["dos"] = compute_dos_from_bands(
            band_energies=merged["energies"],
            plotting=dict(plotting) if plotting is not None else {},
            metadata={"call_link_label": "interpolated_dos"},
        )
    return outputs
