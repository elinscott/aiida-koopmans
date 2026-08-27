"""Band structure of a periodic ΔSCF singlepoint, by unfold-and-interpolate.

A ΔSCF run computes on a Γ-point supercell, so its eigenvalues are folded.
:func:`DscfBandStructureTask` recovers the primitive-cell picture: the final
KI prints one Koopmans Hamiltonian per (filling, spin) manifold, each is
unfolded and interpolated along the primitive-cell k-path with that
manifold's Wannier centres, and the results are concatenated into one band
structure whose reference energy is the valence-band maximum.

Manifold membership and band order come from the ``merge_groups`` partition
the initialisation wannierization emitted, never from the block labels.
"""

# No ``from __future__ import annotations``: stringified annotations hide
# ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the socket type-checker reads.

import io
from typing import Annotated, NotRequired, TypedDict

import numpy as np
from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.workgraphs.block_wannierize import (
    WannierizeBlockOutputs,
    collect_wannier_functions,
)
from aiida_koopmans.workgraphs.kcp import kcp_hamiltonian_filename
from aiida_koopmans.workgraphs.ui import (
    DensityOfStates,
    compute_dos_from_bands,
    interpolate_bands,
)


class DscfBandStructureOutputs(TypedDict):
    """Outputs of :func:`DscfBandStructureTask`.

    * ``band_structure`` — the interpolated Koopmans bands along the input
      k-path, occupied then empty within each spin channel.
    * ``reference`` — the valence-band maximum in eV, for plot alignment.
    * ``dos`` — the bands' Gaussian-smearing total DOS, present only when
      ``do_dos``.
    """

    band_structure: orm.BandsData
    reference: float
    dos: NotRequired[DensityOfStates]


@task.calcfunction
def extract_koopmans_hamiltonian(
    retrieved: orm.FolderData, filename: orm.Str
) -> orm.SinglefileData:
    """Lift one printed Koopmans Hamiltonian out of a kcp.x retrieved folder.

    A calcfunction, not a plain ``@task``: it takes an AiiDA data node,
    which the PyFunction deserializer refuses.
    """
    name = filename.value
    available = retrieved.base.repository.list_object_names()
    if name not in available:
        raise ValueError(
            f"`{name}` is missing from the kcp.x retrieved folder (contents: "
            f"{sorted(available)}). The final KI must run with `write_hr`."
        )
    content = retrieved.base.repository.get_object_content(name, mode="rb")
    return orm.SinglefileData(io.BytesIO(content), filename=name)


@task(outputs=["energies", "reference"])
def merge_manifold_energies(
    occupied: list[list[float]],
    empty: list[list[float]],
    occupied_down: list[list[float]] | None = None,
    empty_down: list[list[float]] | None = None,
) -> dict:
    """Concatenate per-manifold interpolated eigenvalues into one table.

    Within a spin channel the occupied and empty energies join along the
    band axis; both ``*_down`` inputs together add a leading spin axis.
    ``reference`` is the highest occupied energy across the channels.
    """
    if (occupied_down is None) != (empty_down is None):
        raise ValueError(
            "A spin-polarized merge needs both `occupied_down` and `empty_down`; got one."
        )
    occ = np.asarray(occupied, dtype=float)
    emp = np.asarray(empty, dtype=float)
    if occ.shape[0] != emp.shape[0]:
        raise ValueError(
            f"The occupied and empty manifolds were interpolated along different k-paths "
            f"({occ.shape[0]} vs {emp.shape[0]} k-points); they cannot be concatenated."
        )
    if occupied_down is None:
        energies = np.concatenate([occ, emp], axis=1)
        reference = float(occ.max())
    else:
        down = np.concatenate(
            [np.asarray(occupied_down, dtype=float), np.asarray(empty_down, dtype=float)], axis=1
        )
        up = np.concatenate([occ, emp], axis=1)
        if up.shape != down.shape:
            raise ValueError(
                f"The spin channels interpolated to different shapes ({up.shape} vs "
                f"{down.shape}); they cannot be stacked into one band structure."
            )
        energies = np.stack([up, down])
        reference = float(max(occ.max(), np.asarray(occupied_down, dtype=float).max()))
    return {"energies": energies.tolist(), "reference": reference}


@task.calcfunction
def build_band_structure(
    kpath: orm.KpointsData, energies: orm.List, reference: orm.Float
) -> orm.BandsData:
    """Attach interpolated eigenvalues (eV) to their k-path as a ``BandsData``.

    ``reference`` is the valence-band maximum. It is an input rather than
    part of the returned node so a consumer reading the bands off
    provenance finds the energy they align to on the same calculation.
    """
    bands = orm.BandsData()
    bands.set_kpointsdata(kpath)
    bands.set_bands(np.asarray(energies.get_list(), dtype=float), units="eV")
    return bands


def _select_manifold(merge_groups: list, *, filled: bool, spin: SpinChannel) -> list:
    """Return one ``(filled, spin)`` manifold's blocks from the group partition.

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
    return blocks


def _interpolate_manifold(
    blocks: list,
    *,
    label: str,
    filled: bool,
    spin_index: int,
    block_wannierizations,
    koopmans_ham_retrieved,
    structure,
    kpath,
    kgrid: list[int],
    use_ws_distance: bool,
):
    """Add the tasks interpolating one manifold; return its eigenvalue socket.

    The centres come from each block's parsed wannier90 output, keyed so
    lexicographic key order is the manifold's band order.
    """
    hamiltonian = extract_koopmans_hamiltonian(
        retrieved=koopmans_ham_retrieved,
        filename=kcp_hamiltonian_filename(filled=filled, spin_index=spin_index),
        metadata={"call_link_label": f"extract_{label}_hamiltonian"},
    ).result

    wannier_functions = collect_wannier_functions(
        output_parameters={
            f"b{index:02d}": block_wannierizations[block["label"]]["output_parameters"]
            for index, block in enumerate(blocks)
        },
        metadata={"call_link_label": f"collect_{label}_centres"},
    )

    return interpolate_bands(
        kc_ham_file=hamiltonian,
        centres=wannier_functions["centres"],
        structure=structure,
        kpath=kpath,
        kgrid=[int(n) for n in kgrid],
        use_ws_distance=bool(use_ws_distance),
        metadata={"call_link_label": f"interpolate_{label}"},
    ).result


@task.graph
def DscfBandStructureTask(
    structure: orm.StructureData,
    merge_groups: list,
    block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)],
    koopmans_ham_retrieved: orm.FolderData,
    kgrid: list[int],
    kpath: orm.KpointsData,
    spin_polarized: bool = False,
    use_ws_distance: bool = True,
    do_dos: bool = True,
    plotting: dict | None = None,
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
        koopmans_ham_retrieved: the final KI's retrieved folder, holding
            the ``ham_occ_?.dat`` / ``ham_emp_?.dat`` files.
        kgrid: the Monkhorst-Pack grid, which is also the supercell's
            repeat count along each lattice vector.
        kpath: the primitive-cell band path, in crystal coordinates.
        plotting: DOS shaping — ``degauss``, ``nstep``, ``Emin``, ``Emax``.
    """
    spins = [SpinChannel.UP, SpinChannel.DOWN] if spin_polarized else [SpinChannel.NONE]

    energies_by_manifold = {}
    for spin in spins:
        for filled in (True, False):
            manifold = "occ" if filled else "emp"
            label = manifold if spin == SpinChannel.NONE else f"{manifold}_{spin.value}"
            energies_by_manifold[filled, spin] = _interpolate_manifold(
                _select_manifold(merge_groups, filled=filled, spin=spin),
                label=label,
                filled=filled,
                # kcp.x indexes its printed files 1 = up (and the single
                # channel of an unpolarized run), 2 = down.
                spin_index=2 if spin == SpinChannel.DOWN else 1,
                block_wannierizations=block_wannierizations,
                koopmans_ham_retrieved=koopmans_ham_retrieved,
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
