"""Denser-mesh wannierization feeding the smooth-interpolation correction.

Smooth interpolation replaces the DFT part of the Koopmans Hamiltonian
with one wannierized on a mesh ``smooth_int_factor`` times denser: the
coarse DFT Hamiltonian is subtracted in real space and the dense one added
back in k-space (:func:`~aiida_koopmans.workgraphs.ui.helpers.calc_bands`).
This module produces that dense Hamiltonian — the same projection blocks
as the initialisation wannierization, so both Hamiltonians share a gauge,
wannierized off an scf on the coarse mesh and an nscf on the dense one.
"""

from typing import Annotated, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import SpinType
from aiida_workgraph import dynamic, task
from node_graph import reference

from aiida_koopmans.parallelization import ParallelizationDict
from aiida_koopmans.projections import ProjectionBlock
from aiida_koopmans.workgraphs.block_wannierize import (
    WannierizeBlockOutputs,
    WannierizeBlocks,
    WannierizeBlocksCodes,
    WannierizeOverrides,
)


class SmoothWannierizationOutputs(TypedDict):
    """Outputs of :func:`SmoothWannierization`.

    * ``blocks`` — the per-block wannierization outputs on the denser mesh,
      keyed by the same block labels as the coarse run, so a consumer pairs
      the two Hamiltonians block by block.
    """

    blocks: Annotated[dict, dynamic(WannierizeBlockOutputs)]


@task.graph
def SmoothWannierization(
    codes: WannierizeBlocksCodes,
    structure: orm.StructureData,
    blocks: list[ProjectionBlock],
    smooth_kpoints: orm.KpointsData,
    smooth_mp_grid: list[int],
    scf_kpoints: orm.KpointsData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    spin_type: SpinType = SpinType.NONE,
    parallelization: ParallelizationDict | None = None,
) -> SmoothWannierizationOutputs:
    """Wannierize ``blocks`` on a denser mesh than the run that seeded them.

    Args:
        codes: code instances (:class:`WannierizeBlocksCodes`).
        structure: the primitive periodic cell.
        blocks: the same projection blocks the coarse run used — same
            projections, ``num_wann``, band windows and disentanglement
            windows, so the two Hamiltonians share the Wannier gauge the
            smooth-interpolation correction assumes.
        smooth_kpoints: the denser mesh as an explicit k-point list, which
            the nscf and every block's wannier90 / pw2wannier90 sample.
        smooth_mp_grid: the Monkhorst-Pack dimensions of ``smooth_kpoints``
            (wannier90 cannot re-derive them from an explicit list).
        scf_kpoints: the mesh the scf samples — the *coarse* one. The dense
            mesh enters at the nscf; running the scf on it too would only
            re-converge the same density more expensively.
        pseudo_family / protocol / overrides / spin_type / parallelization:
            forwarded to the wannierization builders.
    """
    wannierize = WannierizeBlocks(
        codes={
            "pw": reference(codes, "pw"),
            "pw2wannier90": reference(codes, "pw2wannier90"),
            "wannier90": reference(codes, "wannier90"),
        },
        structure=structure,
        blocks=blocks,
        kpoints=smooth_kpoints,
        mp_grid=list(smooth_mp_grid),
        scf_kpoints=scf_kpoints,
        pseudo_family=pseudo_family,
        protocol=protocol,
        overrides=overrides,
        spin_type=spin_type,
        parallelization=parallelization,
        metadata={"call_link_label": "wannierize", "label": "Wannierization"},
    )
    return SmoothWannierizationOutputs(blocks=wannierize["blocks"])
