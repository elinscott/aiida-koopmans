"""MLWF / projected-WF initialisation of the variational orbitals.

Ports the ``init_orbitals in ('mlwfs', 'projwfs')`` branch of legacy
``InitializationWorkflow`` (``koopmans/workflows/_koopmans_dscf.py:1186-1262``)
for periodic systems:

1. Wannierise every projection block off one shared scf + nscf
   (:func:`~aiida_koopmans.workgraphs.block_wannierize.BlockWannierizeTask`);
2. fold the per-block Wannier orbitals into Γ-point supercell wavefunctions
   (:func:`~aiida_koopmans.workgraphs.folding.FoldToSupercell`);
3. run a from-scratch ``dft_dummy`` kcp.x on the supercell purely to lay out
   the save-directory skeleton;
4. run the real ``dft_init`` kcp.x restarting from the dummy save with
   ``restart_from_wannier_pwscf=.true.``, the folded ``evc_occupied{n}.dat``
   / ``evc0_empty{n}.dat`` files staged into its read ``K00001``;
5. check the result for consistency (PW-vs-CP band gap within 2% of the PW
   gap; kcp.x total energy already converged at its first step).

A failing check fails this graph, and because a ``@task.graph`` runs as its
own sub-process whose outputs only resolve on successful termination, the
downstream screening pipeline (which consumes ``remote_folder``) never
launches off a broken initialisation — matching the legacy workflow, which
raises before proceeding.

The graph returns the ``dft_init`` save (parent of the first trial KI) plus
the folded occupied-manifold files, which the trial KI re-stages into its
own read directory (legacy ``DeltaSCFIterationWorkflow`` links every
``variational_orbital_file``; the ``evc0_empty{n}.dat`` entries flow through
the ``dft_init`` save automatically, so only the ``evc_occupied{n}.dat``
pair needs explicit re-staging).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import SpinType
from aiida_workgraph import dynamic, task

from aiida_koopmans.types import ProjectionBlock, SpinChannel, group_blocks_to_merge
from aiida_koopmans.utils import KOOPMANS_NODE_DESERIALIZERS
from aiida_koopmans.workgraphs import Codes
from aiida_koopmans.workgraphs.block_wannierize import BlockWannierizeTask
from aiida_koopmans.workgraphs.folding import FoldToSupercell, enumerate_fold_targets
from aiida_koopmans.workgraphs.kcp import (
    KcpBaseTask,
    UpfData,
    _build_dft_parameters,
    _build_kcp_inputs,
    _kcp_base_inputs,
)
from aiida_koopmans.workgraphs.supercell import supercell_size

# Legacy consistency thresholds (``_koopmans_dscf.py:1250-1262``).
_GAP_RELATIVE_TOLERANCE = 2.0e-2
_ENERGY_RELATIVE_TOLERANCE = 1.0e-6

_BANDS_DESERIALIZERS = {
    **KOOPMANS_NODE_DESERIALIZERS,
    "aiida.orm.nodes.data.array.bands.BandsData": "aiida_koopmans.utils.passthrough_node",
}


class MlwfInitializationOutputs(TypedDict):
    """Outputs of :func:`MlwfInitialization`.

    * ``remote_folder`` — the ``dft_init`` save; the first trial KI's
      ``parent_folder``.
    * ``evc_occupied1`` / ``evc_occupied2`` — the folded occupied-manifold
      wavefunctions (merge_evc.x ``merged_file`` outputs) the trial KI
      stages into its read ``K00001``.
    * ``report`` — the consistency-check numbers (PW/CP gaps and the
      initial/final kcp.x energies).
    """

    remote_folder: orm.RemoteData
    evc_occupied1: orm.SinglefileData
    evc_occupied2: orm.SinglefileData
    report: dict


@task(deserializers=_BANDS_DESERIALIZERS)
def check_wannier_initialization(
    *,
    nscf_output_parameters: dict,
    nscf_bands: orm.BandsData,
    init_output_parameters: dict,
) -> dict:
    """Check the Wannier-seeded ``dft_init`` against the PW reference.

    Two legacy guards (``_koopmans_dscf.py:1250-1262``), both fatal:

    * the kcp.x (CP) band gap must match the pw.x nscf gap to within 2% of
      the PW gap — a mismatch means the folded Wannier orbitals do not span
      the occupied manifold;
    * the kcp.x total energy must be converged from its very first CG step
      (relative drift below 1e-6) — the run restarts from supposedly
      converged Wannier orbitals, so any minimisation indicates a broken
      restart.

    The PW HOMO / LUMO are recomputed from the nscf eigenvalues +
    occupations (aiida-quantumespresso does not expose them as scalars);
    both codes report energies in eV so the comparison is direct. Raises
    ``ValueError`` on violation; returns the compared numbers otherwise.

    ``nscf_output_parameters`` is accepted (and recorded in provenance)
    even though the gap comes from the bands array — it ties the check to
    the nscf's scalar results should later diagnostics need them.
    """
    del nscf_output_parameters  # provenance-only input for now
    bands = nscf_bands.get_bands()
    try:
        occupations = nscf_bands.get_array("occupations")
    except KeyError as exc:
        raise ValueError(
            "The nscf BandsData carries no occupations array; cannot locate the "
            "PW HOMO / LUMO for the gap-consistency check."
        ) from exc
    occupied = occupations > 0.5 * occupations.max()
    if occupied.all():
        raise ValueError(
            "The nscf run has no empty bands, so the PW LUMO (and hence the "
            "gap-consistency check) is undefined. Increase the nscf nbnd."
        )
    pw_homo = float(bands[occupied].max())
    pw_lumo = float(bands[~occupied].min())
    pw_gap = pw_lumo - pw_homo

    cp_homo = init_output_parameters.get("homo_energy")
    cp_lumo = init_output_parameters.get("lumo_energy")
    if cp_homo is None or cp_lumo is None:
        raise ValueError(
            "The dft_init kcp.x output reports no HOMO / LUMO energies; cannot "
            "run the PW-vs-CP gap-consistency check."
        )
    cp_gap = cp_lumo - cp_homo
    if abs(pw_gap - cp_gap) > _GAP_RELATIVE_TOLERANCE * pw_gap:
        raise ValueError(f"PW and CP band gaps are not consistent: {pw_gap} {cp_gap}")

    convergence = init_output_parameters.get("convergence", {}).get("filled", [])
    if not convergence:
        raise ValueError(
            "The dft_init kcp.x output reports no convergence history; cannot "
            "verify that the Wannier restart was already converged."
        )
    initial_energy = convergence[0]["Etot"]
    final_energy = init_output_parameters["energy"]
    if abs(final_energy - initial_energy) > _ENERGY_RELATIVE_TOLERANCE * abs(final_energy):
        raise ValueError(
            "Too much difference between the initial and final CP energies: "
            f"{initial_energy} {final_energy}"
        )

    return {
        "pw_gap": pw_gap,
        "cp_gap": cp_gap,
        "initial_energy": initial_energy,
        "final_energy": final_energy,
    }


def _build_dft_dummy_parameters(base) -> dict[str, Any]:
    """kcp.x parameters for the ``dft_dummy`` step.

    A from-scratch DFT run whose only purpose is to write a save-directory
    skeleton for ``dft_init`` to restart from — legacy
    ``internal_new_kcp_calculator('dft_dummy')``: outer loops off and no
    ``nbnd`` (the empty states will come from the folded Wannier files, so
    the dummy needn't allocate them).
    """
    parameters = _build_dft_parameters(base, nbnd=0, outerloop=False)
    parameters["SYSTEM"].pop("nbnd", None)
    return parameters


def _build_dft_init_from_wannier_parameters(base, *, nbnd: int) -> dict[str, Any]:
    """kcp.x parameters for the Wannier-seeded ``dft_init`` step.

    Legacy ``internal_new_kcp_calculator('dft_init', restart_mode='restart',
    restart_from_wannier_pwscf=True, do_outerloop=True)``
    (``_koopmans_dscf.py:1241-1242``), plus the blanket solids rule that
    the empty manifold is never minimised (``_koopmans_dscf.py:1123-1126``)
    — the empty variational orbitals stay the folded Wannier functions.
    """
    parameters = _build_dft_parameters(base, nbnd=nbnd, restart_mode="restart", outerloop=True)
    parameters["SYSTEM"]["restart_from_wannier_pwscf"] = True
    parameters["ELECTRONS"]["do_outerloop_empty"] = False
    parameters["ELECTRONS"].pop("empty_states_maxstep", None)
    return parameters


@task.graph
def MlwfInitialization(
    *,
    codes: Codes,
    structure: orm.StructureData,
    supercell: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    blocks: list[ProjectionBlock],
    kpoints: orm.KpointsData,
    kgrid: list[int],
    nelec: int,
    nelup: int,
    neldw: int,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int = 2,
    tot_magnetization: int | None = None,
    spin_polarized: bool = False,
    gamma_only: bool = False,
    pseudo_family: str | None = None,
    wannier_protocol: str | None = None,
    wannier_overrides: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> MlwfInitializationOutputs:
    """Initialise the variational orbitals from (projected) Wannier functions.

    Args:
        codes: code instances; required keys ``pw``, ``wannier90``,
            ``pw2wannier90``, ``wann2kcp``, ``merge_evc``, ``kcp``
            (``projwfc`` only for projection types that need it).
        structure: the *primitive* periodic cell — the wannierisation runs
            here.
        supercell: the ``diag(kgrid)`` repeat of ``structure`` — the kcp.x
            runs happen here, at Γ.
        pseudos: pseudopotentials resolved for the supercell kinds (the
            primitive shares them).
        blocks: projection blocks with *primitive* band indices.
        kpoints: explicit unshifted k-mesh matching ``kgrid``, shared by the
            nscf and every block's wannier90 / pw2wannier90.
        kgrid: the Monkhorst-Pack grid (= supercell repeat counts).
        nelec / nelup / neldw / tot_magnetization: **supercell** electron
            counts (from ``count_electrons_task`` on ``supercell``).
        ecutwfc / ecutrho: plane-wave cutoffs (intensive; shared by pw.x
            and kcp.x).
        nbnd: **supercell** total band count for kcp.x.
        nspin: kcp.x spin treatment (always 2 in the DSCF flow).
        spin_polarized: whether the blocks are spin-resolved.
        gamma_only: Γ-only primitive sampling (forwarded to wann2kcp.x's
            ``gamma_trick``).
        pseudo_family / wannier_protocol / wannier_overrides: forwarded to
            the wannierisation builders.
        options: ``metadata.options`` for the kcp.x / folding CalcJobs.
    """
    # Merge groups are defined by *primitive* occupation counts (the blocks
    # carry primitive band indices); the supercell counts divide back down
    # by the number of primitive cells.
    ncells = supercell_size(kgrid)
    if spin_polarized:
        num_occ_bands = {
            SpinChannel.UP: nelup // ncells,
            SpinChannel.DOWN: neldw // ncells,
        }
    else:
        num_occ_bands = {SpinChannel.NONE: (nelec // ncells) // 2}
    merge_groups = group_blocks_to_merge(blocks, num_occ_bands)

    # --- B1: block-by-block wannierisation on the primitive cell ---
    wannierize = BlockWannierizeTask(
        codes=codes,
        structure=structure,
        blocks=blocks,
        kpoints=kpoints,
        pseudo_family=pseudo_family,
        protocol=wannier_protocol,
        overrides=wannier_overrides,
        spin_type=SpinType.COLLINEAR if spin_polarized else SpinType.NONE,
        metadata={"call_link_label": "wannierize"},
    )

    # --- B2: fold + merge into supercell kcp.x wavefunctions ---
    fold = FoldToSupercell(
        codes=codes,
        blocks=blocks,
        merge_groups=merge_groups,
        nscf_remote_folder=wannierize["nscf"]["remote_folder"],
        block_wannier=wannierize["blocks"],
        kgrid=kgrid,
        gamma_only=gamma_only,
        spin_polarized=spin_polarized,
        options=options,
        metadata={"call_link_label": "fold_to_supercell"},
    )

    # --- B3: dft_dummy — save-skeleton writer on the supercell ---
    base = _kcp_base_inputs(
        supercell,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
    )
    dummy_inputs = _build_kcp_inputs(
        codes["kcp"],
        supercell,
        _build_dft_dummy_parameters(base),
        pseudos,
        options=options,
        name="dft_dummy",
    )
    dummy = KcpBaseTask(**dummy_inputs)

    # --- B4: dft_init — restart from the dummy save with the folded
    # Wannier wavefunctions staged into the read K00001 (legacy
    # ``_koopmans_dscf.py:1243-1245``). The staged stems mirror exactly
    # what the fold produced.
    staged = {
        target["stem"]: fold[target["stem"]]
        for target in enumerate_fold_targets(merge_groups, spin_polarized)
    }
    init_inputs = _build_kcp_inputs(
        codes["kcp"],
        supercell,
        _build_dft_init_from_wannier_parameters(base, nbnd=nbnd),
        pseudos,
        options=options,
        parent_folder=dummy["remote_folder"],
        read_wavefunctions=staged,
        name="dft_init",
    )
    dft_init = KcpBaseTask(**init_inputs)

    # --- B5: consistency checks. Their failure fails this whole graph,
    # whose outputs (below) only resolve on success — that's the barrier
    # that keeps the screening pipeline from launching off a broken
    # initialisation.
    report = check_wannier_initialization(
        nscf_output_parameters=wannierize["nscf"]["output_parameters"],
        nscf_bands=wannierize["nscf"]["output_band"],
        init_output_parameters=dft_init["output_parameters"],
        metadata={"call_link_label": "consistency_check"},
    )

    return MlwfInitializationOutputs(
        remote_folder=dft_init["remote_folder"],
        evc_occupied1=fold["evc_occupied1"],
        evc_occupied2=fold["evc_occupied2"],
        report=report.result,
    )
