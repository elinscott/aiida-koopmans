"""Workgraphs that wrap aiida-wannier90-workflows workchains."""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

import numpy as np
from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_wannier90_workflows.common.types import (
    OptimizeMetric,
    OptimizeMuReference,
    OptimizeStrategy,
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)
from aiida_wannier90_workflows.workflows import Wannier90OptimizeWorkChain, Wannier90WorkChain
from aiida_workgraph import task
from aiida_workgraph.socket_spec import SocketMeta
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.parallelization import (
    ParallelizationDict,
    merge_parallelization_into_existing_namespaces,
    validate_parallelization,
)
from aiida_koopmans.workgraphs import enforce_step_calculation, unwrap_enum

# ``PwOutputs`` is the canonical single-PwBaseWorkChain output shape; it
# lives in ``pw.py`` next to the other pw output types. Re-exported here so
# existing ``from ...wannier90 import PwOutputs`` call sites keep working.
from aiida_koopmans.workgraphs.pw import PwOutputs

__all__ = ["PwOutputs"]


class WannierizeCodes(TypedDict):
    """Codes for the whole-manifold wannierization chain.

    Shared by :func:`Wannierize` and :func:`OptimizeWannierization`, whose
    upstream builders wire the same scf / nscf / pw2wannier90 / wannier90
    steps.
    """

    pw: orm.AbstractCode
    pw2wannier90: orm.AbstractCode
    wannier90: orm.AbstractCode
    projwfc: NotRequired[
        Annotated[
            orm.AbstractCode,
            SocketMeta(help="Needed for SCDM projections and projectability disentanglement."),
        ]
    ]


class Wannier90Outputs(TypedDict):
    """Outputs of a Wannier90BaseWorkChain run."""

    remote_folder: orm.RemoteData
    remote_stash: orm.RemoteData
    retrieved: orm.FolderData
    output_parameters: dict
    interpolated_bands: orm.BandsData
    nnkp_file: orm.SinglefileData
    # Multi-key ArrayData (one array per disentanglement / spread iteration);
    # default deserializer can't flatten — leave as orm.ArrayData.
    disentanglement_data: NotRequired[orm.ArrayData]
    spread_data: NotRequired[orm.ArrayData]


class ProjwfcOutputs(TypedDict, total=False):
    """Outputs of a ProjwfcCalculation."""

    remote_folder: orm.RemoteData
    remote_stash: orm.RemoteData
    retrieved: orm.FolderData
    output_parameters: dict
    Dos: orm.XyData
    Ldos: orm.XyData
    Pdos: orm.XyData
    projections_up: orm.ProjectionData
    bands_up: orm.BandsData
    projections_down: orm.ProjectionData
    bands_down: orm.BandsData
    projections: orm.ProjectionData
    bands: orm.BandsData


class WannierWorkflowOutputs(TypedDict):
    """Output types for Wannier90 workgraph tasks."""

    scf: PwOutputs
    nscf: PwOutputs
    wannier90: Wannier90Outputs
    wannier90_up: Wannier90Outputs
    wannier90_down: Wannier90Outputs
    projwfc: ProjwfcOutputs


Wannier90Step = task(Wannier90WorkChain)
Wannier90OptimizeStep = task(Wannier90OptimizeWorkChain)


def _finalize_wannier_builder(
    builder: Any,
    *,
    kpoint_path: dict[str, Any] | None,
    bands_kpoints: orm.KpointsData | None,
    projector_rotation: np.ndarray | None,
    set_bands_kpoints: bool,
) -> dict[str, Any]:
    """Apply the shared bands-path / projector-rotation wiring, then flatten to a dict.

    Both ``Wannierize`` and ``OptimizeWannierization``
    share this finalisation tail: enforce that ``kpoint_path`` and
    ``bands_kpoints`` are mutually exclusive, wire the explicit bands path
    onto the nested wannier90 builder, apply the optional
    ``projector_rotation``, and reduce the builder to the plain-dict inputs
    the wrapped task expects.

    ``set_bands_kpoints`` distinguishes the two callers: the plain builder
    assigns ``bands_kpoints`` onto ``builder.wannier90.wannier90`` here,
    whereas the optimize builder passes it to ``get_builder_from_protocol``
    upstream and only needs it for the mutual-exclusion check.
    """
    if kpoint_path is not None and bands_kpoints is not None:
        raise ValueError("Cannot specify both `kpoint_path` and `bands_kpoints`.")

    if kpoint_path is not None:
        builder.wannier90.wannier90.kpoint_path = kpoint_path

    if set_bands_kpoints and bands_kpoints is not None:
        builder.wannier90.wannier90.bands_kpoints = bands_kpoints

    if projector_rotation is not None:
        builder.projector_rotation = projector_rotation

    data = get_dict_from_builder(builder)

    # The wannierisation nscf step owns calculation='nscf'; raise if a merged
    # override set it otherwise. The scf/wannier steps carry no such conflict.
    nscf_pw = data.get("nscf", {}).get("pw")
    if nscf_pw is not None and nscf_pw.get("parameters") is not None:
        nscf_pw["parameters"] = orm.Dict(
            enforce_step_calculation(nscf_pw["parameters"].get_dict(), "nscf", "nscf")
        )

    return data


def _apply_kpoint_mesh(
    data: dict[str, Any],
    *,
    kpoints: orm.KpointsData | None,
    mp_grid: list[int] | None,
    scf_kpoints: orm.KpointsData | None,
) -> None:
    """Substitute the caller's Brillouin-zone sampling into a flattened builder.

    ``Wannier90WorkChain.get_builder_from_protocol`` derives one mesh from
    the protocol's ``kpoints_distance`` and uses it for the wannier90 k-list,
    the ``mp_grid`` in the ``.win`` and the nscf; the scf takes the distance
    itself. Each is replaced here when the caller states one, and left to the
    protocol when it does not.

    ``kpoints`` is the explicit k-list wannier90 and the nscf share, so that
    a single node fixes the k-ordering for both. ``mp_grid`` travels beside
    it because an explicit-list ``KpointsData`` cannot represent its parent
    mesh and wannier90 cannot re-derive it.
    """
    if kpoints is not None and mp_grid is None:
        raise ValueError(
            "`kpoints` was given without `mp_grid`: an explicit k-list cannot "
            "state the Monkhorst-Pack dimensions wannier90 requires in the "
            "`.win`. Pass the mesh dimensions the list was generated from."
        )

    if scf_kpoints is not None:
        data["scf"].pop("kpoints_distance", None)
        data["scf"]["kpoints"] = scf_kpoints

    if kpoints is None:
        return

    data["nscf"].pop("kpoints_distance", None)
    data["nscf"]["kpoints"] = kpoints

    w90 = data["wannier90"]["wannier90"]
    w90["kpoints"] = kpoints
    parameters = w90["parameters"].get_dict()
    parameters["mp_grid"] = list(mp_grid)  # type: ignore[arg-type]
    w90["parameters"] = orm.Dict(parameters)


@task.graph
def Wannierize(
    codes: WannierizeCodes,
    structure: orm.StructureData,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    pseudo_family: str | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
    projection_type: WannierProjectionType = WannierProjectionType.ATOMIC_PROJECTORS_QE,
    disentanglement_type: WannierDisentanglementType | None = None,
    frozen_type: WannierFrozenType | None = None,
    only_valence: bool = False,
    exclude_semicore: bool = False,
    external_projectors_path: str | None = None,
    external_projectors: dict[str, Any] | None = None,
    plot_wannier_functions: bool = False,
    retrieve_hamiltonian: bool = False,
    retrieve_matrices: bool = False,
    print_summary: bool = False,
    kpoint_path: dict[str, Any] | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    projector_rotation: np.ndarray | None = None,
    parallelization: ParallelizationDict | None = None,
    kpoints: orm.KpointsData | None = None,
    mp_grid: list[int] | None = None,
    scf_kpoints: orm.KpointsData | None = None,
) -> WannierWorkflowOutputs:
    """Run Wannier90WorkChain using the protocol-based builder pattern.

    If ``projector_rotation`` is provided, the workchain will apply
    ``A' = B @ A`` to the pw2wannier90 projection matrix before
    wannier90 reads it.

    This task wraps Wannier90WorkChain and uses get_builder_from_protocol to
    construct the inputs from a simplified set of arguments.

    Args:
        codes: Dictionary mapping code names to Code instances. Required keys:
            'pw', 'pw2wannier90', 'wannier90'. Optional: 'projwfc'.
        structure: The StructureData instance to use.
        protocol: Protocol to use. If not specified, the default will be used.
        overrides: Optional dictionary of inputs to override protocol defaults.
        pseudo_family: Pseudopotential family to use. If not specified,
            defaults based on spin_type.
        electronic_type: Electronic type - "metal" or "insulator".
        spin_type: Spin type - "none", "collinear", "non_collinear", or "spin_orbit".
        projection_type: Wannier projection type - "scdm", "analytic", "random",
            "atomic_projectors_qe", "atomic_projectors_openmx", or
            "atomic_projectors_external".
        disentanglement_type: Wannier disentanglement type - "none" or "smv".
            If None, chosen automatically based on projection_type.
        frozen_type: Wannier frozen window type. If None, chosen automatically.
        exclude_semicore: If True, exclude semicore states from Wannierisation.
        external_projectors_path: Path to directory containing external projector
            files. Required when projection_type is ATOMIC_PROJECTORS_EXTERNAL.
        external_projectors: Per-element orbital tables describing the external
            projectors, passed through to the upstream builder. Required when
            projection_type is ATOMIC_PROJECTORS_EXTERNAL.
        plot_wannier_functions: If True, plot Wannier functions as xsf files.
        retrieve_hamiltonian: If True, retrieve Wannier Hamiltonian.
        retrieve_matrices: If True, retrieve amn/mmn/eig/chk/spin files.
        print_summary: If True, print a summary of key input parameters.
        kpoints: the explicit k-point list the nscf and wannier90 share.
            Unset leaves both on the protocol's ``kpoints_distance``-derived
            mesh. Requires ``mp_grid``.
        mp_grid: the Monkhorst-Pack dimensions ``kpoints`` was generated
            from, written into the ``.win``.
        scf_kpoints: the mesh the scf samples. Unset falls back to the
            protocol's ``kpoints_distance``.

    Returns:
        Dict with outputs from the Wannier90WorkChain.
    """
    validate_parallelization(parallelization)

    if exclude_semicore and projection_type == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL:
        raise ValueError(
            "`exclude_semicore` is not supported with external projectors: the "
            "upstream semicore selection reads a per-orbital `label` the "
            "synthesized projector tables do not carry."
        )

    # A graph input arrives as a wrapt proxy; coerce to a plain str so the
    # protocol builder's pseudo-family QueryBuilder can bind it, and to
    # genuine enum members for the two enums this builder forwards into
    # ``PwBaseWorkChain``, whose branches test them with ``is``.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None

    builder = Wannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        protocol=protocol,
        overrides=overrides or {},
        pseudo_family=pseudo_family,
        electronic_type=unwrap_enum(electronic_type, ElectronicType),
        spin_type=unwrap_enum(spin_type, SpinType),
        projection_type=projection_type,
        disentanglement_type=disentanglement_type,
        frozen_type=frozen_type,
        exclude_semicore=exclude_semicore,
        only_valence=only_valence,
        external_projectors_path=external_projectors_path,
        external_projectors=external_projectors,
        plot_wannier_functions=plot_wannier_functions,
        retrieve_hamiltonian=retrieve_hamiltonian,
        retrieve_matrices=retrieve_matrices,
        print_summary=print_summary,
    )

    data = _finalize_wannier_builder(
        builder,
        kpoint_path=kpoint_path,
        bands_kpoints=bands_kpoints,
        projector_rotation=projector_rotation,
        set_bands_kpoints=True,
    )

    _apply_kpoint_mesh(data, kpoints=kpoints, mp_grid=mp_grid, scf_kpoints=scf_kpoints)

    # Per-code parallelization into whichever calcjob namespaces this run has.
    merge_parallelization_into_existing_namespaces(
        data,
        parallelization,
        [
            (("scf", "pw"), "pw"),
            (("nscf", "pw"), "pw"),
            (("wannier90", "wannier90"), "wannier90"),
            (("pw2wannier90", "pw2wannier90"), "pw2wannier90"),
            (("projwfc", "projwfc"), "projwfc"),
        ],
    )

    # Submit the workchain with converted inputs
    outputs = Wannier90Step(**data)

    # Return available outputs
    return WannierWorkflowOutputs(
        scf=outputs.scf,
        nscf=outputs.nscf,
        wannier90=outputs.wannier90,
        wannier90_up=outputs.wannier90_up,
        wannier90_down=outputs.wannier90_down,
        projwfc=outputs.projwfc,
    )


class WannierOptimizeOutputs(TypedDict, total=False):
    """Output types for Wannier90 optimize workgraph tasks."""

    scf: PwOutputs
    nscf: PwOutputs
    wannier90: Wannier90Outputs
    wannier90_up: Wannier90Outputs
    wannier90_down: Wannier90Outputs
    wannier90_optimal: Wannier90Outputs
    wannier90_optimal_up: Wannier90Outputs
    wannier90_optimal_down: Wannier90Outputs
    projwfc: ProjwfcOutputs
    bands_distance: float


@task.graph
def OptimizeWannierization(
    codes: WannierizeCodes,
    structure: orm.StructureData,
    reference_bands: orm.BandsData | None = None,
    bands_distance_threshold: float = 1e-2,
    optimize_strategy: OptimizeStrategy = OptimizeStrategy.GRID,
    optimize_metric: OptimizeMetric = OptimizeMetric.FERMI_DIRAC,
    optimize_max_iterations: int | None = None,
    optimize_disprojmax_range: list[float] | None = None,
    optimize_disprojmin_range: list[float] | None = None,
    optimize_mu_shift: float = 2.0,
    optimize_sigma: float = 0.1,
    optimize_mu_reference: OptimizeMuReference = OptimizeMuReference.FERMI_ENERGY,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    pseudo_family: str | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
    projection_type: WannierProjectionType = WannierProjectionType.ATOMIC_PROJECTORS_QE,
    disentanglement_type: WannierDisentanglementType | None = None,
    frozen_type: WannierFrozenType | None = None,
    only_valence: bool = False,
    exclude_semicore: bool = False,
    external_projectors_path: str | None = None,
    external_projectors: dict[str, Any] | None = None,
    plot_wannier_functions: bool = False,
    retrieve_hamiltonian: bool = False,
    retrieve_matrices: bool = False,
    print_summary: bool = False,
    kpoint_path: dict[str, Any] | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    projector_rotation: np.ndarray | None = None,
) -> WannierOptimizeOutputs:
    """Run Wannier90OptimizeWorkChain using the protocol-based builder pattern.

    Wraps Wannier90OptimizeWorkChain to optimize dis_proj_min/max for
    projectability disentanglement, using either grid search or Bayesian
    optimization.

    Args:
        codes: Dictionary mapping code names to Code instances.
        structure: The StructureData instance to use.
        reference_bands: DFT reference bands for computing bands distance.
            Required for Bayesian strategy.
        bands_distance_threshold: Stop optimization when bands distance
            drops below this threshold (eV).
        optimize_strategy: Search strategy - GRID or BAYESIAN.
        optimize_metric: Metric for evaluating band quality -
            FERMI_DIRAC_EF2 or UNWEIGHTED_RMS.
        optimize_max_iterations: Maximum iterations for Bayesian strategy.
        protocol: Protocol to use. If not specified, the default will be used.
        overrides: Optional dictionary of inputs to override protocol defaults.
        pseudo_family: Pseudopotential family to use.
        electronic_type: Electronic type - "metal" or "insulator".
        spin_type: Spin type.
        projection_type: Wannier projection type.
        disentanglement_type: Wannier disentanglement type.
        frozen_type: Wannier frozen window type.
        exclude_semicore: If True, exclude semicore states.
        external_projectors_path: Path to external projector files.
        external_projectors: Dictionary describing external projectors.
        plot_wannier_functions: If True, plot Wannier functions.
        retrieve_hamiltonian: If True, retrieve Wannier Hamiltonian.
        retrieve_matrices: If True, retrieve amn/mmn/eig/chk/spin files.
        print_summary: If True, print a summary of key input parameters.
        kpoint_path: Explicit k-point path dictionary.
        bands_kpoints: Explicit k-point path as KpointsData.

    Returns:
        Dict with outputs including optimal Wannier90 results and bands_distance.
    """
    # A graph input arrives as a wrapt proxy; coerce to a plain str so the
    # protocol builder's pseudo-family QueryBuilder can bind it, and to
    # genuine enum members for the two enums this builder forwards into
    # ``PwBaseWorkChain``, whose branches test them with ``is``.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None

    builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        reference_bands=reference_bands,
        bands_distance_threshold=bands_distance_threshold,
        optimize_strategy=optimize_strategy,
        optimize_metric=optimize_metric,
        optimize_max_iterations=optimize_max_iterations,
        protocol=protocol,
        overrides=overrides or {},
        pseudo_family=pseudo_family,
        electronic_type=unwrap_enum(electronic_type, ElectronicType),
        spin_type=unwrap_enum(spin_type, SpinType),
        projection_type=projection_type,
        disentanglement_type=disentanglement_type,
        frozen_type=frozen_type,
        exclude_semicore=exclude_semicore,
        only_valence=only_valence,
        external_projectors_path=external_projectors_path,
        external_projectors=external_projectors,
        plot_wannier_functions=plot_wannier_functions,
        retrieve_hamiltonian=retrieve_hamiltonian,
        retrieve_matrices=retrieve_matrices,
        print_summary=print_summary,
        bands_kpoints=bands_kpoints,
    )

    if optimize_disprojmax_range is not None:
        builder.optimize_disprojmax_range = optimize_disprojmax_range
    if optimize_disprojmin_range is not None:
        builder.optimize_disprojmin_range = optimize_disprojmin_range

    builder.optimize_mu_shift = optimize_mu_shift
    builder.optimize_sigma = optimize_sigma
    # ``to_aiida_type`` maps ``Enum -> EnumData``, but the port wants ``orm.Str``;
    # extract ``.value`` so the default serializer wraps a plain str into ``orm.Str``.
    builder.optimize_mu_reference = optimize_mu_reference.value

    # ``bands_kpoints`` is already wired through ``get_builder_from_protocol``
    # above, so the finaliser only needs it for the mutual-exclusion check.
    data = _finalize_wannier_builder(
        builder,
        kpoint_path=kpoint_path,
        bands_kpoints=bands_kpoints,
        projector_rotation=projector_rotation,
        set_bands_kpoints=False,
    )

    outputs = Wannier90OptimizeStep(**data)

    return WannierOptimizeOutputs(
        scf=outputs.scf,
        nscf=outputs.nscf,
        wannier90=outputs.wannier90,
        wannier90_up=outputs.wannier90_up,
        wannier90_down=outputs.wannier90_down,
        wannier90_optimal=outputs.wannier90_optimal,
        wannier90_optimal_up=outputs.wannier90_optimal_up,
        wannier90_optimal_down=outputs.wannier90_optimal_down,
        projwfc=outputs.projwfc,
        bands_distance=outputs.bands_distance,
    )
