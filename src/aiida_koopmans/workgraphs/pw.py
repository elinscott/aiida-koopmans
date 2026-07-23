"""Workgraphs that wrap aiida-quantumespresso.pw workchains."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_workgraph import task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.types import ParallelizationDict
from aiida_koopmans.workgraphs import (
    inject_pseudo_family,
    merge_parallelization_into_overrides,
    validate_parallelization,
)


class PwOutputs(TypedDict, total=False):
    """Outputs of a single PwBaseWorkChain run."""

    remote_folder: orm.RemoteData
    remote_stash: orm.RemoteData
    retrieved: orm.FolderData
    output_parameters: dict
    output_structure: orm.StructureData
    output_band: orm.BandsData
    output_atomic_occupations: dict
    output_kpoints: orm.KpointsData
    output_trajectory: orm.TrajectoryData


class ScfBandsOutputs(TypedDict):
    """Outputs of a PwBandsWorkChain run (SCF + bands)."""

    scf_parameters: dict
    band_structure: orm.BandsData


class ScfNscfOutputs(TypedDict):
    """Outputs of a chained SCF + NSCF PwBaseWorkChain run."""

    scf_remote_folder: orm.RemoteData
    nscf_remote_folder: orm.RemoteData
    nscf_retrieved: orm.FolderData
    nscf_output_parameters: dict
    nscf_output_band: orm.BandsData
    nscf_output_kpoints: NotRequired[orm.KpointsData]


PwBaseStep = task(PwBaseWorkChain)
PwBandsStep = task(PwBandsWorkChain)


@task.graph
def RunPwBands(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    parallelization: ParallelizationDict | None = None,
    bands_kpoints: orm.KpointsData | None = None,
) -> ScfBandsOutputs:
    """Run PwBandsWorkChain using the protocol-based builder pattern.

    This task wraps PwBandsWorkChain and uses get_builder_from_protocol to
    construct the inputs from a simplified set of arguments (code, structure,
    protocol, overrides, parallelization).

    Args:
        code: The Code instance configured for the quantumespresso.pw plugin.
        structure: The StructureData instance to use.
        pseudo_family: Pseudo family label (e.g. ``"PseudoDojo/0.4/PBE/SR/standard/upf"``).
            If not specified, the protocol default is used.
        protocol: Protocol to use. If not specified, the default will be used.
        overrides: Optional dictionary of inputs to override protocol defaults.
        parallelization: Per-code parallelization mapping (keyed by code name);
            the ``pw`` entry sets the scf/bands pw.x ``metadata.options`` and
            ``-npool``.
        bands_kpoints: Explicit KpointsData for the bands path. If provided,
            seekpath is bypassed entirely.

    Returns:
        Dict with scf_parameters and band_structure outputs.
    """
    validate_parallelization(parallelization)

    overrides = overrides or {}

    # Inject pseudo_family into both scf and bands overrides
    inject_pseudo_family(overrides, pseudo_family, ("scf", "bands"))
    merge_parallelization_into_overrides(
        overrides, parallelization, [(("scf", "pw"), "pw"), (("bands", "pw"), "pw")]
    )

    builder = PwBandsWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
    )

    data = get_dict_from_builder(builder)

    # If nbnd is explicitly set, remove nbands_factor to avoid conflict
    bands_system = overrides.get("bands", {}).get("pw", {}).get("parameters", {}).get("SYSTEM", {})
    if "nbnd" in bands_system:
        data.pop("nbands_factor", None)

    # Inject explicit bands_kpoints to bypass seekpath
    if bands_kpoints is not None:
        data.pop("bands_kpoints_distance", None)
        data["bands_kpoints"] = bands_kpoints

    # Submit the workchain with converted inputs
    output = PwBandsStep(**data)

    return ScfBandsOutputs(
        scf_parameters=output.scf_parameters,
        band_structure=output.band_structure,
    )


@task.graph
def RunScfNscf(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    parallelization: ParallelizationDict | None = None,
    nscf_kpoints: orm.KpointsData | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
) -> ScfNscfOutputs:
    """Run SCF + NSCF using two PwBaseWorkChain steps.

    The SCF step uses protocol defaults. The NSCF step reuses the SCF
    charge density via ``parent_folder`` and sets ``calculation = 'nscf'``.

    Overrides are split by namespace: ``overrides["scf"]`` applies to the
    SCF step and ``overrides["nscf"]`` applies to the NSCF step.

    Args:
        code: The Code instance configured for the quantumespresso.pw plugin.
        structure: The StructureData instance to use.
        pseudo_family: Pseudo family label (e.g. ``"PseudoDojo/0.4/PBE/SR/standard/upf"``).
            If not specified, the protocol default is used.
        protocol: Protocol to use. If not specified, the default will be used.
        overrides: Optional dictionary with ``"scf"`` and/or ``"nscf"`` keys.
        parallelization: Per-code parallelization mapping (keyed by code name);
            the ``pw`` entry sets the scf/nscf pw.x ``metadata.options`` and
            ``-npool``.
        nscf_kpoints: Explicit k-points for the NSCF step, replacing the
            protocol's ``kpoints_distance``. A wannierisation NSCF must run
            on the full (symmetry-unreduced) grid in the k-point order the
            downstream wannier90 expects.
        electronic_type: Defaults to ``INSULATOR`` (fixed occupations):
            Koopmans functionals treat insulators exclusively, and kcw.x
            refuses non-fixed occupations outright.

    Returns:
        Dict with remote folders and retrieved data from both steps.
    """
    from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

    overrides = overrides or {}

    # Inject pseudo_family as a top-level override for both steps
    inject_pseudo_family(overrides, pseudo_family, ("scf", "nscf"))
    merge_parallelization_into_overrides(
        overrides, parallelization, [(("scf", "pw"), "pw"), (("nscf", "pw"), "pw")]
    )
    scf_overrides = overrides.get("scf", {})

    # --- SCF builder ---
    scf_builder = PwBaseWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol=protocol,
        overrides=scf_overrides,
        electronic_type=electronic_type,
    )
    scf_data = get_dict_from_builder(scf_builder)
    scf_data.pop("clean_workdir", None)
    scf_data.setdefault("metadata", {})["call_link_label"] = "scf"
    scf_outputs = PwBaseStep(**scf_data)

    # --- NSCF builder ---
    # Start from protocol defaults, then merge NSCF-specific overrides
    # (pseudo_family already seeded above).
    nscf_overrides = overrides.get("nscf", {})

    # Ensure calculation type is nscf
    nscf_defaults: dict[str, Any] = {
        "pw": {
            "parameters": {
                "CONTROL": {"calculation": "nscf"},
            },
        },
    }
    nscf_merged = recursive_merge(nscf_defaults, nscf_overrides)

    nscf_builder = PwBaseWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol=protocol,
        overrides=nscf_merged,
        electronic_type=electronic_type,
    )
    nscf_data = get_dict_from_builder(nscf_builder)
    nscf_data.pop("clean_workdir", None)

    # The workchain accepts exactly one of ``kpoints`` / ``kpoints_distance``:
    # replace the protocol's distance-derived mesh with the caller's grid.
    if nscf_kpoints is not None:
        nscf_data.pop("kpoints_distance", None)
        nscf_data.pop("kpoints_force_parity", None)
        nscf_data["kpoints"] = nscf_kpoints

    # Explicit NSCF k-mesh: PwBaseWorkChain accepts exactly one of
    # ``kpoints`` / ``kpoints_distance``, so drop the protocol's distance
    # (and its companion parity flag) before setting the mesh.
    if nscf_kpoints is not None:
        nscf_data.pop("kpoints_distance", None)
        nscf_data.pop("kpoints_force_parity", None)
        nscf_data["kpoints"] = nscf_kpoints

    # Wire SCF remote_folder → NSCF parent_folder
    nscf_data["pw"]["parent_folder"] = scf_outputs["remote_folder"]

    nscf_data.setdefault("metadata", {})["call_link_label"] = "nscf"
    nscf_outputs = PwBaseStep(**nscf_data)

    return ScfNscfOutputs(
        scf_remote_folder=scf_outputs["remote_folder"],
        nscf_remote_folder=nscf_outputs["remote_folder"],
        nscf_retrieved=nscf_outputs["retrieved"],
        nscf_output_parameters=nscf_outputs["output_parameters"],
        nscf_output_band=nscf_outputs["output_band"],
        nscf_output_kpoints=nscf_outputs["output_kpoints"],
    )
