"""Workgraph that wraps aiida-quantumespresso PdosWorkChain."""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from aiida import orm
from aiida_quantumespresso.workflows.pdos import PdosWorkChain
from aiida_workgraph import task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.workgraphs import Codes, inject_pseudo_family


class PdosOutputs(TypedDict):
    """Outputs of a PdosWorkChain run (NSCF + DOS + projwfc)."""

    nscf_remote_folder: orm.RemoteData
    nscf_output_parameters: dict
    nscf_output_band: orm.BandsData
    dos_output_dos: orm.XyData
    projwfc_projections: NotRequired[orm.ProjectionData]
    projwfc_projections_up: NotRequired[orm.ProjectionData]
    projwfc_projections_down: NotRequired[orm.ProjectionData]
    projwfc_Pdos: NotRequired[orm.XyData]


PdosStep = task(PdosWorkChain)


@task.graph
def RunPdos(
    codes: Codes,
    structure: orm.StructureData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> PdosOutputs:
    """Run PdosWorkChain using the protocol-based builder pattern.

    This task wraps PdosWorkChain (SCF + NSCF + DOS + PROJWFC) and uses
    get_builder_from_protocol to construct the inputs from a simplified
    set of arguments.

    Args:
        codes: Dict with ``pw``, ``dos``, and ``projwfc`` code instances.
        structure: The StructureData instance to use.
        pseudo_family: Pseudo family label. If not specified, protocol default is used.
        protocol: Protocol to use. If not specified, the default will be used.
        overrides: Optional dictionary of inputs to override protocol defaults.
        options: Dictionary of options for metadata.options of nested CalcJobs.

    Returns:
        Dict with NSCF, DOS, and PROJWFC outputs.
    """
    overrides = overrides or {}

    # Inject pseudo_family into scf and nscf overrides
    inject_pseudo_family(overrides, pseudo_family, ("scf", "nscf"))

    builder = PdosWorkChain.get_builder_from_protocol(
        pw_code=codes["pw"],
        dos_code=codes["dos"],
        projwfc_code=codes["projwfc"],
        structure=structure,
        protocol=protocol,
        overrides=overrides,
        options=options or {},
    )

    data = get_dict_from_builder(builder)

    output = PdosStep(**data)

    return PdosOutputs(
        nscf_remote_folder=output.nscf__remote_folder,
        nscf_output_parameters=output.nscf__output_parameters,
        nscf_output_band=output.nscf__output_band,
        dos_output_dos=output.dos__output_dos,
        projwfc_projections=output.projwfc__projections,
        projwfc_projections_up=output.projwfc__projections_up,
        projwfc_projections_down=output.projwfc__projections_down,
        projwfc_Pdos=output.projwfc__Pdos,
    )
