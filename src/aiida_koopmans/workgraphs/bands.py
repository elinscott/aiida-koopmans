"""SCF-NSCF-Bands workgraph using aiida-workgraph.

This module provides a WorkGraph that wraps the standard SCF -> NSCF -> Bands
calculation sequence using aiida-quantumespresso's PwBaseWorkChain.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict, Any

from aiida import orm
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_workgraph import task
from aiida_workgraph.utils import get_dict_from_builder


@task.calcfunction
def prepare_nscf_parameters(scf_parameters: orm.Dict) -> orm.Dict:
    """Prepare NSCF parameters based on SCF output.

    Takes the SCF output parameters and modifies them for an NSCF calculation
    (sets calculation='nscf' and removes incompatible settings).
    """
    params = scf_parameters.get_dict()

    # Modify for NSCF calculation
    if "CONTROL" not in params:
        params["CONTROL"] = {}
    params["CONTROL"]["calculation"] = "nscf"

    # NSCF should not use smearing (if present, set to fixed occupations)
    if "SYSTEM" in params and "occupations" in params["SYSTEM"]:
        if params["SYSTEM"]["occupations"] == "smearing":
            params["SYSTEM"]["occupations"] = "fixed"
            params["SYSTEM"].pop("smearing", None)
            params["SYSTEM"].pop("degauss", None)

    return orm.Dict(params)


@task.calcfunction
def prepare_bands_parameters(scf_parameters: orm.Dict) -> orm.Dict:
    """Prepare bands parameters based on SCF output.

    Takes the SCF output parameters and modifies them for a bands calculation
    (sets calculation='bands').
    """
    params = scf_parameters.get_dict()

    # Modify for bands calculation
    if "CONTROL" not in params:
        params["CONTROL"] = {}
    params["CONTROL"]["calculation"] = "bands"

    # Bands calculation should use fixed occupations
    if "SYSTEM" in params:
        params["SYSTEM"]["occupations"] = "fixed"
        params["SYSTEM"].pop("smearing", None)
        params["SYSTEM"].pop("degauss", None)

    return orm.Dict(params)

class ScfBandsOutputs(TypedDict):
    scf_parameters: orm.Dict
    band_structure: orm.BandsData


def _builder_to_dict(builder) -> dict:
    """Recursively convert a ProcessBuilder/ProcessBuilderNamespace to a plain dict.

    This is needed because workgraph tasks expect plain dicts, not ProcessBuilder objects.
    """
    from aiida.engine import ProcessBuilderNamespace

    result = {}
    for key in builder:
        value = builder[key]
        if isinstance(value, ProcessBuilderNamespace):
            # Recursively convert nested namespaces
            result[key] = _builder_to_dict(value)
        else:
            result[key] = value
    return result


# Create the task from PwBandsWorkChain at module level
PwBandsTask = task(PwBandsWorkChain)

# @task
# def PwBandsBuilder(
#     code: orm.AbstractCode,
#     structure: orm.StructureData,
#     protocol: str | None = None,
#     overrides: dict[str, Any] | None = None,
#     options: dict[str, Any] | None = None,
# ) -> dict[str, Any]: # Need a proper annotation here to expose the outputs of this task
#     """Build inputs for PwBandsWorkChain using the protocol-based builder pattern."""
# 
#     builder = PwBandsWorkChain.get_builder_from_protocol(
#         code=code,
#         structure=structure,
#         protocol=protocol,
#         overrides=overrides or {},
#         options=options or {},
#     )
# 
#     # Convert builder to plain dict (ProcessBuilderNamespace objects don't work with workgraph)
#     return _builder_to_dict(builder)

@task.graph
def PwBandsTaskViaBuilder(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> ScfBandsOutputs:
    """Run PwBandsWorkChain using the protocol-based builder pattern.

    This task wraps PwBandsWorkChain and uses get_builder_from_protocol to
    construct the inputs from a simplified set of arguments (code, structure,
    protocol, overrides, options).

    Args:
        code: The Code instance configured for the quantumespresso.pw plugin.
        structure: The StructureData instance to use.
        protocol: Protocol to use. If not specified, the default will be used.
        overrides: Optional dictionary of inputs to override protocol defaults.
        options: Dictionary of options for metadata.options of nested CalcJobs.

    Returns:
        Dict with scf_parameters and band_structure outputs.
    """

    builder = PwBandsWorkChain.get_builder_from_protocol(
        code=code,
        structure=structure,
        protocol=protocol,
        overrides=overrides or {},
        options=options or {},
    )

    data = get_dict_from_builder(builder)

    # Submit the workchain with converted inputs
    output = PwBandsTask(**data)

    return ScfBandsOutputs(
        scf_parameters=output.scf_parameters,
        band_structure=output.band_structure,
    )


