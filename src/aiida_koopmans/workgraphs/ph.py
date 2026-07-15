"""Dielectric-constant workgraph wrapping aiida-quantumespresso's ph.x workchain.

One scf, then ph.x with an electric-field perturbation only
(``epsil = .true.``, ``trans = .false.``) at q = Gamma. The macroscopic
dielectric tensor is parsed upstream by aiida-quantumespresso's ``PhParser``
(from ``tensors.xml`` into ``output_parameters["dielectric_constant"]``);
its isotropic average (``tr(eps)/3``) is exposed as ``eps_inf`` — the value
the Koopmans DFPT screen step and the Gygi-Baldereschi / Makov-Payne
corrections consume.
"""

from __future__ import annotations

from typing import Any, TypedDict

from aiida import orm
from aiida_quantumespresso.workflows.ph.base import PhBaseWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_workgraph import task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.workgraphs.pw import PwBaseStep


class DielectricConstant(TypedDict):
    """The macroscopic dielectric tensor and its isotropic average."""

    eps_inf: float
    dielectric_tensor: list[list[float]]


class DielectricOutputs(DielectricConstant):
    """Outputs of a DielectricTask run (scf + ph.x with epsil)."""

    ph_output_parameters: dict


PhBaseTask = task(PhBaseWorkChain)


@task
def extract_dielectric_constant(ph_parameters: dict) -> DielectricConstant:
    """Extract the dielectric tensor and its isotropic average from ph.x output.

    ``eps_inf`` is the mean of the diagonal of the macroscopic dielectric
    tensor (``np.trace(tensor) / 3``). The
    image-correction schemes assume a uniform dielectric, so only this
    scalar average is consumed downstream; the full tensor is exposed
    alongside it for inspection.
    """
    try:
        tensor = ph_parameters["dielectric_constant"]
    except KeyError as exc:
        raise ValueError(
            "ph.x output_parameters carry no 'dielectric_constant': the run must set "
            "INPUTPH epsil = .true. (and the system must be an insulator)."
        ) from exc
    eps_inf = sum(tensor[i][i] for i in range(3)) / 3.0
    return DielectricConstant(eps_inf=eps_inf, dielectric_tensor=tensor)


@task.graph
def DielectricTask(
    pw_code: orm.AbstractCode,
    ph_code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
) -> DielectricOutputs:
    """Compute the macroscopic dielectric tensor: scf, then ph.x with epsil.

    Chains a ``PwBaseWorkChain`` ground state into a ``PhBaseWorkChain``
    restricted to the electric-field perturbation (``epsil = .true.``,
    ``trans = .false.``) at q = Gamma — the dielectric tensor is a q = 0
    response, so the phonon protocol's q-mesh is bypassed.

    Args:
        pw_code: Code configured for the quantumespresso.pw plugin.
        ph_code: Code configured for the quantumespresso.ph plugin.
        structure: The StructureData instance to use.
        pseudo_family: Pseudo family label. If not specified, the protocol
            default is used.
        protocol: Protocol to use for both steps. If not specified, the
            default will be used.
        overrides: Optional dictionary with ``"scf"`` (PwBaseWorkChain
            overrides) and/or ``"ph"`` (PhBaseWorkChain overrides) keys.
            The epsil / trans / Gamma-q keys are forced on top of any
            caller-supplied ph overrides — they are what makes this a
            dielectric run.
        options: Dictionary of options for metadata.options of nested CalcJobs.

    Returns:
        Dict with the scalar ``eps_inf`` (isotropic average), the full
        3x3 ``dielectric_tensor``, and the raw ph.x output parameters.
    """
    from aiida_quantumespresso.workflows.protocols.utils import recursive_merge

    overrides = overrides or {}

    # ``.build()`` executes this body eagerly, where graph inputs arrive as
    # provenance-tagged proxies; the family label ends up bound as an SQL
    # parameter inside ``get_builder_from_protocol``, which needs a plain str.
    if pseudo_family is not None:
        pseudo_family = str(pseudo_family)

    scf_overrides = overrides.get("scf", {})
    if pseudo_family is not None:
        scf_overrides.setdefault("pseudo_family", pseudo_family)

    scf_builder = PwBaseWorkChain.get_builder_from_protocol(
        code=pw_code,
        structure=structure,
        protocol=protocol,
        overrides=scf_overrides,
        options=options or {},
    )
    scf_builder.pop("clean_workdir", None)
    scf_data = get_dict_from_builder(scf_builder)
    scf_data.setdefault("metadata", {})["call_link_label"] = "scf"
    scf_outputs = PwBaseStep(**scf_data)

    ph_defaults: dict[str, Any] = {
        "qpoints": [1, 1, 1],
        "ph": {"parameters": {"INPUTPH": {"epsil": True, "trans": False}}},
    }
    ph_overrides = recursive_merge(overrides.get("ph", {}), ph_defaults)

    ph_builder = PhBaseWorkChain.get_builder_from_protocol(
        code=ph_code,
        protocol=protocol,
        overrides=ph_overrides,
        options=options or {},
    )
    ph_builder.pop("clean_workdir", None)
    ph_data = get_dict_from_builder(ph_builder)

    # Wire SCF remote_folder → ph.x parent_folder
    ph_data["ph"]["parent_folder"] = scf_outputs["remote_folder"]
    ph_data.setdefault("metadata", {})["call_link_label"] = "ph"
    ph_outputs = PhBaseTask(**ph_data)

    extracted = extract_dielectric_constant(
        ph_parameters=ph_outputs["output_parameters"],
        metadata={"call_link_label": "extract_dielectric_constant"},
    )

    return DielectricOutputs(
        eps_inf=extracted["eps_inf"],
        dielectric_tensor=extracted["dielectric_tensor"],
        ph_output_parameters=ph_outputs["output_parameters"],
    )
