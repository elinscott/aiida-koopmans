"""Workgraphs that wrap aiida-quantumespresso.pw workchains."""

import copy
from typing import Annotated, Any, NotRequired, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_workgraph import task
from aiida_workgraph.socket_spec import SocketMeta
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.parallelization import (
    ParallelizationDict,
    merge_parallelization_into_inputs,
    merge_parallelization_into_overrides,
    validate_parallelization,
)
from aiida_koopmans.workgraphs import (
    enforce_step_calculation,
    force_pw_verbosity,
    inject_pseudo_family,
    name_step,
    pin_kpoints,
    unwrap_enum,
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


#: Annotation for the pw.x code as the workflows that run scf + nscf wire it.
PwCode = Annotated[
    orm.AbstractCode,
    SocketMeta(help="Needed to compute DFT ground state properties."),
]


class PwBandsCodes(TypedDict):
    """Codes for :func:`RunPwBands`."""

    pw: Annotated[
        orm.AbstractCode,
        SocketMeta(help="Needed to compute the DFT ground state and band structure."),
    ]


PwBaseStep = task(PwBaseWorkChain)
PwBandsStep = task(PwBandsWorkChain)


def _finish_pw_base_step(
    builder: Any,
    *,
    step: str,
    display: str,
    kpoints: orm.KpointsData | None = None,
    parent_folder: Any = None,
    parallelization: ParallelizationDict | None = None,
) -> dict[str, Any]:
    """Turn a built ``PwBaseWorkChain`` builder into one step's task inputs.

    The post-build half every pw.x step of this package shares: drop
    ``clean_workdir`` (the graph owns the scratch), stamp
    ``CONTROL.verbosity``, replace the protocol's distance-derived mesh with
    ``kpoints``, wire ``parent_folder`` and the per-code parallelization,
    and name the step for a reader. The step's ``CONTROL.calculation``
    belongs to the *overrides* the builder was built from, where
    :func:`enforce_step_calculation` can tell a caller's explicit value from
    the protocol's default.

    ``display`` names the step on both the workchain and the pw.x
    calculation it wraps, so the step is named whichever of the two a
    restart leaves visible.

    Args:
        builder: A ``PwBaseWorkChain`` builder, or its flattened inputs.
        step: The step's ``call_link_label``.
        display: The name a reader sees.
        kpoints: The mesh (or explicit list) the step samples.
        parent_folder: The scratch the step restarts from.
        parallelization: Per-code parallelization mapping.

    Returns:
        The step's inputs, ready to pass to ``PwBaseStep``.
    """
    data = builder if isinstance(builder, dict) else get_dict_from_builder(builder)
    data.pop("clean_workdir", None)
    pw_inputs = data["pw"]
    force_pw_verbosity(pw_inputs)
    pin_kpoints(data, kpoints)
    if parent_folder is not None:
        pw_inputs["parent_folder"] = parent_folder
    if parallelization is not None:
        merge_parallelization_into_inputs(pw_inputs, parallelization, "pw")
    data.setdefault("metadata", {})["call_link_label"] = step
    name_step(data, display)
    name_step(pw_inputs, display)
    return data


def assemble_pw_base_step(
    pw_code: orm.AbstractCode,
    structure: orm.StructureData,
    *,
    calculation: str,
    call_link_label: str,
    display: str,
    overrides: dict[str, Any] | None = None,
    protocol: str | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    kpoints: orm.KpointsData | None = None,
    parent_folder: Any = None,
    parallelization: ParallelizationDict | None = None,
) -> Any:
    """Assemble one ``PwBaseWorkChain`` step inside a graph body.

    Stamp the step's ``CONTROL.calculation`` into ``overrides`` (raising on
    a conflicting explicit value), build the step from the protocol builder
    with those overrides merged on top, finish it with
    :func:`_finish_pw_base_step`, and add it to the surrounding graph under
    ``call_link_label``. A plain graph-assembly helper: it must be called
    inside a ``@task.graph`` body.
    """
    overrides = overrides or {}
    enforce_step_calculation(
        overrides.setdefault("pw", {}).setdefault("parameters", {}),
        call_link_label,
        calculation,
    )
    builder = PwBaseWorkChain.get_builder_from_protocol(
        code=pw_code,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
        electronic_type=unwrap_enum(electronic_type, ElectronicType),
    )
    data = _finish_pw_base_step(
        builder,
        step=call_link_label,
        display=display,
        kpoints=kpoints,
        parent_folder=parent_folder,
        parallelization=parallelization,
    )
    return PwBaseStep(**data)


def run_bands_step(
    pw_code: orm.AbstractCode,
    structure: orm.StructureData,
    bands_kpoints: orm.KpointsData,
    scf_remote_folder: orm.RemoteData,
    nscf_overrides: dict[str, Any] | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    parallelization: ParallelizationDict | None = None,
) -> Any:
    """Assemble a pw.x ``bands`` step along ``bands_kpoints`` off an scf density.

    A plain graph-assembly helper, not a task: it must be called inside a
    ``@task.graph`` body, where the ``PwBaseStep`` it creates joins the
    surrounding graph (``call_link_label`` ``bands``). The step is seeded
    from ``nscf_overrides`` (the caller's nscf-shaped protocol override
    dict) with the calculation type forced on top, and reads the density
    from ``scf_remote_folder``. The run computes only what the seed states:
    in particular ``nbnd`` must be in it, or pw.x defaults to roughly the
    nelec/2 occupied bands — a caller whose nscf resolves its band count
    outside its overrides (e.g. inside a workchain builder) injects the
    resolved value into the seed. Returns the step's outputs
    (``output_band`` holds the eigenvalues along the path).
    """
    # ``.build()`` executes graph bodies eagerly, where graph inputs arrive as
    # provenance-tagged proxies; the family label ends up bound as an SQL
    # parameter inside ``get_builder_from_protocol``, which needs a plain str.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None

    # Deep-copy the seed: the shared assembly stamps this step's calculation
    # type into the overrides, which must never leak into the caller's nscf
    # override through shared nested dicts.
    bands_overrides = copy.deepcopy(dict(nscf_overrides or {}))
    # A calculation type riding along in the seed is residue, not a conflict.
    bands_overrides.get("pw", {}).get("parameters", {}).get("CONTROL", {}).pop("calculation", None)
    if pseudo_family is not None:
        bands_overrides.setdefault("pseudo_family", pseudo_family)
    return assemble_pw_base_step(
        pw_code,
        structure,
        calculation="bands",
        call_link_label="bands",
        display="Band structure",
        overrides=bands_overrides,
        protocol=protocol,
        electronic_type=electronic_type,
        kpoints=bands_kpoints,
        parent_folder=scf_remote_folder,
        parallelization=parallelization,
    )


@task.graph
def RunPwBands(
    codes: PwBandsCodes,
    structure: orm.StructureData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    parallelization: ParallelizationDict | None = None,
    scf_kpoints: orm.KpointsData | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
) -> ScfBandsOutputs:
    """Run PwBandsWorkChain using the protocol-based builder pattern.

    This task wraps PwBandsWorkChain and uses get_builder_from_protocol to
    construct the inputs from a simplified set of arguments (codes, structure,
    protocol, overrides, parallelization).

    Args:
        codes: Code instances; ``codes["pw"]`` runs both the scf and bands steps.
        structure: The StructureData instance to use.
        pseudo_family: Pseudo family label (e.g. ``"PseudoDojo/0.4/PBE/SR/standard/upf"``).
            If not specified, the protocol default is used.
        protocol: Protocol to use. If not specified, the default will be used.
        overrides: Optional dictionary of inputs to override protocol defaults.
        parallelization: Per-code parallelization mapping (keyed by code name);
            the ``pw`` entry sets the scf/bands pw.x ``metadata.options`` and
            ``-npool``.
        scf_kpoints: Explicit k-points for the SCF step, replacing the
            protocol's ``kpoints_distance``. Leave unset only where no mesh
            is prescribed and the protocol should choose one. The bands step
            is unaffected: it samples the path, not a mesh.
        bands_kpoints: Explicit KpointsData for the bands path. If provided,
            seekpath is bypassed entirely.
        electronic_type: Defaults to ``INSULATOR`` (fixed occupations):
            Koopmans functionals treat insulators exclusively, and kcw.x
            refuses non-fixed occupations outright.
        spin_type: Spin regime for both steps. ``COLLINEAR`` sets
            ``nspin = 2``; ``NON_COLLINEAR`` and ``SPIN_ORBIT`` set
            ``noncolin = .true.``, the latter adding ``lspinorb``. With the
            ``INSULATOR`` default, ``COLLINEAR`` also needs a
            ``tot_magnetization`` in ``overrides``: pw.x rejects fixed
            occupations under LSDA without one.

    Returns:
        Dict with scf_parameters and band_structure outputs.
    """
    validate_parallelization(parallelization)

    # A graph input arrives as a wrapt proxy; coerce to a plain str so the
    # protocol builder's pseudo-family QueryBuilder can bind it.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None
    overrides = overrides or {}

    # Inject pseudo_family into both scf and bands overrides
    inject_pseudo_family(overrides, pseudo_family, ("scf", "bands"))
    merge_parallelization_into_overrides(
        overrides, parallelization, [(("scf", "pw"), "pw"), (("bands", "pw"), "pw")]
    )

    builder = PwBandsWorkChain.get_builder_from_protocol(
        code=codes["pw"],
        structure=structure,
        protocol=protocol,
        overrides=overrides,
        electronic_type=unwrap_enum(electronic_type, ElectronicType),
        spin_type=unwrap_enum(spin_type, SpinType),
    )

    data = get_dict_from_builder(builder)

    # Each step owns its calculation mode: the scf step stays 'scf' and the
    # bands step stays 'bands', raising if a merged override set either
    # otherwise.
    for step, expected, display in (("scf", "scf", "SCF"), ("bands", "bands", "Band structure")):
        pw_inputs = data[step]["pw"]
        pw_inputs["parameters"] = orm.Dict(
            enforce_step_calculation(pw_inputs["parameters"].get_dict(), step, expected)
        )
        # ``PwBandsWorkChain`` sets only ``call_link_label`` on the inputs it
        # exposes, so a label given here reaches the step it names.
        name_step(data[step], display)
        name_step(pw_inputs, display)
        force_pw_verbosity(pw_inputs)

    pin_kpoints(data["scf"], scf_kpoints)

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
    pw_code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    parallelization: ParallelizationDict | None = None,
    scf_kpoints: orm.KpointsData | None = None,
    nscf_kpoints: orm.KpointsData | None = None,
    nbnd: int | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
) -> ScfNscfOutputs:
    """Run SCF + NSCF using two PwBaseWorkChain steps.

    Both steps are seeded from
    ``Wannier90WorkChain.get_scf_nscf_builders_from_protocol``, so the NSCF
    carries the invariants a Wannierization needs — ``nosym`` / ``noinv``
    for the full k-grid, ``diago_full_acc`` for the empty states,
    ``startingpot = 'file'`` off the SCF density, and the k-points listed
    in wannier90's own order — from the same recipe
    ``Wannier90WorkChain`` runs itself. Each step samples the Brillouin
    zone on the mesh it is given, falling back to the protocol's
    ``kpoints_distance`` when none is.

    The pw.x spin regime comes from ``overrides``, not from a ``spin_type``
    argument: callers force it (``nspin``, ``noncolin``, ``lspinorb``)
    on top of the merged parameters, which the recipe's protocol-level
    ``spin_type`` could not express.

    Overrides are split by namespace: ``overrides["scf"]`` applies to the
    SCF step and ``overrides["nscf"]`` applies to the NSCF step.

    Args:
        pw_code: The Code instance configured for the quantumespresso.pw plugin.
        structure: The StructureData instance to use.
        pseudo_family: Pseudo family label (e.g. ``"PseudoDojo/0.4/PBE/SR/standard/upf"``).
            If not specified, the protocol default is used.
        protocol: Protocol to use. If not specified, the default will be used.
        overrides: Optional dictionary with ``"scf"`` and/or ``"nscf"`` keys.
        parallelization: Per-code parallelization mapping (keyed by code name);
            the ``pw`` entry sets the scf/nscf pw.x ``metadata.options`` and
            ``-npool``.
        scf_kpoints: Explicit k-points for the SCF step, replacing the
            protocol's ``kpoints_distance``. Leave unset only where no
            mesh is prescribed and the protocol should choose one.
        nscf_kpoints: Explicit k-points for the NSCF step, replacing the
            protocol's ``kpoints_distance``. A mesh is expanded to the
            explicit list in wannier90's k-point order; an explicit list is
            used as given.
        nbnd: Band count for the NSCF step. Leave unset to keep whatever
            ``overrides["nscf"]`` states, or the protocol default where it
            states nothing.
        electronic_type: Defaults to ``INSULATOR`` (fixed occupations):
            Koopmans functionals treat insulators exclusively, and kcw.x
            refuses non-fixed occupations outright.

    Returns:
        Dict with remote folders and retrieved data from both steps.
    """
    from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

    # A graph input arrives as a wrapt proxy; coerce to a plain str so the
    # protocol builder's pseudo-family QueryBuilder can bind it.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None
    overrides = overrides or {}

    # Inject pseudo_family as a top-level override for both steps
    inject_pseudo_family(overrides, pseudo_family, ("scf", "nscf"))
    merge_parallelization_into_overrides(
        overrides, parallelization, [(("scf", "pw"), "pw"), (("nscf", "pw"), "pw")]
    )

    # Each step owns its ``CONTROL.calculation``, stamped into the overrides
    # the recipe merges on top of the protocol so a conflicting explicit
    # value raises rather than being dropped.
    for namespace, calculation in (("scf", "scf"), ("nscf", "nscf")):
        enforce_step_calculation(
            overrides.setdefault(namespace, {}).setdefault("pw", {}).setdefault("parameters", {}),
            namespace,
            calculation,
        )

    # ``spin_type`` stays at its default: the regimes our callers run
    # (collinear, noncollinear, spin-orbit) are forced through ``overrides``
    # instead, so the recipe never reaches its SOC refusal.
    scf_builder, nscf_builder = Wannier90WorkChain.get_scf_nscf_builders_from_protocol(
        pw_code,
        structure=structure,
        nbnd=int(nbnd) if nbnd is not None else None,
        kpoints=nscf_kpoints,
        protocol=protocol,
        overrides={key: overrides[key] for key in ("scf", "nscf") if key in overrides},
        electronic_type=unwrap_enum(electronic_type, ElectronicType),
    )

    scf_data = _finish_pw_base_step(
        get_dict_from_builder(scf_builder), step="scf", display="SCF", kpoints=scf_kpoints
    )
    scf_outputs = PwBaseStep(**scf_data)

    # The recipe pins the nscf k-points itself, expanding a mesh into
    # wannier90's order, so re-pin what it settled on rather than the
    # caller's node; that also drops the ``kpoints_force_parity`` which only
    # qualifies a distance.
    nscf_data = get_dict_from_builder(nscf_builder)
    nscf_data = _finish_pw_base_step(
        nscf_data,
        step="nscf",
        display="NSCF",
        kpoints=nscf_data.get("kpoints", nscf_kpoints),
    )
    nscf_data["pw"]["parent_folder"] = scf_outputs["remote_folder"]
    nscf_outputs = PwBaseStep(**nscf_data)

    return ScfNscfOutputs(
        scf_remote_folder=scf_outputs["remote_folder"],
        nscf_remote_folder=nscf_outputs["remote_folder"],
        nscf_retrieved=nscf_outputs["retrieved"],
        nscf_output_parameters=nscf_outputs["output_parameters"],
        nscf_output_band=nscf_outputs["output_band"],
        nscf_output_kpoints=nscf_outputs["output_kpoints"],
    )
