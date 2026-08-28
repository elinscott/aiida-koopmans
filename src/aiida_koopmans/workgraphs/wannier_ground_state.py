"""The Wannier ground state: the scf + nscf every Wannierization starts from.

Built from aw90's recipe. Its two callers:
:func:`~aiida_koopmans.workgraphs.block_wannierize.WannierizeBlocks` and
:func:`~aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow`.
"""

from typing import Any, NotRequired, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType
from aiida_workgraph import task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.owned_keywords import owned
from aiida_koopmans.parallelization import ParallelizationDict, merge_parallelization_into_overrides
from aiida_koopmans.workgraphs import enforce_step_calculation, inject_pseudo_family, unwrap_enum
from aiida_koopmans.workgraphs.pw import PwBaseStep, _finish_pw_base_step


class ScfNscfOutputs(TypedDict):
    """Outputs of a chained SCF + NSCF PwBaseWorkChain run."""

    scf_remote_folder: orm.RemoteData
    nscf_remote_folder: orm.RemoteData
    nscf_retrieved: orm.FolderData
    nscf_output_parameters: dict
    nscf_output_band: orm.BandsData
    nscf_output_kpoints: NotRequired[orm.KpointsData]


@task.graph
def RunWannierGroundState(
    pw_code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    parallelization: ParallelizationDict | None = None,
    scf_kpoints: orm.KpointsData | None = None,
    nscf_kpoints: orm.KpointsData | None = None,
    scf_remote_folder: orm.RemoteData | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
) -> ScfNscfOutputs:
    """Run the Wannier ground state: chained PwBaseWorkChain scf and nscf steps.

    Every Wannierization starts from this pair. Its only two callers,
    :func:`~aiida_koopmans.workgraphs.block_wannierize.WannierizeBlocks`
    and :func:`~aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow`,
    both Wannierize; a route that does not (the molecular DSCF KS-init,
    which runs kcp.x's own ground state, or plain DFT bands/eps) does not
    call it.

    Both steps are seeded from
    ``Wannier90WorkChain.get_scf_nscf_builders_from_protocol``, so the NSCF
    carries the invariants a Wannierization needs — ``diago_full_acc`` for
    the empty states, ``startingpot = 'file'`` off the SCF density, and the
    k-points listed in wannier90's own order — from the same recipe
    ``Wannier90WorkChain`` runs itself. The NSCF also always runs on the
    full, symmetry-unreduced k-list wannier90 orders: ``nosym`` / ``noinv``
    are forced on top of any override, since a caller cannot switch off the
    grid wannier90 needs. Each step samples the Brillouin zone on the mesh
    it is given, falling back to the protocol's ``kpoints_distance`` when
    none is.

    The pw.x spin regime comes from ``overrides``, not from a ``spin_type``
    argument: callers force ``nspin`` / ``noncolin`` / ``lspinorb`` on top
    of the merged parameters.

    Overrides are split by namespace: ``overrides["scf"]`` applies to the
    SCF step and ``overrides["nscf"]`` applies to the NSCF step.

    With ``scf_remote_folder`` given, no SCF runs: only the recipe's NSCF
    builder is used, reading that density as its ``pw.parent_folder``. The
    NSCF is the same one the internal pair runs, invariants and all, so a
    caller re-Wannierizing on a denser mesh gets the recipe's NSCF rather
    than a second hand-built one.

    Args:
        pw_code: The Code instance configured for the quantumespresso.pw plugin.
        structure: The StructureData instance to use.
        pseudo_family: Pseudo family label (e.g. ``"PseudoDojo/0.4/PBE/SR/standard/upf"``).
            If not specified, the protocol default is used.
        protocol: One of ``moderate`` (the default), ``precise`` or
            ``fast``; anything else raises. The recipe resolves the name
            against ``Wannier90WorkChain``'s protocols, not pw.x's.
        overrides: Optional dictionary with ``"scf"`` and/or ``"nscf"`` keys.
        parallelization: Per-code parallelization mapping (keyed by code name);
            the ``pw`` entry sets the scf/nscf pw.x ``metadata.options`` and
            ``-npool``.
        scf_kpoints: Explicit k-points for the SCF step, replacing the
            protocol's ``kpoints_distance``. Leave unset only where no
            mesh is prescribed and the protocol should choose one. Rejected
            together with ``scf_remote_folder``, which runs no SCF.
        nscf_kpoints: Explicit k-points for the NSCF step, replacing the
            protocol's ``kpoints_distance``. A mesh is expanded to the
            explicit list in wannier90's k-point order; an explicit list is
            used as given.
        scf_remote_folder: A converged SCF scratch on ``structure``. Given,
            the SCF step is skipped and the NSCF restarts from this density;
            ``scf_remote_folder`` comes back out unchanged, so a consumer
            reads one output whichever mode ran.
        electronic_type: Defaults to ``INSULATOR`` (fixed occupations):
            Koopmans functionals treat insulators exclusively, and kcw.x
            refuses non-fixed occupations outright.

    Returns:
        Dict with remote folders and retrieved data from both steps.

    Raises:
        ValueError: If ``scf_kpoints`` is given alongside
            ``scf_remote_folder``, where no SCF runs to sample it.
    """
    from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

    # A graph input arrives as a wrapt proxy; coerce to a plain str so the
    # protocol builder's pseudo-family QueryBuilder can bind it.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None
    overrides = overrides or {}

    if scf_remote_folder is not None and scf_kpoints is not None:
        raise ValueError(
            "scf_kpoints was given together with scf_remote_folder; no scf "
            "runs here, so the mesh would be silently ignored. Set it on "
            "whoever ran the scf."
        )

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

    # wannier90 reads the eigenstates on the full, symmetry-unreduced grid
    # it orders itself; force that on top of the merged overrides so a
    # caller's own nosym/noinv (or the protocol's default) cannot switch it
    # off.
    nscf_system = overrides["nscf"]["pw"]["parameters"].setdefault("SYSTEM", {})
    nscf_system.update(owned("pw.SYSTEM", {"nosym": True, "noinv": True}))

    # ``spin_type`` stays at its default: the regimes our callers run
    # (collinear, noncollinear, spin-orbit) are forced through ``overrides``
    # instead, so the recipe never reaches its SOC refusal.
    scf_builder, nscf_builder = Wannier90WorkChain.get_scf_nscf_builders_from_protocol(
        pw_code,
        structure=structure,
        kpoints=nscf_kpoints,
        protocol=protocol,
        overrides={key: overrides[key] for key in ("scf", "nscf") if key in overrides},
        electronic_type=unwrap_enum(electronic_type, ElectronicType),
    )

    # An external density skips the scf step; the recipe still builds both
    # builders (they are independent), and the scf half is discarded.
    if scf_remote_folder is None:
        scf_data = _finish_pw_base_step(
            get_dict_from_builder(scf_builder), step="scf", display="SCF", kpoints=scf_kpoints
        )
        scf_remote_folder = PwBaseStep(**scf_data)["remote_folder"]

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
    nscf_data["pw"]["parent_folder"] = scf_remote_folder
    nscf_outputs = PwBaseStep(**nscf_data)

    return ScfNscfOutputs(
        scf_remote_folder=scf_remote_folder,
        nscf_remote_folder=nscf_outputs["remote_folder"],
        nscf_retrieved=nscf_outputs["retrieved"],
        nscf_output_parameters=nscf_outputs["output_parameters"],
        nscf_output_band=nscf_outputs["output_band"],
        nscf_output_kpoints=nscf_outputs["output_kpoints"],
    )
