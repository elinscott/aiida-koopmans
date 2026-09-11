"""Yambo BSE with Koopmans eigenvalues: the quasiparticle-database step.

Yambo's BSE solver reads its quasiparticle corrections from an ``ndb.QP``
database in the SAVE directory, rather than computing a GW correction
itself. Feeding it Koopmans (KI or pKI) eigenvalues in place of a GW run
lets a BSE spectrum be built directly from a kcw.x ``ham`` calculation.

:func:`generate_qp_database` builds that ``ndb.QP`` by driving k2y's
``KcwQpDatabaseGenerator`` directly on parsed AiiDA data: the kcw.x
``ham`` output's chosen Koopmans/KS eigenvalue grids and the k-points of
the nscf run that seeded it, mapped onto the yambo p2y run's own k-point grid
(``ns.db1``). This is deliberately not k2y's ``from_aiida`` classmethod
(walks PKs, bypassing AiiDA provenance) nor its ``set_koopmans_eval``
file route (re-parses a kcw.x stdout file and needs the optional
``ase-koopmans`` dependency this package does not carry) nor
``generate_kcw_qp_database``/``generate_QP_db_SinglefileData`` (the
former only wraps ``from_aiida`` for provenance; the latter writes its
output to the current working directory).

:func:`RunBse` wires pw.x -> p2y -> yambo init -> :func:`generate_qp_database`
-> yambo BSE into one ``@task.graph``; :func:`SinglepointBSEWorkflow` composes
a :func:`~aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow` in front of
it, reading the ham step's eigenvalues and the shared ground state's nscf
output straight off the DFPT chain's own outputs.
"""

# No ``from __future__ import annotations`` in this module: stringified
# annotations hide ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the dispatcher reads off the Codes
# TypedDicts.

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, TypedDict, cast

import numpy as np
from aiida import orm
from aiida.common.folders import SandboxFolder
from aiida_quantumespresso.common.types import ElectronicType
from aiida_workgraph import dynamic, task
from aiida_workgraph.socket_spec import SocketMeta
from aiida_workgraph.utils import get_dict_from_builder

# k2y.k2y imports only netCDF4/ase/xarray/numpy/yambopy -- no AiiDA import,
# so unlike aiida_yambo.workflows it never touches AiiDA's profile config at
# import time and is safe to import at module scope.
from k2y.k2y import KcwQpDatabaseGenerator
from node_graph import reference
from qe_tools import CONSTANTS

from aiida_koopmans.parallelization import (
    ParallelizationDict,
    merge_parallelization_into_inputs,
    resolve_parallelization,
    validate_parallelization,
)
from aiida_koopmans.workgraphs.block_wannierize import WannierizeOverrides
from aiida_koopmans.workgraphs.dfpt import (
    ChannelResults,
    DfptCodes,
    KoopmansDFPTOutputs,
    ManifoldBlocks,
    SinglepointDFPTWorkflow,
    emit_namespace_dict_field,
)
from aiida_koopmans.workgraphs.pw import PwCode

P2yCode = Annotated[
    orm.AbstractCode,
    SocketMeta(help="Needed to convert the pw.x ground state into a yambo SAVE directory."),
]
YamboCode = Annotated[
    orm.AbstractCode,
    SocketMeta(help="Needed to run the BSE calculation."),
]


class BseCodes(TypedDict):
    """Codes the Koopmans-eigenvalues BSE route will need."""

    pw: PwCode
    p2y: P2yCode
    yambo: YamboCode


#: ``eigenvalues`` values accepted by :func:`generate_qp_database`, and the
#: ``output_parameters`` key each reads (see
#: ``aiida_koopmans.parsers.kcw.KcwHamParameters``): one row per grid
#: k-point, one column per Wannier-Hamiltonian band, in eV.
_EIGENVALUE_KEYS = {"ki": "ki_eigenvalues_on_grid", "pki": "pki_eigenvalues_on_grid"}
_KS_KEY = "ks_eigenvalues_on_grid"


@task.calcfunction
def generate_qp_database(
    yambo_save: orm.FolderData,
    ham_output_parameters: orm.Dict,
    nscf_output_band: orm.BandsData,
    eigenvalues: orm.Str,
) -> orm.SinglefileData:
    """Build a yambo ``ndb.QP`` from a kcw.x ``ham`` run's Koopmans eigenvalues.

    ``yambo_save`` is a yambo p2y run's ``retrieved`` folder: every file
    it holds is staged into one directory, so ``ns.db1`` sits alongside
    ``ndb.gops`` / ``ndb.kindx`` when those are present (needed for the
    SERIAL_NUMBER that lets yambo match the database to its own SAVE).

    ``eigenvalues`` selects which Koopmans eigenvalue flavor seeds the QP
    corrections: ``'ki'`` reads the ham run's ``ki_eigenvalues_on_grid``,
    ``'pki'`` its ``pki_eigenvalues_on_grid``. k2y's own file-based route
    (``KcwQpDatabaseGenerator.set_koopmans_eval``) defaults to
    ``pki_eigenvalues_on_grid``; this input makes that choice explicit
    instead of hardcoding either flavor.

    ``ham_output_parameters`` is the kcw.x ``ham`` run's parsed
    ``output_parameters`` and must carry the requested eigenvalue grid
    alongside ``ks_eigenvalues_on_grid``, each shaped
    ``(n_grid_kpoints, n_bands)``.

    ``nscf_output_band`` is the ``output_band`` of the nscf run that
    produced the kcw.x k-point grid, and must carry the same number of
    k-points as the requested eigenvalue grid and
    ``ks_eigenvalues_on_grid`` -- a mismatch means the two inputs come
    from different runs, and k2y's own k-point matching would otherwise
    mis-associate the surplus rows against a stale index rather than fail
    loudly. ``BandsData.get_array('kpoints')`` returns crystal
    (reciprocal-lattice-fraction) coordinates by default
    (``KpointsData.get_kpoints(cartesian=False)``), which is the
    coordinate system k2y's k-point matching expects.
    """
    flavor = eigenvalues.value
    if flavor not in _EIGENVALUE_KEYS:
        raise ValueError(
            f"`eigenvalues` is {flavor!r} -- must be one of {tuple(_EIGENVALUE_KEYS)}."
        )
    eigenvalue_key = _EIGENVALUE_KEYS[flavor]
    required_keys = (eigenvalue_key, _KS_KEY)

    params = ham_output_parameters.get_dict()
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise ValueError(
            f"`ham_output_parameters` is missing {missing} -- pass the "
            f"`output_parameters` of a `ham`-mode kcw.x run that computed "
            f"`eigenvalues={flavor!r}` eigenvalues. Keys present: {sorted(params)}."
        )

    n_nscf_kpoints = nscf_output_band.get_array("kpoints").shape[0]
    for key in required_keys:
        n_grid_kpoints = np.asarray(params[key]).shape[0]
        if n_grid_kpoints != n_nscf_kpoints:
            raise ValueError(
                f"`{key}` has {n_grid_kpoints} k-points but `nscf_output_band` has "
                f"{n_nscf_kpoints} -- pass the `output_band` of the same kcw.x `ham` "
                "run's own seeding nscf, not some other run's grid."
            )

    # SandboxFolder, not ``tempfile`` directly: AiiDA's own local-scratch
    # abstraction for this get-then-consume shape (see ``convert_spin``'s
    # get/put staging), auto-erased on exit like ``TemporaryDirectory``.
    with SandboxFolder() as sandbox:
        for filename in yambo_save.base.repository.list_object_names():
            with yambo_save.base.repository.open(filename, "rb") as handle:
                Path(sandbox.get_abs_path(filename)).write_bytes(handle.read())
        if "ns.db1" not in yambo_save.base.repository.list_object_names():
            raise ValueError(
                "`yambo_save` has no `ns.db1` -- pass the p2y (yambo init) run's "
                "retrieved folder, not some other calculation's."
            )
        ns_db1 = sandbox.get_abs_path("ns.db1")

        generator = KcwQpDatabaseGenerator(ns_db1=str(ns_db1))
        generator.eigenvalues_KI = np.array(params[eigenvalue_key])
        generator.eigenvalues_KS = np.array(params[_KS_KEY])
        generator.kpoints_grid_kcw = nscf_output_band.get_array("kpoints")
        generator.kpoints_type = "crystal"
        generator.generate_mappings()

        qp_path = sandbox.get_abs_path("ndb.QP")
        generator.generate_QP_db(str(qp_path))
        return orm.SinglefileData(file=str(qp_path), filename="ndb.QP")


@lru_cache(maxsize=1)
def _load_yambo_workflow() -> Any:
    """Import and return the ``YamboWorkflow`` process class, lazily.

    ``aiida_yambo.workflows.utils.helpers_yambowf`` evaluates
    ``Bool(False)`` as a default argument of its own
    ``QP_bands_interface`` at import time; constructing that ``orm.Bool``
    node needs an already-loaded AiiDA profile, so importing
    ``aiida_yambo`` before one exists raises ``ConfigurationError``.
    Deferring the import to this function's first call from inside a
    graph body -- never at ``bse.py`` import time -- keeps every route
    that never touches BSE working with no profile loaded yet.
    """
    from aiida.plugins import WorkflowFactory

    return WorkflowFactory("yambo.yambo.yambowf")


def _cutoffs_in_ry(nscf_output_parameters: dict) -> tuple[float, float]:
    """Return ``(ecutwfc, ecutrho)`` in Ry from a pw.x run's parsed output.

    ``output_parameters`` carries ``wfc_cutoff``/``rho_cutoff`` in eV (the
    aiida-quantumespresso XML parser's own units); ``PwBaseWorkChain.
    get_builder_from_protocol`` wants the Ry values pw.x's ``SYSTEM``
    namelist takes.
    """
    params = dict((nscf_output_parameters or {}).items())
    ry_to_ev = CONSTANTS.hartree_to_ev / 2
    return params["wfc_cutoff"] / ry_to_ev, params["rho_cutoff"] / ry_to_ev


def _bse_mpi_roles(parallelization: ParallelizationDict | None) -> dict[str, str]:
    """Return yambo's ``BS_CPU``/``BS_ROLEs`` for the ``yambo`` entry's rank count.

    Puts every rank on the ``k`` (k-point) role, leaving ``eh``/``t``
    unsplit -- a safe default, not a tuned one; no BSE run has yet
    benchmarked a split for this route. A missing or single-rank
    ``yambo`` entry omits both keys, so yambo runs its own single-rank
    default.
    """
    options, _ = resolve_parallelization(parallelization, "yambo")
    ntasks = options.get("resources", {}).get("num_mpiprocs_per_machine")
    if not ntasks or int(ntasks) <= 1:
        return {}
    return {"BS_CPU": f"{int(ntasks)} 1 1", "BS_ROLEs": "k eh t"}


class BseOutputs(TypedDict, total=False):
    """Outputs of :func:`RunBse`: the BSE ``YamboWorkflow``'s own outputs, plus its QP database.

    ``output_ywfl_parameters`` carries the ``additional_parsing``
    quantities (``lowest_exciton`` / ``brightest_exciton``); ``array_eps``
    the BSE absorption spectrum; ``array_excitonic_states`` the exciton
    eigenvectors. Each is present exactly when the BSE ``YamboWorkflow``
    itself produced it, same as any other CalcJob-namespace output.
    """

    remote_folder: orm.RemoteData
    retrieved: orm.FolderData
    output_parameters: dict
    output_ywfl_parameters: dict
    array_eps: orm.ArrayData
    array_excitonic_states: orm.ArrayData
    qp_database: orm.SinglefileData


@task.graph
def RunBse(
    codes: BseCodes,
    structure: orm.StructureData,
    kpoints: orm.KpointsData,
    nscf_output_band: orm.BandsData,
    nscf_output_parameters: dict,
    ham_output_parameters: dict,
    bse_parameters: dict,
    eigenvalues: str = "ki",
    pseudo_family: str | None = None,
    protocol: str | None = None,
    protocol_qe: str | None = None,
    parallelization: ParallelizationDict | None = None,
) -> BseOutputs:
    """Run a yambo BSE spectrum seeded by Koopmans (KI/pKI) quasiparticle corrections.

    A fresh scf -> nscf -> p2y ("yambo init") builds the yambo SAVE
    directory: the koopmans nscf cannot be reused, since kcw.x needs an
    nspin=2 scratch and yambo's own p2y convention differs. ``kpoints`` (a
    Monkhorst-Pack mesh) must be the mesh the koopmans nscf ran on -- it
    seeds the yambo init's own nscf k-mesh, so the resulting SAVE
    directory's k-grid lines up with ``nscf_output_band`` /
    ``ham_output_parameters``, the grid :func:`generate_qp_database` maps
    the Koopmans eigenvalues onto.

    ``nscf_output_parameters`` is the koopmans nscf's own parsed
    ``output_parameters`` (``wfc_cutoff`` / ``rho_cutoff``, in eV): both
    reach the init and BSE steps' scf/nscf ``SYSTEM`` overrides in Ry,
    since ``PwBaseWorkChain.get_builder_from_protocol`` silently replaces
    an ``ecutwfc``-only override with the pseudo family's own recommended
    cutoffs unless ``ecutrho`` rides along with it.

    ``ham_output_parameters`` / ``nscf_output_band`` are a kcw.x ``ham``
    run's parsed ``output_parameters`` and its own seeding nscf's
    ``output_band`` (see :func:`generate_qp_database`, which this graph
    calls); ``eigenvalues`` selects which Koopmans flavor ('ki'/'pki')
    seeds the QP database.

    ``bse_parameters`` is the yambo BSE runcard's own ``arguments`` /
    ``variables`` (``BndsRnXs``, ``NGsBlkXs``, ``BSENGBlk``, ``BSEBands``,
    ``BEnRange``, ``BEnSteps``, ``BDmRange``, each ``[value, unit]`` per
    yambopy's convention) and ``metadata`` (the BSE step's own scheduler
    options). ``variables['KfnQPdb']`` is this graph's own -- pointing at
    the QP database it builds -- and refused if the caller states it.
    ``parallelization``'s ``yambo`` entry sets the BSE step's rank count
    and, past one rank, a k-point-only ``BS_CPU``/``BS_ROLEs`` MPI split
    (see :func:`_bse_mpi_roles`).

    Raises:
        ValueError: If ``bse_parameters['variables']`` states ``KfnQPdb``.
    """
    validate_parallelization(parallelization)

    # ``.build()`` executes graph bodies eagerly, where graph inputs arrive as
    # provenance-tagged proxies; the family label ends up bound as an SQL
    # parameter inside ``get_builder_from_protocol``, which needs a plain str
    # (same fix as ``run_bands_step`` in ``workgraphs/pw.py``).
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None

    variables = dict((bse_parameters or {}).get("variables", {}).items())
    if "KfnQPdb" in variables:
        raise ValueError(
            "bse_parameters['variables'] states 'KfnQPdb' -- RunBse sets it itself, "
            "pointing at the Koopmans QP database it builds. Remove it from bse_parameters."
        )
    arguments = list((bse_parameters or {}).get("arguments", []))
    bse_metadata = dict((bse_parameters or {}).get("metadata", {}).items())

    ecutwfc, ecutrho = _cutoffs_in_ry(nscf_output_parameters)
    cutoff_system = {"SYSTEM": {"ecutwfc": ecutwfc, "ecutrho": ecutrho}}

    try:
        mesh, _offset = kpoints.get_kpoints_mesh()
    except AttributeError:
        raise ValueError(
            "`kpoints` must be a Monkhorst-Pack mesh (`set_kpoints_mesh`), matching the "
            "koopmans nscf mesh yambo's p2y step must reproduce."
        ) from None
    yambo_kpoints = orm.KpointsData()
    yambo_kpoints.set_kpoints_mesh([int(size) for size in mesh])

    yambo_workflow = _load_yambo_workflow()
    yambo_step = task(yambo_workflow)

    init_overrides = {
        "scf": {"pw": {"parameters": deepcopy(cutoff_system)}},
        "nscf": {
            "pw": {
                "parameters": {
                    **deepcopy(cutoff_system),
                    "ELECTRONS": {"diagonalization": "cg"},
                },
            },
        },
    }
    init_builder = yambo_workflow.get_builder_from_protocol(
        pw_code=codes["pw"],
        preprocessing_code=codes["p2y"],
        code=codes["yambo"],
        protocol_qe=protocol_qe or "moderate",
        protocol=protocol or "moderate",
        structure=structure,
        pseudo_family=pseudo_family,
        overrides=init_overrides,
        electronic_type=ElectronicType.INSULATOR,
        calc_type="bse",
    )
    init_data = get_dict_from_builder(init_builder)
    init_data.pop("clean_workdir", None)
    init_data["nscf"]["kpoints"] = yambo_kpoints
    init_data["yres"]["yambo"]["settings"] = orm.Dict({"INITIALISE": True})
    merge_parallelization_into_inputs(init_data["yres"]["yambo"], parallelization, "yambo")
    init_data.setdefault("metadata", {})["call_link_label"] = "yambo_init"
    init = yambo_step(**init_data)

    ham_params = dict((ham_output_parameters or {}).items())
    qp_db = generate_qp_database(
        yambo_save=init["retrieved"],
        ham_output_parameters=ham_params,
        nscf_output_band=nscf_output_band,
        eigenvalues=eigenvalues,
        metadata={"call_link_label": "generate_qp_database"},
    ).result

    bse_variables = {**variables, "KfnQPdb": "E < ./ndb.QP"}
    bse_overrides = {
        "scf": {"pw": {"parameters": deepcopy(cutoff_system)}},
        "nscf": {
            "pw": {
                "parameters": {
                    **deepcopy(cutoff_system),
                    "ELECTRONS": {"diagonalization": "cg"},
                },
            },
        },
        "yres": {
            "yambo": {
                "parameters": {"arguments": arguments, "variables": bse_variables},
                "metadata": bse_metadata,
            },
        },
    }
    bse_builder = yambo_workflow.get_builder_from_protocol(
        pw_code=codes["pw"],
        preprocessing_code=codes["p2y"],
        code=codes["yambo"],
        protocol_qe=protocol_qe or "moderate",
        protocol=protocol or "moderate",
        structure=structure,
        pseudo_family=pseudo_family,
        overrides=bse_overrides,
        electronic_type=ElectronicType.INSULATOR,
        calc_type="bse",
    )
    bse_data = get_dict_from_builder(bse_builder)
    bse_data.pop("clean_workdir", None)
    bse_data["nscf"]["kpoints"] = yambo_kpoints
    bse_data["parent_folder"] = init["remote_folder"]
    bse_data["additional_parsing"] = ["lowest_exciton", "brightest_exciton"]

    # ``get_builder_from_protocol`` already returns an ``orm.Dict`` for
    # ``parameters``: rebuild it to add the MPI-role split, the same
    # get-then-rebuild idiom the k2y BSE example script uses to strip
    # GW-only keys from the same builder output.
    bse_params_dict = bse_data["yres"]["yambo"]["parameters"].get_dict()
    bse_params_dict["variables"].update(_bse_mpi_roles(parallelization))
    bse_data["yres"]["yambo"]["parameters"] = orm.Dict(bse_params_dict)
    bse_data["yres"]["yambo"]["QP_corrections"] = qp_db
    merge_parallelization_into_inputs(bse_data["yres"]["yambo"], parallelization, "yambo")
    bse_data.setdefault("metadata", {})["call_link_label"] = "bse"
    bse = yambo_step(**bse_data)

    return BseOutputs(
        remote_folder=bse["remote_folder"],
        retrieved=bse["retrieved"],
        output_parameters=bse["output_parameters"],
        output_ywfl_parameters=bse["output_ywfl_parameters"],
        array_eps=bse["array_eps"],
        array_excitonic_states=bse["array_excitonic_states"],
        qp_database=qp_db,
    )


class BseSinglepointCodes(DfptCodes, BseCodes):  # type: ignore[misc]
    """Codes for :func:`SinglepointBSEWorkflow`: the DFPT chain's codes plus BSE's own.

    Both parents declare ``pw`` as the same ``PwCode`` -- mypy flags any
    TypedDict multiple-inheritance merge that redeclares a field, even with
    an identical type, hence the ignore.
    """


class SinglepointBseOutputs(TypedDict):
    """Outputs of :func:`SinglepointBSEWorkflow`: the DFPT chain's outputs, plus the BSE run."""

    dfpt: KoopmansDFPTOutputs
    bse: BseOutputs


class SelectedChannelOutputs(TypedDict):
    """Outputs of :func:`SelectDfptChannel`."""

    ham_parameters: dict


@task.graph
def SelectDfptChannel(
    channels: Annotated[dict, dynamic(ChannelResults)],
    channel_key: str,
) -> SelectedChannelOutputs:
    """Pick one channel's ``ham_parameters`` out of a DFPT chain's dynamic namespace.

    Mirrors :func:`~aiida_koopmans.workgraphs.dfpt.RunDFPT`'s own handling
    of a nested sub-graph's dynamic namespace output (its
    ``block_wannier`` argument): the namespace has no per-key sockets
    until the graph that produced it has actually run its own deferred
    body, so a caller must hand the whole namespace to a nested graph
    rather than subscript it directly. This graph exists solely to be
    that nested graph for :class:`~aiida_koopmans.workgraphs.dfpt.KoopmansDFPTOutputs`'
    ``channels``.
    """
    channel = channels[str(channel_key)]
    return SelectedChannelOutputs(
        ham_parameters=emit_namespace_dict_field(
            value=channel["ham_parameters"],
            metadata={"call_link_label": "emit_ham_parameters"},
        ).result
    )


def _dfpt_codes_from(codes: BseSinglepointCodes) -> DfptCodes:
    """Return :func:`~aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow`'s codes namespace.

    Wires every code :class:`DfptCodes` requires, read off its own
    ``__required_keys__`` rather than hard-coded, so the two TypedDicts
    can never drift apart silently. Its ``NotRequired`` members (``ph``,
    ``projwfc``) ride along unconditionally too: whether they are used is
    ``SinglepointDFPTWorkflow``'s own entry decision (mirrors
    :func:`~aiida_koopmans.workgraphs.dfpt._wannierize_codes_for_channel`).
    """
    dfpt_codes: dict[str, Any] = {
        name: reference(codes, name) for name in DfptCodes.__required_keys__
    }
    dfpt_codes["ph"] = reference(codes, "ph")
    dfpt_codes["projwfc"] = reference(codes, "projwfc")
    return cast("DfptCodes", dfpt_codes)


def _bse_codes_from(codes: BseSinglepointCodes) -> BseCodes:
    """Return :func:`RunBse`'s codes namespace out of the composed BSE codes."""
    bse_codes: dict[str, Any] = {
        name: reference(codes, name) for name in BseCodes.__required_keys__
    }
    return cast("BseCodes", bse_codes)


@task.graph
def SinglepointBSEWorkflow(
    codes: BseSinglepointCodes,
    structure: orm.StructureData,
    manifolds: dict[str, ManifoldBlocks],
    kpoints: orm.KpointsData,
    bse_parameters: dict,
    scf_kpoints: orm.KpointsData | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    protocol_qe: str | None = None,
    overrides: WannierizeOverrides | None = None,
    eigenvalues: str = "ki",
    parallelization: ParallelizationDict | None = None,
) -> SinglepointBseOutputs:
    """Run a Koopmans DFPT singlepoint, then a BSE spectrum seeded by its eigenvalues.

    Composes :func:`~aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow`
    with :func:`RunBse`: the DFPT chain's shared ground state (its
    ``nscf_output_band`` / ``nscf_output_parameters``) and the ``none``
    channel's ``ham_parameters`` feed the BSE step directly, so a caller
    states the DFPT/wannierization inputs once.

    Phase-1 scope, refused explicitly:

    * ``spin`` is always ``NONE`` -- ``manifolds`` must carry exactly one
      ``"none"`` key. A collinear or spinor DFPT chain has no single
      channel for the BSE step to read; run
      ``SinglepointDFPTWorkflow`` and :func:`RunBse` separately for those.
    * ``structure`` must be periodic in all three directions -- yambo's
      p2y step needs a periodic ground state; molecular BSE is
      unimplemented.
    * The DFPT dispatcher's own correction-scope gap (``PKIPZ``/``PZ``/
      ``ALL`` raise ``NotImplementedError`` upstream) is this route's only
      guard against a wrongly-corrected ``ham`` run reaching
      :func:`generate_qp_database`: that calcfunction checks only that the
      requested ``eigenvalues`` key is *present* in ``ham_output_parameters``,
      not that the values under it were actually produced by that
      correction.

    ``protocol`` reaches both chains: ``SinglepointDFPTWorkflow``'s own QE
    protocol and :func:`RunBse`'s yambo protocol. ``protocol_qe`` sets only
    the BSE route's own fresh scf/nscf/p2y -- pass it when that QE step
    should run at a different precision than ``protocol``.
    ``bse_parameters`` / ``eigenvalues`` / the BSE half of
    ``parallelization`` pass straight to :func:`RunBse`; every other
    argument passes straight to ``SinglepointDFPTWorkflow``.
    """
    if set(manifolds) != {"none"}:
        raise NotImplementedError(
            "SinglepointBSEWorkflow only supports spin='none' (manifolds keyed by a "
            f"single 'none' entry), got manifold keys {sorted(manifolds)}. Run "
            "SinglepointDFPTWorkflow and RunBse separately for a collinear or spinor chain."
        )
    if not all(structure.pbc):
        raise NotImplementedError(
            "SinglepointBSEWorkflow supports periodic structures only: yambo's p2y step "
            "needs a periodic ground state. Molecular BSE is unimplemented."
        )

    dfpt = SinglepointDFPTWorkflow(
        codes=_dfpt_codes_from(codes),
        structure=structure,
        manifolds=manifolds,
        kpoints=kpoints,
        scf_kpoints=scf_kpoints,
        pseudo_family=pseudo_family,
        protocol=protocol,
        overrides=overrides,
        parallelization=parallelization,
        metadata={"call_link_label": "dfpt", "label": "Koopmans DFPT"},
    )
    channel = SelectDfptChannel(
        channels=dfpt["channels"],
        channel_key="none",
        metadata={"call_link_label": "select_channel"},
    )
    bse = RunBse(
        codes=_bse_codes_from(codes),
        structure=structure,
        kpoints=kpoints,
        nscf_output_band=dfpt["ground_state"]["nscf_output_band"],
        nscf_output_parameters=dfpt["ground_state"]["nscf_output_parameters"],
        ham_output_parameters=channel["ham_parameters"],
        bse_parameters=bse_parameters,
        eigenvalues=eigenvalues,
        pseudo_family=pseudo_family,
        protocol=protocol,
        protocol_qe=protocol_qe,
        parallelization=parallelization,
        metadata={"call_link_label": "bse", "label": "BSE spectrum"},
    )
    return SinglepointBseOutputs(dfpt=dfpt, bse=bse)
