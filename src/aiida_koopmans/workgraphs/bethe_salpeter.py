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

:func:`RunBetheSalpeter` wires pw.x -> p2y -> yambo init -> :func:`generate_qp_database`
-> yambo BSE into one ``@task.graph``; :func:`SinglepointBetheSalpeterWorkflow` composes
a :func:`~aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow` in front of
it, reading the ham step's eigenvalues and the shared ground state's nscf
output straight off the DFPT chain's own outputs.
"""

# No ``from __future__ import annotations`` in this module: stringified
# annotations hide ``NotRequired`` from ``TypedDict.__required_keys__``
# (python/cpython#97727), which the dispatcher reads off the Codes
# TypedDicts.

from collections.abc import Mapping
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

from aiida_koopmans.parallelization import (
    ParallelizationDict,
    merge_parallelization_into_inputs,
    merge_parallelization_into_overrides,
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


class BetheSalpeterCodes(TypedDict):
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

#: Runcard keys ``YamboRestart.get_builder_from_protocol`` computes for a GW
#: run (``GbndRnge``/``FFTGvecs``/``GTermKind``) and leaves behind even under
#: a ``'bse_*'`` protocol: ``GbndRnge`` is assigned from the pre-rename
#: ``BndsRnXp`` before the BSE branch pops that key, and ``GTermKind`` /
#: ``FFTGvecs`` come from the protocol's own ``default_inputs``, applied to
#: every protocol regardless of calc type. None of the three mean anything to
#: a BSE run; the k2y BSE example script strips them from the same builder
#: output before submission.
_GW_ONLY_RUNCARD_KEYS = ("GbndRnge", "FFTGvecs", "GTermKind")

#: Yambo BSE runcard variables :func:`RunBetheSalpeter` determines for itself:
#: ``KfnQPdb`` points at the QP database it builds (see
#: :func:`generate_qp_database`); ``BS_CPU``/``BS_ROLEs`` are the BSE step's
#: own MPI role split, sized off the ``yambo`` parallelization entry's rank
#: count (see :func:`_bse_mpi_roles`). Stating one in
#: ``bse_parameters['variables']`` is refused.
#:
#: Mirrors :mod:`aiida_koopmans.owned_keywords`'s ``OWNED``/``owned``/
#: ``reject_owned`` pattern (same roster shape, same refusal message) rather
#: than joining that module's own ``OWNED``: that dict backs koopmans'
#: generated input-file schema (``koopmans.input_file._codegen.generate``'s
#: ``covered`` check requires every ``OWNED`` block to have a generated
#: model), and koopmans has no yambo input-file block yet -- BSE is not wired
#: into its dispatcher. Move this roster into ``OWNED`` once it does.
_YAMBO_OWNED: frozenset[str] = frozenset({"KfnQPdb", "BS_CPU", "BS_ROLEs"})


def _reject_owned_yambo(keywords: Mapping[str, Any]) -> None:
    """Raise if the caller states a yambo runcard variable this route owns.

    Raises:
        ValueError: If ``keywords`` states a keyword in :data:`_YAMBO_OWNED`.
    """
    stated = sorted(set(keywords) & _YAMBO_OWNED)
    if stated:
        raise ValueError(
            f"yambo {', '.join(stated)} is owned: the route determines it and forces "
            f"its own value, so the value given here would be discarded. Drop it from "
            f"bse_parameters."
        )


def _owned_yambo[T: Mapping[str, Any]](keywords: T) -> T:
    """Return ``keywords`` after checking every one of them is an owned yambo keyword.

    Raises:
        ValueError: If a keyword is not in :data:`_YAMBO_OWNED`.
    """
    undeclared = sorted(set(keywords) - _YAMBO_OWNED)
    if undeclared:
        raise ValueError(
            f"the route forces yambo {', '.join(undeclared)}, which _YAMBO_OWNED does "
            f"not classify. Add it there."
        )
    return keywords


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
    graph body -- never at ``bethe_salpeter.py`` import time -- keeps every route
    that never touches BSE working with no profile loaded yet.
    """
    from aiida.plugins import WorkflowFactory

    return WorkflowFactory("yambo.yambo.yambowf")


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


class BetheSalpeterOutputs(TypedDict, total=False):
    """Outputs of :func:`RunBetheSalpeter`.

    The BSE ``YamboWorkflow``'s own outputs, plus its QP database.
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
def RunBetheSalpeter(
    codes: BetheSalpeterCodes,
    structure: orm.StructureData,
    kpoints: orm.KpointsData,
    nscf_output_band: orm.BandsData,
    ecutwfc: float,
    ecutrho: float,
    ham_output_parameters: dict,
    bse_parameters: dict,
    eigenvalues: str = "ki",
    pseudo_family: str | None = None,
    protocol: str | None = None,
    protocol_qe: str | None = None,
    parallelization: ParallelizationDict | None = None,
) -> BetheSalpeterOutputs:
    """Run a yambo BSE spectrum seeded by Koopmans (KI) quasiparticle corrections.

    A fresh scf -> nscf -> p2y ("yambo init") builds the yambo SAVE
    directory: the koopmans nscf cannot be reused, since kcw.x needs an
    nspin=2 scratch and yambo's own p2y convention differs. ``kpoints`` (a
    Monkhorst-Pack mesh) must be the mesh the koopmans nscf ran on -- it
    seeds the yambo init's own nscf k-mesh, so the resulting SAVE
    directory's k-grid lines up with ``nscf_output_band`` /
    ``ham_output_parameters``, the grid :func:`generate_qp_database` maps
    the Koopmans eigenvalues onto.

    ``ecutwfc`` / ``ecutrho`` are the koopmans nscf's own plane-wave cutoffs,
    in Ry: both reach the init and BSE steps' scf/nscf ``SYSTEM`` overrides
    directly. ``PwBaseWorkChain.get_builder_from_protocol`` applies overrides
    on top of the pseudo family's recommended cutoffs, and only skips that
    recommendation when both ``ecutwfc`` and ``ecutrho`` are present in
    the override together -- passing ``ecutwfc`` alone still wins for
    ``ecutwfc`` itself, but leaves ``ecutrho`` at the family's own
    recommendation instead of this run's.

    ``ham_output_parameters`` / ``nscf_output_band`` are a kcw.x ``ham``
    run's parsed ``output_parameters`` and its own seeding nscf's
    ``output_band`` (see :func:`generate_qp_database`, which this graph
    calls). ``eigenvalues`` selects which Koopmans flavor seeds the QP
    database; only ``'ki'`` is accepted here -- ``'pki'`` is refused, since
    no ak2 kcw.x ``ham`` parser yet emits ``pki_eigenvalues_on_grid``
    (tracked in aiida-koopmans#134).

    ``bse_parameters`` is the yambo BSE runcard's own ``arguments`` /
    ``variables`` (``BndsRnXs``, ``NGsBlkXs``, ``BSENGBlk``, ``BSEBands``,
    ``BEnRange``, ``BEnSteps``, ``BDmRange``, each ``[value, unit]`` per
    yambopy's convention) and ``metadata`` (the BSE step's own scheduler
    options). ``variables['KfnQPdb']`` is this graph's own -- pointing at
    the QP database it builds -- and refused if the caller states it, as is
    ``variables['BS_CPU']``/``['BS_ROLEs']`` (this graph's own MPI-role
    split, below): both are in :data:`_YAMBO_OWNED`. The same
    ``arguments``/``variables`` also reach the init step (minus ``KfnQPdb``),
    so its own nscf sees the same ``BndsRnXs`` as the BSE step's:
    ``YamboWorkflow.get_builder_from_protocol`` sizes each nscf's own
    ``nbnd`` off the runcard's requested band range, and a mismatch between
    the two nscf's ``nbnd`` makes ``YamboWorkflow`` redo the BSE step's nscf
    and p2y at run time, against a fresh SAVE that the already-built
    quasiparticle database was not made from.
    ``parallelization``'s ``pw`` entry reaches every scf/nscf pw.x step (init
    and BSE alike); its ``yambo`` entry sets the BSE step's rank count and,
    past one rank, a k-point-only ``BS_CPU``/``BS_ROLEs`` MPI split (see
    :func:`_bse_mpi_roles`).

    ``protocol_qe`` defaults to ``'moderate'`` on its own, independent of
    ``protocol``: a caller passing ``protocol='precise'`` without also
    setting ``protocol_qe`` still gets a ``'moderate'``-precision fresh
    scf/nscf/p2y here.

    Raises:
        ValueError: If ``bse_parameters['variables']`` states an owned
            ``"yambo"`` keyword (``KfnQPdb``, ``BS_CPU``, ``BS_ROLEs``).
        NotImplementedError: If ``eigenvalues`` is ``'pki'``.
    """
    validate_parallelization(parallelization)

    if eigenvalues == "pki":
        raise NotImplementedError(
            "eigenvalues='pki' has no producing parser yet -- no ak2 kcw.x `ham` parser "
            "emits `pki_eigenvalues_on_grid` (aiida-koopmans#134). Use 'ki'."
        )

    # ``.build()`` executes graph bodies eagerly, where graph inputs arrive as
    # provenance-tagged proxies; the family label ends up bound as an SQL
    # parameter inside ``get_builder_from_protocol``, which needs a plain str
    # (same fix as ``run_bands_step`` in ``workgraphs/pw.py``).
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None

    variables = dict((bse_parameters or {}).get("variables", {}).items())
    _reject_owned_yambo(variables)
    arguments = list((bse_parameters or {}).get("arguments", []))
    bse_metadata = dict((bse_parameters or {}).get("metadata", {}).items())

    cutoff_system = {"SYSTEM": {"ecutwfc": float(ecutwfc), "ecutrho": float(ecutrho)}}

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

    def _pw_overrides() -> dict[str, Any]:
        """Return fresh scf/nscf ``pw`` overrides: both cutoffs, parallelization applied.

        Built fresh per call: :func:`merge_parallelization_into_overrides`
        mutates its ``overrides`` argument in place, and the init and BSE
        steps each need their own dict.
        """
        overrides: dict[str, Any] = {
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
        merge_parallelization_into_overrides(
            overrides, parallelization, [(("scf", "pw"), "pw"), (("nscf", "pw"), "pw")]
        )
        return overrides

    init_overrides = {
        **_pw_overrides(),
        # Same runcard variables the BSE step gets (minus 'KfnQPdb', which only
        # the BSE step points at a QP database): both steps' nscf must agree on
        # the band range YamboWorkflow.get_builder_from_protocol derives 'nbnd'
        # from, or it redoes the BSE step's nscf+p2y at run time against a SAVE
        # the QP database was never built from.
        "yres": {
            "yambo": {"parameters": {"arguments": arguments, "variables": deepcopy(variables)}},
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

    bse_variables = {**variables, **_owned_yambo({"KfnQPdb": "E < ./ndb.QP"})}
    bse_overrides = {
        **_pw_overrides(),
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
    # ``parameters``: rebuild it to strip the GW-only leftover keys and add
    # the MPI-role split, the same get-then-rebuild idiom the k2y BSE example
    # script uses on the same builder output.
    bse_params_dict = bse_data["yres"]["yambo"]["parameters"].get_dict()
    for gw_only_key in _GW_ONLY_RUNCARD_KEYS:
        bse_params_dict["variables"].pop(gw_only_key, None)
    bse_params_dict["variables"].update(_owned_yambo(_bse_mpi_roles(parallelization)))
    bse_data["yres"]["yambo"]["parameters"] = orm.Dict(bse_params_dict)
    bse_data["yres"]["yambo"]["QP_corrections"] = qp_db
    merge_parallelization_into_inputs(bse_data["yres"]["yambo"], parallelization, "yambo")
    bse_data.setdefault("metadata", {})["call_link_label"] = "bse"
    bse = yambo_step(**bse_data)

    return BetheSalpeterOutputs(
        remote_folder=bse["remote_folder"],
        retrieved=bse["retrieved"],
        output_parameters=bse["output_parameters"],
        output_ywfl_parameters=bse["output_ywfl_parameters"],
        array_eps=bse["array_eps"],
        array_excitonic_states=bse["array_excitonic_states"],
        qp_database=qp_db,
    )


class SinglepointBetheSalpeterCodes(DfptCodes, BetheSalpeterCodes):  # type: ignore[misc]
    """Codes for :func:`SinglepointBetheSalpeterWorkflow`: the DFPT chain's codes plus BSE's own.

    Both parents declare ``pw`` as the same ``PwCode`` -- mypy flags any
    TypedDict multiple-inheritance merge that redeclares a field, even with
    an identical type, hence the ignore.
    """


class SinglepointBetheSalpeterOutputs(TypedDict):
    """Outputs of :func:`SinglepointBetheSalpeterWorkflow`.

    The DFPT chain's outputs, plus the BSE run.
    """

    dfpt: KoopmansDFPTOutputs
    bse: BetheSalpeterOutputs


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


def _dfpt_codes_from(codes: SinglepointBetheSalpeterCodes) -> DfptCodes:
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


def _bse_codes_from(codes: SinglepointBetheSalpeterCodes) -> BetheSalpeterCodes:
    """Return :func:`RunBetheSalpeter`'s codes namespace out of the composed BSE codes."""
    bse_codes: dict[str, Any] = {
        name: reference(codes, name) for name in BetheSalpeterCodes.__required_keys__
    }
    return cast("BetheSalpeterCodes", bse_codes)


def _pw_cutoffs_from(overrides: WannierizeOverrides | None) -> tuple[float, float]:
    """Return the shared ``(ecutwfc, ecutrho)`` in Ry out of the DFPT chain's own scf pw overrides.

    ``overrides['scf']['pw']['parameters']['SYSTEM']`` is where a caller
    already states both cutoffs for
    :func:`~aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow` (see
    :func:`~aiida_koopmans.workgraphs.block_wannierize._builder_overrides`,
    which reads the same namespace); :func:`RunBetheSalpeter`'s own fresh
    scf/nscf reuses those values directly, rather than re-deriving them from
    a parsed pw.x output.

    Raises:
        ValueError: If either cutoff is absent from the ``SYSTEM`` namelist.
    """
    system = (overrides or {}).get("scf", {}).get("pw", {}).get("parameters", {}).get("SYSTEM", {})
    missing = [name for name in ("ecutwfc", "ecutrho") if name not in system]
    if missing:
        raise ValueError(
            f"SinglepointBetheSalpeterWorkflow is missing {missing} in "
            "overrides['scf']['pw']['parameters']['SYSTEM'] -- RunBetheSalpeter's own fresh "
            "scf/nscf need the same cutoffs the DFPT chain runs on. Set both there."
        )
    return float(system["ecutwfc"]), float(system["ecutrho"])


@task.graph
def SinglepointBetheSalpeterWorkflow(
    codes: SinglepointBetheSalpeterCodes,
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
) -> SinglepointBetheSalpeterOutputs:
    """Run a Koopmans DFPT singlepoint, then a BSE spectrum seeded by its eigenvalues.

    Composes :func:`~aiida_koopmans.workgraphs.dfpt.SinglepointDFPTWorkflow`
    with :func:`RunBetheSalpeter`: the DFPT chain's shared ground state's
    ``nscf_output_band``, the ``ecutwfc``/``ecutrho`` already sitting in
    ``overrides['scf']['pw']['parameters']['SYSTEM']``, and the ``none``
    channel's ``ham_parameters`` feed the BSE step directly, so a caller
    states the DFPT/wannierization inputs once.

    Phase-1 scope, refused explicitly:

    * ``spin`` is always ``NONE`` -- ``manifolds`` must carry exactly one
      ``"none"`` key. A collinear or spinor DFPT chain has no single
      channel for the BSE step to read; run
      ``SinglepointDFPTWorkflow`` and :func:`RunBetheSalpeter` separately for those.
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
    protocol and :func:`RunBetheSalpeter`'s yambo protocol. ``protocol_qe`` sets only
    the BSE route's own fresh scf/nscf/p2y -- pass it when that QE step
    should run at a different precision than ``protocol``.
    ``bse_parameters`` / ``eigenvalues`` / the BSE half of
    ``parallelization`` pass straight to :func:`RunBetheSalpeter`; every other
    argument passes straight to ``SinglepointDFPTWorkflow``.
    """
    if set(manifolds) != {"none"}:
        raise NotImplementedError(
            "SinglepointBetheSalpeterWorkflow only supports spin='none' (manifolds keyed by a "
            f"single 'none' entry), got manifold keys {sorted(manifolds)}. Run "
            "SinglepointDFPTWorkflow and RunBetheSalpeter separately for a collinear or spinor "
            "chain."
        )
    if not all(structure.pbc):
        raise NotImplementedError(
            "SinglepointBetheSalpeterWorkflow supports periodic structures only: yambo's p2y step "
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
    ecutwfc, ecutrho = _pw_cutoffs_from(overrides)
    bse = RunBetheSalpeter(
        codes=_bse_codes_from(codes),
        structure=structure,
        kpoints=kpoints,
        nscf_output_band=dfpt["ground_state"]["nscf_output_band"],
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        ham_output_parameters=channel["ham_parameters"],
        bse_parameters=bse_parameters,
        eigenvalues=eigenvalues,
        pseudo_family=pseudo_family,
        protocol=protocol,
        protocol_qe=protocol_qe,
        parallelization=parallelization,
        metadata={"call_link_label": "bse", "label": "BSE spectrum"},
    )
    return SinglepointBetheSalpeterOutputs(dfpt=dfpt, bse=bse)
