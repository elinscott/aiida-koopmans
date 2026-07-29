"""Block-by-block Wannierisation of a periodic system.

A single shared scf + nscf is run once (via :func:`RunScfNscf`, or skipped
entirely when the caller supplies an existing ``nscf_remote_folder``), then
each projection block (occupied / empty manifold, per spin) is Wannierised
in its own ``Wannier90WorkChain`` that *skips* scf and nscf and reads the
shared nscf scratch directly. The per-block fan-out is a native ``for``
loop over ``blocks`` inside the ``@task.graph`` body -- do not convert it
to a ``Map`` zone. Results are collected into a dict keyed by each block's
stable ``label`` (e.g. ``"block_1"`` / ``"block_1_spin_up"``) and returned
as a dynamic output namespace.

Per-block file staging that the supercell fold consumes:

* ``retrieved`` -- the wannier90 ``retrieved`` :class:`~aiida.orm.FolderData`,
  which holds ``aiida_hr.dat`` (the real-space Hamiltonian, written because
  ``write_hr=True``) plus ``aiida.chk``, ``aiida_u.mat``, ``aiida_centres.xyz``
  and, for disentangling blocks, ``aiida_u_dis.mat``. All but ``aiida.chk``
  are retrieved by upstream's default suffix list once written (the ``write_*``
  pins are what guarantee they exist); ``aiida.chk`` is force-retrieved.
  Downstream consumers such as pw2wannier90 ``wan_mode='decompose'`` and the
  wannierjl split read them.
* ``remote_folder`` -- the wannier90 ``RemoteData`` scratch.
* ``nnkp_file`` -- the ``aiida.nnkp`` :class:`~aiida.orm.SinglefileData`
  emitted by the wannier90 post-processing (``-pp``) run.

Alongside the file staging, each block also exposes the parsed wannier90
``output_parameters`` :class:`~aiida.orm.Dict` (per-WF spreads / centres,
Omega decomposition), so downstream consumers that depend on parsed
quantities — e.g. the DFPT spread-based orbital grouping — read them from
the parser output rather than re-parsing the raw ``.wout``.

Because every downstream code consumes a *unified* view of the
Wannierisation (kcw.x reads one occupied + one empty file set, the fold
route merges per manifold), :func:`WannierizeBlocks` also emits unified,
band-ordered ``centres`` and ``spreads`` arrays concatenated across all
blocks by :func:`collect_wannier_functions`. Band order is taken from the
input ``blocks`` list order — the single authority — never reconstructed
from block labels or output-namespace keys.

:func:`WannierizeBlocks` also carries an optional split mode (triggered by a
``bands_kpoints`` input): a pw.x ``bands`` run feeds a runtime band-group
detection, and each block is handled by the automated block-splitting chain
of :mod:`aiida_koopmans.workgraphs.auto_wannierize` instead of a plain
:func:`WannierizeBlock`. The split machinery is lazy-imported inside the
graph body: it depends on ``aiida-wannierjl``, which the plain mode must not
require.
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.workflows import Wannier90WorkChain
from aiida_workgraph import dynamic, task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.types import (
    ParallelizationDict,
    ProjectionBlock,
    SpinChannel,
    VariationalOrbital,
    block_w90_kwargs,
)
from aiida_koopmans.workgraphs import (
    Codes,
    merge_parallelization_into_inputs,
    validate_parallelization,
)
from aiida_koopmans.workgraphs.pw import PwOutputs, RunScfNscf
from aiida_koopmans.workgraphs.variational_orbitals import (
    BlockOrbitalSpec,
    initial_orbital_partition,
    ordered_block_specs,
)
from aiida_koopmans.workgraphs.wannier90 import Wannier90Step

# ``aiida.chk`` is the only wannier90 product upstream excludes from its
# retrieve-everything default: ``_DEFAULT_RETRIEVE_SUFFIXES`` in
# aiida-wannier90's ``Wannier90Calculation`` already covers ``_u.mat`` /
# ``_u_dis.mat`` / ``_centres.xyz`` / ``_hr.dat``, so once those files are
# written they land in ``retrieved`` automatically. What guarantees the
# product set is therefore the ``write_hr`` / ``write_u_matrices`` /
# ``write_xyz`` pins below, not this list. The supercell fold needs
# ``aiida.chk`` to unitarily rotate the per-block manifolds, so force it.
_W90_RETRIEVE_SETTINGS: dict[str, list[str]] = {"additional_retrieve_list": ["aiida.chk"]}

#: Projection sources the block wannierization supports: explicit orbital
#: lists, and — for automated wannierization — pseudoatomic orbitals fetched
#: from the pseudopotentials.
SUPPORTED_PROJECTION_TYPES = (
    WannierProjectionType.ANALYTIC,
    WannierProjectionType.ATOMIC_PROJECTORS_QE,
)


def validate_projection_type(projection_type: WannierProjectionType) -> None:
    """Reject projection types the block wannierization does not support.

    ``ATOMIC_PROJECTORS_EXTERNAL`` (pseudoatomic orbitals from an external
    directory) is the intended second automated source, but the external
    projector inputs are not wired through the per-block builder yet, so it
    is rejected naming that gap; every other type (SCDM, random, ...) is out
    of scope. Compares by ``.value``: in a graph body the enum arrives as a
    provenance-tagged proxy, which fails the ``Enum`` constructor's by-value
    lookup, while attribute access and ``==`` delegate cleanly.
    """
    value = getattr(projection_type, "value", projection_type)
    if any(value == member.value for member in SUPPORTED_PROJECTION_TYPES):
        return
    if value == WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL.value:
        raise ValueError(
            "Projection type 'atomic_projectors_external' is not supported yet: the "
            "external projector directory and tables are not wired through the "
            "per-block wannier builder."
        )
    raise ValueError(
        f"Projection type '{value}' is not supported: blocks are "
        "wannierized from explicit projections ('analytic') or pseudoatomic "
        "projectors fetched from the pseudopotentials ('atomic_projectors_qe')."
    )


class WannierizeOverrides(TypedDict, total=False):
    """Flat, semantic overrides for :func:`WannierizeBlocks` / :func:`WannierizeBlock`.

    Deliberately NOT the upstream namespace-mirroring override shape
    (``wannier90.wannier90.parameters...``): that nesting stutters, is easy
    to mis-wrap, and a wrong depth is silently ignored by
    ``recursive_merge``. The upstream builder shape is produced in exactly
    one place — the builder call inside :func:`WannierizeBlock`.

    * ``scf`` / ``nscf`` — ``PwBaseWorkChain``-protocol override dicts for
      the shared scf/nscf pair (upstream shape, consumed verbatim by
      :func:`RunScfNscf`).
    * ``wannier90`` — a flat ``.win`` keyword dict (e.g.
      ``{"dis_froz_max": 10.6}``) applied to every block's wannier90.
    * ``pw2wannier90`` — a flat ``INPUTPP`` keyword dict (e.g.
      ``{"write_unk": True}``) applied to every block's pw2wannier90.
    """

    scf: dict[str, Any]
    nscf: dict[str, Any]
    wannier90: dict[str, Any]
    pw2wannier90: dict[str, Any]


class WannierizeBlockOutputs(TypedDict):
    """The flat per-block contract, and the entry shape of ``blocks``.

    Every :func:`WannierizeBlocks` mode declares this socket set per block,
    and a field never changes meaning per route: anything that would (the
    whole-block run's folders, whose product files describe the pre-split
    gauge on the split route) is populated on the plain route only. If a
    split consumer someday needs the whole-block artifacts, they get
    explicitly named new optional fields — never overloaded ones.

    Always populated, always the entry's *final* gauge:

    * ``u_file`` / ``hr_file`` / ``centres_file`` -- the gauge-product trio
      (``aiida_u.mat`` / ``aiida_hr.dat`` / ``aiida_centres.xyz``):
      extracted from the wannier90 ``retrieved`` folder for a
      plainly-Wannierised block, merged block-diagonally from the per-group
      runs for a split one.
    * ``nnkp_file`` -- the ``aiida.nnkp`` SinglefileData from the ``-pp``
      run (gauge-independent, hence shared by both routes).
    * ``output_parameters`` -- the parsed wannier90 output Dict, holding at
      least the per-WF ``wannier_functions_output`` table (spreads /
      centres, 1-based block-wide ``wf_ids``) and ``number_wfs``: the
      producing run's Dict for a plainly-Wannierised block, the per-group
      parsed outputs concatenated in band order for a split one (whose
      merged Dict carries only the honestly mergeable keys).

    Populated on the plain route only, because on the split route no folder
    of the final gauge exists (the whole-block run's folders describe the
    pre-split gauge, and a field never changes meaning). Split entries
    leave them unpopulated (consumers read ``None`` at runtime, not a
    ``KeyError``), so e.g. a decompose-style consumer reading ``retrieved``
    off a split entry fails loudly instead of silently working with the
    pre-split gauge. All current readers of these fields (``RunDFPT``,
    ``FoldToSupercell``, the decompose dataset route) are fed by plain-mode
    ``WannierizeBlocks`` calls, so none is affected:

    * ``retrieved`` -- the wannier90 ``retrieved`` FolderData (holds
      ``aiida_hr.dat``, ``aiida.chk``, ``aiida_u.mat``, ``aiida_centres.xyz``
      and, when the block disentangles, ``aiida_u_dis.mat``).
    * ``remote_folder`` -- the wannier90 ``RemoteData`` scratch.
    """

    u_file: orm.SinglefileData
    hr_file: orm.SinglefileData
    centres_file: orm.SinglefileData
    nnkp_file: orm.SinglefileData
    output_parameters: orm.Dict
    retrieved: NotRequired[orm.FolderData]
    remote_folder: NotRequired[orm.RemoteData]


class CollectedWannierFunctions(TypedDict):
    """Outputs of :func:`collect_wannier_functions`.

    * ``centres`` -- per-WF centres as ``[x, y, z]`` lists (Å), band-ordered
      across all blocks. Coordinates the upstream parser could not read are
      ``None`` (it None-pads individually), so consumers that need numbers
      must check.
    * ``spreads`` -- per-WF final-state spreads (Å²), same ordering. Strict:
      a block without a parsed final-state spread table is rejected.
    """

    centres: list
    spreads: list


class WannierizeBlocksOutputs(TypedDict):
    """Outputs of :func:`WannierizeBlocks`.

    * ``blocks`` -- a dynamic namespace keyed by block label; every entry is
      the uniform :class:`WannierizeBlockOutputs` set, identical across
      modes, consumable downstream as a namespace.
    * ``centres`` / ``spreads`` -- the unified, band-ordered per-WF arrays of
      :class:`CollectedWannierFunctions`, concatenated across all blocks in
      input-list order (every downstream code wants the unified view);
      final-gauge in both modes.
    * ``nscf`` -- the shared nscf :class:`PwOutputs` so the supercell fold
      can read ``nscf["remote_folder"]`` (the nscf scratch every block was
      built on). Absent when the caller supplied its own
      ``nscf_remote_folder`` and the internal scf + nscf was skipped.
    * ``bands`` / ``groups`` -- split mode only: the pw.x ``bands``-run
      eigenvalues the grouping was detected on, and the detected 1-indexed
      band groups (global indices).
    * ``orbitals`` -- one :class:`~aiida_koopmans.types.VariationalOrbital`
      per Wannier function in per-channel iwann order (occupied ascending
      then empty ascending — the kcw.x / kcp.x orbital order), with
      ``manifold`` = the block label, ``filled`` / ``spin`` from the
      block, and ``group_id`` encoding the coarsest screening partition
      consistent with the blocks' exact splits (filling, spin and
      manifold — :func:`~aiida_koopmans.workgraphs.variational_orbitals.initial_orbital_partition`).
      Emitted only when every input block carries a ``filled`` occupancy
      stamp; identical across plain and split modes (a split block's
      orbitals keep the parent block's label as their ``manifold``).
      Engine storage returns the list with each orbital's ``spin``
      degraded to a plain ``str`` — consumers compare with ``==``, never
      ``is`` (see :class:`~aiida_koopmans.types.VariationalOrbital`).
    """

    blocks: Annotated[dict, dynamic(WannierizeBlockOutputs)]
    centres: list
    spreads: list
    nscf: NotRequired[PwOutputs]
    bands: NotRequired[orm.BandsData]
    groups: NotRequired[list[list[int]]]
    orbitals: NotRequired[list[VariationalOrbital]]


def _builder_overrides(overrides: WannierizeOverrides) -> dict[str, Any] | None:
    """Wrap the flat keyword dicts into the upstream builder override shape.

    The ONLY place the upstream override nesting is produced. The protocol
    overrides mirror the workchain's input namespace tree: base-workchain
    namespace -> calculation namespace -> ``parameters`` — hence
    ``wannier90.wannier90.parameters`` for ``.win`` keywords and
    ``pw2wannier90.pw2wannier90.parameters.INPUTPP`` for the pw2wannier90
    namelist. Callers supply the flat :class:`WannierizeOverrides` and never
    touch this shape.
    """
    wannier90 = overrides.get("wannier90")
    pw2wannier90 = overrides.get("pw2wannier90")
    builder_overrides: dict[str, Any] = {}
    if wannier90:
        builder_overrides["wannier90"] = {"wannier90": {"parameters": dict(wannier90)}}
    if pw2wannier90:
        builder_overrides["pw2wannier90"] = {
            "pw2wannier90": {"parameters": {"INPUTPP": dict(pw2wannier90)}}
        }
    return builder_overrides or None


@task.calcfunction(outputs=["u_file", "hr_file", "centres_file"])
def extract_wannier_output_files(retrieved: orm.FolderData) -> dict:
    """Pull the gauge-product trio out of a wannier90 ``retrieved`` folder.

    Wraps ``aiida_u.mat`` / ``aiida_hr.dat`` / ``aiida_centres.xyz`` as
    individual :class:`~aiida.orm.SinglefileData` nodes so a plainly
    Wannierised block exposes the same file sockets as a split one. The
    files exist because :func:`WannierizeBlock` pins ``write_hr`` /
    ``write_u_matrices`` / ``write_xyz`` (upstream's default retrieve
    suffixes then pick them up). A calcfunction (not a plain ``@task``): it
    takes an AiiDA data node, which the PyFunction deserializer refuses.
    """
    import io

    def _single(filename: str) -> orm.SinglefileData:
        if filename not in retrieved.base.repository.list_object_names():
            raise ValueError(
                f"``{filename}`` is missing from the wannier90 retrieved folder. "
                "The wannier90 run must set ``write_hr = True``, "
                "``write_u_matrices = True`` and ``write_xyz = True``."
            )
        content = retrieved.base.repository.get_object_content(filename, mode="rb")
        return orm.SinglefileData(io.BytesIO(content), filename=filename)

    return {
        "u_file": _single("aiida_u.mat"),
        "hr_file": _single("aiida_hr.dat"),
        "centres_file": _single("aiida_centres.xyz"),
    }


@task.graph
def WannierizeBlock(
    codes: Codes,
    structure: orm.StructureData,
    block: ProjectionBlock,
    projection_type: WannierProjectionType,
    nscf_remote_folder: orm.RemoteData,
    kpoints: orm.KpointsData,
    mp_grid: list[int] | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
    parallelization: ParallelizationDict | None = None,
) -> WannierizeBlockOutputs:
    """Wannierise a single projection block off the shared nscf scratch.

    ``overrides`` is the flat :class:`WannierizeOverrides`; this block-level
    graph consumes its ``wannier90`` / ``pw2wannier90`` entries
    (the ``scf`` / ``nscf`` entries belong to the shared scf+nscf pair and
    are ignored here). This function is the single place the flat keyword
    dicts are wrapped into the upstream builder's namespace-mirroring
    override shape.

    Seeds a ``Wannier90WorkChain`` builder via ``get_builder_from_protocol``
    for this block's ``projection_type``, then:

    * pops the ``scf`` namespace and the ``nscf`` namespace so the workchain
      skips both steps (upstream gates each on ``"scf" in inputs`` /
      ``"nscf" in inputs``), and points the pw2wannier90 step at the shared
      nscf scratch via ``pw2wannier90.pw2wannier90.parent_folder`` -- the only
      parent the validator accepts once both scf and nscf are absent;
    * overrides the per-block ``num_wann`` / ``num_bands`` / ``exclude_bands``
      (and ``projections`` for explicit blocks) from
      :func:`block_w90_kwargs`;
    * forces ``write_hr`` / ``write_u_matrices`` / ``write_xyz`` so
      ``aiida_hr.dat`` / ``aiida_u.mat`` / ``aiida_u_dis.mat`` /
      ``aiida_centres.xyz`` are written (upstream's default retrieve list then
      picks them up), and force-retrieves ``aiida.chk``, which upstream
      excludes by default.
    """
    validate_projection_type(projection_type)
    overrides = overrides or {}
    wannier90 = overrides.get("wannier90")

    # ``.build()`` executes this body eagerly, where graph inputs arrive as
    # provenance-tagged proxies; the family label ends up bound as an SQL
    # parameter inside ``get_builder_from_protocol``, which needs a plain str.
    pseudo_family = str(pseudo_family) if pseudo_family is not None else None

    builder = Wannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        protocol=protocol,
        overrides=_builder_overrides(overrides),
        pseudo_family=pseudo_family,
        electronic_type=electronic_type,
        spin_type=spin_type,
        projection_type=projection_type,
        # For koopmans we do not exclude semicore states automatically.
        exclude_semicore=False,
        # The hamiltonian-retrieval protocol override sets ``write_hr`` /
        # ``write_tb`` and the hr retrieve handling.
        retrieve_hamiltonian=True,
        print_summary=False,
    )
    # Flatten to a plain dict up front; every edit below is a dict edit.
    data = get_dict_from_builder(builder)
    w90 = data["wannier90"]["wannier90"]

    # --- per-block wannier90 parameters / projections ---
    w90_kwargs = block_w90_kwargs(block)
    w90_params = w90["parameters"].get_dict()
    w90_params["num_wann"] = w90_kwargs["num_wann"]
    w90_params["num_bands"] = w90_kwargs["num_bands"]
    if "exclude_bands" in w90_kwargs:
        w90_params["exclude_bands"] = w90_kwargs["exclude_bands"]
    else:
        # A block that excludes nothing must not inherit an exclusion the
        # protocol machinery may have seeded.
        w90_params.pop("exclude_bands", None)
    # Per-block disentanglement handling: a block with extra bands genuinely
    # disentangles, so give it wannier90's real default iteration budget (the
    # aiida-wannier90-workflows protocol pins ``dis_num_iter: 0``, which
    # freezes the initial projection subspace); a block with
    # num_bands == num_wann cannot disentangle, so strip the (globally
    # supplied) windows outright.
    if w90_kwargs["num_bands"] != w90_kwargs["num_wann"]:
        if "dis_num_iter" not in (wannier90 or {}):
            w90_params["dis_num_iter"] = 5000
    else:
        for key in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"):
            w90_params.pop(key, None)
    # ``write_hr`` is set by the retrieve_hamiltonian override above; pin it
    # explicitly so a stripped-down override dict can't silently drop it.
    # ``write_u_matrices`` / ``write_xyz`` produce the U matrices and Wannier
    # centres that pw2wannier90 ``wan_mode='decompose'`` and the wannierjl
    # split consume.
    w90_params["write_hr"] = True
    w90_params["write_u_matrices"] = True
    w90_params["write_xyz"] = True
    # The protocol builder froze ``mp_grid`` from its own distance-derived
    # mesh, which goes stale once the shared k-list is substituted below.
    # Pin the real mesh dimensions when given (wannier90 cannot re-derive
    # them from an explicit list); otherwise drop the key so a mesh
    # ``kpoints`` input lets the calculation re-derive it.
    if mp_grid is not None:
        w90_params["mp_grid"] = mp_grid
    else:
        w90_params.pop("mp_grid", None)
    w90["parameters"] = orm.Dict(w90_params)

    # Explicit (ANALYTIC) blocks carry resolved projection orbitals; automatic
    # blocks rely on ``projection_type`` alone (no ``projections`` key).
    if "projections" in w90_kwargs:
        w90["projections"] = orm.List(list=w90_kwargs["projections"])

    # Share the nscf k-mesh so the per-block wannier90 / pw2wannier90 read
    # eigenstates on the exact grid the shared nscf produced.
    w90["kpoints"] = kpoints

    # Force-retrieve ``aiida.chk`` (upstream's only non-default product), merged
    # on top of whatever ``settings`` the protocol set; the workchain only adds
    # its own ``postproc_setup`` key on top of this.
    existing_settings: dict = {}
    if "settings" in w90:
        existing_settings = w90["settings"].get_dict()
    existing_settings.update(_W90_RETRIEVE_SETTINGS)
    w90["settings"] = orm.Dict(existing_settings)

    # Skip scf + nscf and reuse the shared nscf scratch. With both namespaces
    # absent the workchain validator requires the parent on the pw2wannier90
    # step.
    data.pop("scf", None)
    data.pop("nscf", None)
    data.pop("clean_workdir", None)
    data["pw2wannier90"]["pw2wannier90"]["parent_folder"] = nscf_remote_folder

    # Per-code parallelization: wannier90.x takes ntasks only (no pool/pd
    # concept); pw2wannier90.x takes ntasks plus -npool / -pd. QE rejects
    # pw2wannier90 pools under gamma_only, but this block wannierization is a
    # periodic (full-grid nscf) path, so no schema guard is needed here.
    merge_parallelization_into_inputs(data["wannier90"]["wannier90"], parallelization, "wannier90")
    merge_parallelization_into_inputs(
        data["pw2wannier90"]["pw2wannier90"], parallelization, "pw2wannier90"
    )

    data.setdefault("metadata", {})["call_link_label"] = "wannier90"
    outputs = Wannier90Step(**data)

    output_files = extract_wannier_output_files(
        retrieved=outputs["wannier90"]["retrieved"],
        metadata={"call_link_label": "extract_wannier_output_files"},
    )

    return WannierizeBlockOutputs(
        u_file=output_files["u_file"],
        hr_file=output_files["hr_file"],
        centres_file=output_files["centres_file"],
        retrieved=outputs["wannier90"]["retrieved"],
        remote_folder=outputs["wannier90"]["remote_folder"],
        nnkp_file=outputs["wannier90_pp"]["nnkp_file"],
        output_parameters=outputs["wannier90"]["output_parameters"],
    )


@task
def collect_wannier_functions(
    output_parameters: Annotated[dict, dynamic(orm.Dict)],
) -> CollectedWannierFunctions:
    """Concatenate per-block parsed wannier90 outputs into unified arrays.

    Walks each block's ``output_parameters`` (arriving as plain dicts via
    aiida-pythonjob's built-in ``Dict`` deserializer) and concatenates the
    final-state per-WF centres and spreads from ``wannier_functions_output``
    (a list of ``{wf_ids, wf_centres, wf_spreads}`` dicts with 1-based
    ``wf_ids``; distinct from the manifold-total ``Omega_*`` scalars) into
    one band-ordered array pair. Within a block the entries are ordered by
    ``wf_ids``.

    The input namespace is keyed ``b{i:02d}`` by the block's position in
    :func:`WannierizeBlocks`'s band-ordered input list. That keying is a
    private transport detail between the graph body and this task (producer
    and consumer sit a few lines apart) — it is *not* a cross-graph
    contract, and no other code may rely on it.
    """
    centres: list[list[float | None]] = []
    spreads: list[float] = []
    for key in sorted(output_parameters):
        parameters = output_parameters[key]
        wfs = parameters.get("wannier_functions_output") or []
        if len(wfs) != parameters.get("number_wfs"):
            raise ValueError(
                f"A block's wannier90 ``output_parameters`` lists {len(wfs)} "
                "final-state Wannier functions but the run declares "
                f"number_wfs = {parameters.get('number_wfs')}."
            )
        if any("wf_spreads" not in wf for wf in wfs):
            # A wannier90 restart-for-plotting run parses only wf_ids +
            # im_re_ratio per WF (no final-state spread table).
            raise ValueError(
                "A ``wannier_functions_output`` entry carries no ``wf_spreads`` — "
                "the run did not minimise to a final state (e.g. a "
                "restart-for-plotting run)."
            )
        for wf in sorted(wfs, key=lambda wf: int(wf["wf_ids"])):
            spreads.append(float(wf["wf_spreads"]))
            coords = wf.get("wf_centres") or (None, None, None)
            centres.append([None if c is None else float(c) for c in coords])
    return CollectedWannierFunctions(centres=centres, spreads=spreads)


def _validate_block_projection_types(blocks: list[ProjectionBlock]) -> None:
    """Reject any block whose projection type is unsupported."""
    for block in blocks:
        validate_projection_type(block["projection_type"])


def _maybe_emit_orbital_partition(
    outputs: WannierizeBlocksOutputs, blocks: list[ProjectionBlock]
) -> None:
    """Wire the initial orbital partition into ``outputs`` when the blocks carry occupancy.

    Emission is gated on the caller stamping ``filled`` on every block:
    occupancy is structural information the caller owns, so it travels
    on the blocks and is never derived here. A partial stamping is a
    caller bug rather than a smaller feature, so it raises. The
    partition task takes a JSON-pure reduced view of the blocks: a full
    block carries a non-``str`` enum (``projection_type``) that the
    PyFunction input serializer cannot store.
    """
    stamped = [("filled" in block) for block in blocks]
    if any(stamped) and not all(stamped):
        unstamped = [
            str(block["label"]) for block, has in zip(blocks, stamped, strict=True) if not has
        ]
        raise ValueError(
            "Some blocks carry a `filled` occupancy stamp and some do not "
            f"({unstamped}); stamp every block to emit the orbital partition, "
            "or none to skip it."
        )
    if not (blocks and all(stamped)):
        return
    specs = [
        BlockOrbitalSpec(
            label=str(block["label"]),
            spin=SpinChannel(block["spin"]),
            filled=bool(block["filled"]),
            num_wann=int(block["num_wann"]),
        )
        for block in blocks
    ]
    # The partition lists orbitals in emitted order while the unified
    # ``spreads`` / ``centres`` concatenate in input-list order; a consumer
    # pairing the two positionally would mis-align silently if the orders
    # diverged, so require the input to already be the emitted order.
    ordered = ordered_block_specs(specs)
    if ordered != specs:
        raise ValueError(
            f"The blocks are not in emitted orbital order: got labels "
            f"{[s['label'] for s in specs]} but the partition walks "
            f"{[s['label'] for s in ordered]} (spin channels up, down, none — each "
            "contiguous, with every occupied block before every empty one). Reorder "
            "the blocks so `orbitals` stays aligned with the input-list-ordered "
            "`spreads` / `centres`."
        )
    partition = initial_orbital_partition(
        blocks=specs,
        metadata={"call_link_label": "initial_orbital_partition"},
    )
    outputs["orbitals"] = partition.result


def _resolve_split_mode(
    codes: Codes,
    blocks: list[ProjectionBlock],
    mp_grid: list[int] | None,
    nscf_remote_folder: orm.RemoteData | None,
    split_threshold: float | None,
    bands_kpoints: orm.KpointsData | None,
    num_occ_bands: int | None,
    wjl_options: dict[str, Any] | None,
    subblock_wannier90_options: dict[str, Any] | None,
    cubic_pw2wannier90_options: dict[str, Any] | None,
) -> bool:
    """Decide split-vs-plain for :func:`WannierizeBlocks` and validate the inputs.

    Split mode triggers on the *need* for splitting: a gap threshold was
    requested (``split_threshold``), or a block's band groups are only
    discovered at runtime (an automatic-projections block — no
    ``projections`` key). ``bands_kpoints`` is a requirement of split mode,
    not its trigger. Plain mode rejects split-only knobs rather than
    silently ignore them; every violation raises a ``ValueError`` naming
    the gap.
    """
    split = split_threshold is not None or any("projections" not in block for block in blocks)
    if not split:
        split_only = {
            "bands_kpoints": bands_kpoints,
            "num_occ_bands": num_occ_bands,
            "wjl_options": wjl_options,
            "subblock_wannier90_options": subblock_wannier90_options,
            "cubic_pw2wannier90_options": cubic_pw2wannier90_options,
        }
        given = [name for name, value in split_only.items() if value is not None]
        if given:
            raise ValueError(
                "Split-only inputs were given without a split trigger "
                f"(`split_threshold` or an automatic-projections block): {given}; "
                "they would be silently ignored."
            )
        return False
    if bands_kpoints is None:
        raise ValueError(
            "Split mode requires `bands_kpoints`: the band-group detection reads "
            "the eigenvalues of a pw.x bands run along it."
        )
    if num_occ_bands is None:
        raise ValueError(
            "Split mode requires `num_occ_bands`: the group detection always "
            "opens a group at the occupied/empty boundary."
        )
    if "wannierjl" not in codes:
        raise ValueError(
            "Split mode requires a `wannierjl` code: the detected groups are "
            "split with Wannier.jl parallel transport."
        )
    if nscf_remote_folder is not None:
        raise ValueError(
            "Split mode cannot build on an external `nscf_remote_folder`: the "
            "bands step the detection reads runs off the internal scf's remote "
            "folder, and an external nscf scratch carries no scf density to run "
            "it from."
        )
    if mp_grid is None:
        raise ValueError(
            "Split mode requires `mp_grid`: the per-group re-Wannierisation "
            "writes it into each sub-block `.win`."
        )
    return True


@task.graph
def WannierizeBlocks(
    codes: Codes,
    structure: orm.StructureData,
    blocks: list[ProjectionBlock],
    kpoints: orm.KpointsData,
    mp_grid: list[int] | None = None,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: WannierizeOverrides | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
    parallelization: ParallelizationDict | None = None,
    nscf_remote_folder: orm.RemoteData | None = None,
    split_threshold: float | None = None,
    bands_kpoints: orm.KpointsData | None = None,
    num_occ_bands: int | None = None,
    wjl_options: dict[str, Any] | None = None,
    subblock_wannier90_options: dict[str, Any] | None = None,
    cubic_pw2wannier90_options: dict[str, Any] | None = None,
) -> WannierizeBlocksOutputs:
    """Wannierise a periodic system block-by-block off one shared scf + nscf.

    A single :func:`RunScfNscf` runs scf + nscf once; every projection
    block is then Wannierised in its own ``Wannier90WorkChain`` that skips
    scf / nscf and reads the shared nscf scratch (``nscf["remote_folder"]``).
    The per-block fan-out is a native ``for`` loop over ``blocks`` inside this
    ``@task.graph`` body; the per-block outputs are collected into a dict
    keyed by block label and returned as the ``blocks`` dynamic namespace,
    and the per-block parsed outputs are concatenated in input-list order
    into the unified ``centres`` / ``spreads`` outputs
    (:func:`collect_wannier_functions`).

    The automated block-splitting mode triggers when a gap threshold was
    requested (``split_threshold``) or when any block uses automatic
    projections, whose band grouping only exists at runtime.
    In split mode a pw.x ``bands`` step runs along the required
    ``bands_kpoints`` off the internal scf density, the runtime
    ``detect_band_groups`` task turns the eigenvalues into energy-separated
    groups (always splitting at the occupied/empty boundary
    ``num_occ_bands``, and at every gap wider than ``split_threshold`` eV),
    and each block is handled by a nested
    :func:`~aiida_koopmans.workgraphs.auto_wannierize.WannierizeAndSplitBlock`
    that receives the resolved groups and splits the block when they divide
    it. Either way every ``blocks`` entry emits the same
    :class:`WannierizeBlockOutputs` socket set — one final-gauge
    ``output_parameters`` per block included — so downstream consumers
    never branch on the mode and the unified ``centres`` / ``spreads``
    are collected unconditionally.

    Args:
        codes: code instances. Required keys: ``pw``, ``wannier90``,
            ``pw2wannier90``; ``projwfc`` is accepted but unused by the
            supported projection types (:data:`SUPPORTED_PROJECTION_TYPES`).
        structure: the periodic ``StructureData``.
        blocks: the resolved projection blocks, in band order (the unified
            outputs concatenate in this order); occupied and empty manifolds
            appear as separate blocks. Each is Wannierised independently.
            When every block carries a ``filled`` occupancy stamp, the
            ``orbitals`` initial screening partition is emitted as well
            (stamping only some blocks raises).
        kpoints: the explicit k-point list shared by the nscf and every
            block's wannier90 / pw2wannier90 (one node, so the k-ordering
            cannot drift between the steps).
        mp_grid: the Monkhorst-Pack dimensions ``kpoints`` was generated
            from. Carried separately because an explicit-list
            ``KpointsData`` cannot represent its parent mesh, and
            wannier90 requires ``mp_grid`` in the ``.win`` (it cannot
            re-derive it from the list).
        pseudo_family: pseudopotential family label.
        protocol: protocol name passed to both builders.
        overrides: optional :class:`WannierizeOverrides` — flat, semantic
            keys (``scf`` / ``nscf`` pw-protocol dicts feed
            :func:`RunScfNscf`; ``wannier90`` / ``pw2wannier90``
            flat keyword dicts feed every per-block wannier builder). Never
            the upstream namespace-nested shape.
        electronic_type / spin_type: forwarded to the wannier builder.
        nscf_remote_folder: an existing nscf scratch to build every block
            on. When given, the internal scf + nscf is skipped (and the
            ``nscf`` output namespace is absent); the caller owns keeping
            ``kpoints`` consistent with the scratch's k-list. This is how a
            workflow with one scratch shared *across* several
            ``WannierizeBlocks`` calls (e.g. one per spin channel) routes
            through here without rerunning the ground state. Incompatible
            with split mode, which needs the internal scf's density for its
            bands step.
        split_threshold: minimum gap (eV) between consecutive bands for a
            split; setting it is one of the two split-mode triggers.
            ``None`` (with automatic-projections blocks supplying the other
            trigger) splits only at the occupied/empty boundary.
        bands_kpoints: split mode only (required there) — the k-path for
            the pw.x ``bands`` run the group detection reads.
        num_occ_bands: split mode only (required there) — occupied-band
            count of the channel (the detection always opens a new group at
            this boundary).
        wjl_options / cubic_pw2wannier90_options: split mode only — optional
            ``metadata.options`` for the Wannier.jl CalcJobs and the cubic
            pw2wannier90 run. Both CalcJobs carry their own ``resources``
            defaults, so normally leave these unset: a non-None dict
            currently trips aiida-wannierjl's options handling inside a
            graph body (dict graph inputs arrive as TaggedValue proxies,
            which node-graph refuses to assign into ``metadata.options``).
        subblock_wannier90_options: split mode only — optional
            ``metadata.options`` for the per-group re-wannierisation
            ``Wannier90Calculation`` (defaults to single-machine resources;
            that CalcJob has no resources default of its own).

    Returns:
        A :class:`WannierizeBlocksOutputs`: the ``blocks`` namespace keyed by
        block label, the unified ``centres`` / ``spreads`` (plain mode), the
        ``bands`` / ``groups`` detection outputs (split mode), the
        ``orbitals`` initial partition (only when every block is stamped
        with ``filled``), and (only when the scf + nscf ran here) the
        shared ``nscf`` outputs.
    """
    overrides = overrides or {}
    validate_parallelization(parallelization)
    _validate_block_projection_types(blocks)

    split = _resolve_split_mode(
        codes=codes,
        blocks=blocks,
        mp_grid=mp_grid,
        nscf_remote_folder=nscf_remote_folder,
        split_threshold=split_threshold,
        bands_kpoints=bands_kpoints,
        num_occ_bands=num_occ_bands,
        wjl_options=wjl_options,
        subblock_wannier90_options=subblock_wannier90_options,
        cubic_pw2wannier90_options=cubic_pw2wannier90_options,
    )

    # --- shared scf + nscf (run once, or reuse the caller's scratch) ---
    if nscf_remote_folder is not None:
        if "scf" in overrides or "nscf" in overrides:
            raise ValueError(
                "scf/nscf overrides were given together with an external "
                "nscf_remote_folder; the internal scf + nscf is skipped, so "
                "they would be silently ignored."
            )
        scf_nscf = None
        nscf_scratch = nscf_remote_folder
    else:
        scf_nscf_overrides: dict[str, Any] = {}
        if "scf" in overrides:
            scf_nscf_overrides["scf"] = overrides["scf"]
        if "nscf" in overrides:
            scf_nscf_overrides["nscf"] = overrides["nscf"]

        scf_nscf = RunScfNscf(
            code=codes["pw"],
            structure=structure,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides=scf_nscf_overrides or None,
            # The blocks' wannier90 / pw2wannier90 read eigenstates on the
            # explicit ``kpoints`` mesh, so the nscf must run on exactly that
            # grid (not the protocol's kpoints_distance-derived one).
            nscf_kpoints=kpoints,
            parallelization=parallelization,
            metadata={"call_link_label": "scf_nscf"},
        )
        nscf_scratch = scf_nscf["nscf_remote_folder"]

        # --- split mode: bands step + runtime group detection ---
        # Nested under the internal scf + nscf on purpose: split mode
        # rejects an external scratch, so the bands step always has this
        # scf's remote folder. After validation ``bands_kpoints is not
        # None`` is equivalent to ``split`` (the test also narrows the
        # Optionals). The split machinery depends on aiida-wannierjl, so it
        # is imported only on this branch; the import direction
        # (auto_wannierize imports this module at module level, this body
        # imports auto_wannierize lazily) avoids the cycle.
        if bands_kpoints is not None:
            from aiida_koopmans.workgraphs.auto_wannierize import (
                add_bands_step,
                detect_band_groups,
            )

            bands_outputs = add_bands_step(
                code=codes["pw"],
                structure=structure,
                bands_kpoints=bands_kpoints,
                scf_remote_folder=scf_nscf["scf_remote_folder"],
                nscf_overrides=overrides.get("nscf"),
                pseudo_family=pseudo_family,
                protocol=protocol,
                electronic_type=electronic_type,
                parallelization=parallelization,
            )
            # The detection is restricted to the Wannierised manifold — the
            # disentanglement pool above it must not influence the grouping.
            detect = detect_band_groups(
                bands=bands_outputs["output_band"],
                num_occ_bands=num_occ_bands,
                threshold=split_threshold,
                num_bands_total=sum(int(block["num_wann"]) for block in blocks),
            )

    # --- per-block Wannierisation: native for-loop fan-out ---
    # Each iteration adds an independent per-block graph (they share only
    # the read-only nscf scratch, so they run in parallel), collected into a
    # dict keyed by block label -> the ``blocks`` dynamic output namespace.
    # The parsed per-block outputs feed the unify task positionally, read
    # straight off each graph call (not off the ``blocks`` entries, whose
    # label keys would impose a sort order): the ``blocks`` input-list
    # order is the band-order authority.
    block_outputs: dict[str, Any] = {}
    collect_inputs: dict[str, Any] = {}
    for i, block in enumerate(blocks):
        if split:
            from aiida_koopmans.workgraphs.auto_wannierize import WannierizeAndSplitBlock

            wannierized = WannierizeAndSplitBlock(
                codes=codes,
                structure=structure,
                block=block,
                groups=detect.result,
                nscf_remote_folder=nscf_scratch,
                kpoints=kpoints,
                mp_grid=mp_grid,
                pseudo_family=pseudo_family,
                protocol=protocol,
                overrides=overrides or None,
                electronic_type=electronic_type,
                spin_type=spin_type,
                parallelization=parallelization,
                wjl_options=wjl_options,
                wannier90_options=subblock_wannier90_options,
                pw2wannier90_options=cubic_pw2wannier90_options,
                metadata={"call_link_label": f"wannierize_split_{block['label']}"},
            )
        else:
            wannierized = WannierizeBlock(
                codes=codes,
                structure=structure,
                block=block,
                projection_type=block["projection_type"],
                nscf_remote_folder=nscf_scratch,
                kpoints=kpoints,
                mp_grid=mp_grid,
                pseudo_family=pseudo_family,
                protocol=protocol,
                overrides=overrides or None,
                electronic_type=electronic_type,
                spin_type=spin_type,
                parallelization=parallelization,
                metadata={"call_link_label": f"wannierize_{block['label']}"},
            )
        # Both per-block graphs return the flat WannierizeBlockOutputs
        # shape, forwarded whole into the entry (split-mode entries leave
        # the plain-route-only folder sockets unpopulated).
        block_outputs[block["label"]] = wannierized
        collect_inputs[f"b{i:02d}"] = wannierized["output_parameters"]

    collected = collect_wannier_functions(
        output_parameters=collect_inputs,
        metadata={"call_link_label": "collect_wannier_functions"},
    )
    outputs = WannierizeBlocksOutputs(
        blocks=block_outputs,
        centres=collected["centres"],
        spreads=collected["spreads"],
    )
    _maybe_emit_orbital_partition(outputs, blocks)
    if split:
        outputs["bands"] = bands_outputs["output_band"]
        outputs["groups"] = detect.result
    if scf_nscf is not None:
        outputs["nscf"] = PwOutputs(
            remote_folder=nscf_scratch,
            output_parameters=scf_nscf["nscf_output_parameters"],
            output_band=scf_nscf["nscf_output_band"],
        )
    return outputs
