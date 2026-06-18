"""Block-by-block Wannierisation of a periodic system.

Step "B1" of the periodic MLWF / projwfs Koopmans port. A single shared
scf + nscf is run once (via :func:`PwScfNscfTask`), then each projection
block (occupied / empty manifold, per spin) is Wannierised in its own
``Wannier90WorkChain`` that *skips* scf and nscf and reads the shared nscf
scratch directly.

The fan-out over blocks uses an ``aiida-workgraph`` ``Map`` zone keyed by
each block's stable ``label`` (e.g. ``"block_1"`` / ``"block_1_spin_up"``),
mirroring the per-orbital Map pattern in
``aiida_koopmans/workgraphs/kcp.py``.

Per-block file staging that the later fold-to-supercell step (B2) consumes:

* ``hr_retrieved`` -- the wannier90 ``retrieved`` :class:`~aiida.orm.FolderData`,
  which holds ``aiida_hr.dat`` (the real-space Hamiltonian, written because
  ``write_hr=True``).
* ``remote_folder`` -- the wannier90 ``RemoteData`` scratch, which holds
  ``aiida.chk`` (forced into the retrieve list, since wannier90 does not
  retrieve the checkpoint by default).
* ``nnkp_file`` -- the ``aiida.nnkp`` :class:`~aiida.orm.SinglefileData`
  emitted by the wannier90 post-processing (``-pp``) run.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.workflows import Wannier90WorkChain
from aiida_workgraph import Map, dynamic, task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.types import ProjectionBlock, block_w90_kwargs
from aiida_koopmans.workgraphs import Codes
from aiida_koopmans.workgraphs.pw import PwOutputs, PwScfNscfTask

# Force retrieval of the wannier90 checkpoint: ``aiida.chk`` is not in the
# default retrieve list, but B2 (fold-to-supercell) needs it to unitarily
# rotate the per-block manifolds. ``aiida_hr.dat`` lands in ``retrieved``
# automatically once ``write_hr=True``.
_W90_RETRIEVE_SETTINGS: dict[str, list[str]] = {"additional_retrieve_list": ["aiida.chk"]}


class BlockWannierOutputs(TypedDict):
    """Per-block Wannierisation outputs that the supercell fold (B2) reads.

    * ``hr_retrieved`` -- wannier90 ``retrieved`` FolderData (holds
      ``aiida_hr.dat``).
    * ``remote_folder`` -- wannier90 ``RemoteData`` scratch (holds
      ``aiida.chk``).
    * ``nnkp_file`` -- ``aiida.nnkp`` SinglefileData from the ``-pp`` run.
    """

    hr_retrieved: orm.FolderData
    remote_folder: orm.RemoteData
    nnkp_file: orm.SinglefileData


class BlockWannierizeOutputs(TypedDict):
    """Outputs of :func:`BlockWannierizeTask`.

    * ``nscf`` -- the shared nscf :class:`PwOutputs` so B2 can read
      ``nscf["remote_folder"]`` (the nscf scratch every block was built on).
    * ``blocks`` -- a dynamic namespace keyed by block label; each entry is
      a :class:`BlockWannierOutputs`.
    """

    nscf: PwOutputs
    blocks: Annotated[dict, dynamic(BlockWannierOutputs)]


Wannier90Task = task(Wannier90WorkChain)


@task
def blocks_to_map_source(
    blocks: list[ProjectionBlock],
) -> Annotated[dict, dynamic(dict)]:
    """Materialise the per-block iterator dict for the ``Map`` zone.

    ``aiida-workgraph``'s ``Map`` zone iterates over a dict and uses the
    key as the iteration handle / sub-task link label, so keys must be
    strings -- here each block's stable ``label``. Each value is the block
    dict itself (a plain-primitive :class:`ProjectionBlock`, so it survives
    AiiDA storage). Building the dict inside a real ``@task`` keeps the
    list-to-dict transform off the raw socket inside the graph body.
    """
    return {block["label"]: block for block in blocks}


@task(outputs=["block", "projection_type"])
def unpack_block_item(item: dict) -> dict:
    """Explode one ``Map`` item into named output sockets.

    Returns the block dict and its ``projection_type`` (a real
    :class:`WannierProjectionType`) separately so the per-block graph can
    consume them without subscripting a socket inside the Map zone.
    """
    return {"block": item, "projection_type": item["projection_type"]}


@task.graph
def BlockWannierize(
    codes: Codes,
    structure: orm.StructureData,
    block: ProjectionBlock,
    projection_type: WannierProjectionType,
    nscf_remote_folder: orm.RemoteData,
    kpoints: orm.KpointsData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
) -> BlockWannierOutputs:
    """Wannierise a single projection block off the shared nscf scratch.

    Seeds a ``Wannier90WorkChain`` builder via ``get_builder_from_protocol``
    for this block's ``projection_type``, then:

    * pops the ``scf`` namespace and the ``nscf`` namespace so the workchain
      skips both steps (``should_run_scf`` / ``should_run_nscf`` are simply
      ``"scf" in inputs`` / ``"nscf" in inputs`` upstream), and points the
      pw2wannier90 step at the shared nscf scratch via
      ``pw2wannier90.pw2wannier90.parent_folder`` -- the only parent the
      validator accepts once both scf and nscf are absent
      (``wannier90.py`` ``validate_inputs`` lines 94-99; consumed in
      ``setup`` lines 763-767);
    * overrides the per-block ``num_wann`` / ``num_bands`` / ``exclude_bands``
      (and ``projections`` for explicit blocks) from
      :func:`block_w90_kwargs`;
    * forces ``write_hr=True`` and ``aiida.chk`` retrieval.
    """
    overrides = overrides or {}

    builder = Wannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
        # ``pseudo_family`` arrives as an ``aiida-workgraph`` TaggedValue
        # proxy; unwrap to a plain ``str`` so the upstream pseudo loader
        # (which does identity / membership checks) sees a real string.
        pseudo_family=str(pseudo_family) if pseudo_family is not None else None,
        electronic_type=electronic_type,
        spin_type=spin_type,
        projection_type=projection_type,
        # The hamiltonian-retrieval protocol override sets ``write_hr`` /
        # ``write_tb`` and the hr retrieve handling.
        retrieve_hamiltonian=True,
        print_summary=False,
    )

    # --- per-block wannier90 parameters / projections ---
    w90_kwargs = block_w90_kwargs(block)
    w90_params = builder.wannier90.wannier90.parameters.get_dict()
    w90_params["num_wann"] = w90_kwargs["num_wann"]
    w90_params["num_bands"] = w90_kwargs["num_bands"]
    if "exclude_bands" in w90_kwargs:
        w90_params["exclude_bands"] = w90_kwargs["exclude_bands"]
    # ``write_hr`` is set by the retrieve_hamiltonian override above; pin it
    # explicitly so a stripped-down override dict can't silently drop it.
    w90_params["write_hr"] = True
    builder.wannier90.wannier90.parameters = orm.Dict(w90_params)

    # Explicit (ANALYTIC) blocks carry resolved projection orbitals; automatic
    # blocks rely on ``projection_type`` alone (no ``projections`` key).
    if "projections" in w90_kwargs:
        builder.wannier90.wannier90.projections = orm.List(list=w90_kwargs["projections"])

    # Share the nscf k-mesh so the per-block wannier90 / pw2wannier90 read
    # eigenstates on the exact grid the shared nscf produced.
    builder.wannier90.wannier90.kpoints = kpoints

    # Force ``aiida.chk`` into the wannier90 retrieve list (merged on top of
    # whatever ``settings`` the protocol set; the workchain only adds its own
    # ``postproc_setup`` key on top of this).
    existing_settings: dict = {}
    if "settings" in builder.wannier90.wannier90:
        existing_settings = builder.wannier90.wannier90.settings.get_dict()
    existing_settings.update(_W90_RETRIEVE_SETTINGS)
    builder.wannier90.wannier90.settings = orm.Dict(existing_settings)

    data = get_dict_from_builder(builder)

    # Skip scf + nscf and reuse the shared nscf scratch. With both namespaces
    # absent the workchain validator requires the parent on the pw2wannier90
    # step (``validate_inputs`` lines 94-99).
    data.pop("scf", None)
    data.pop("nscf", None)
    data.pop("clean_workdir", None)
    data["pw2wannier90"]["pw2wannier90"]["parent_folder"] = nscf_remote_folder

    data.setdefault("metadata", {})["call_link_label"] = "wannier90"
    outputs = Wannier90Task(**data)

    return BlockWannierOutputs(
        hr_retrieved=outputs["wannier90"]["retrieved"],
        remote_folder=outputs["wannier90"]["remote_folder"],
        nnkp_file=outputs["wannier90_pp"]["nnkp_file"],
    )


@task.graph
def BlockWannierizeTask(
    codes: Codes,
    structure: orm.StructureData,
    blocks: list[ProjectionBlock],
    kpoints: orm.KpointsData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
) -> BlockWannierizeOutputs:
    """Wannierise a periodic system block-by-block off one shared scf + nscf.

    A single :func:`PwScfNscfTask` runs scf + nscf once; every projection
    block is then Wannierised in its own ``Wannier90WorkChain`` that skips
    scf / nscf and reads the shared nscf scratch
    (``nscf["remote_folder"]``). The per-block fan-out is an
    ``aiida-workgraph`` ``Map`` zone keyed by each block's ``label``.

    Args:
        codes: code instances. Required keys: ``pw``, ``wannier90``,
            ``pw2wannier90``; ``projwfc`` is needed only for projection types
            that run projwfc (e.g. SCDM / energy-auto frozen window).
        structure: the periodic ``StructureData``.
        blocks: the resolved projection blocks (B0 output); occupied and
            empty manifolds appear as separate blocks. Each is Wannierised
            independently.
        kpoints: the explicit k-mesh shared by the nscf and every block's
            wannier90 / pw2wannier90.
        pseudo_family: pseudopotential family label.
        protocol: protocol name passed to both builders.
        overrides: optional overrides. ``overrides["scf"]`` / ``["nscf"]``
            feed :func:`PwScfNscfTask`; ``overrides["wannier90"]`` feeds every
            per-block wannier builder.
        electronic_type / spin_type: forwarded to the wannier builder.

    Returns:
        A :class:`BlockWannierizeOutputs`: the shared ``nscf`` outputs plus a
        ``blocks`` namespace keyed by block label.
    """
    overrides = overrides or {}

    # --- shared scf + nscf (run once) ---
    scf_nscf_overrides: dict[str, Any] = {}
    if "scf" in overrides:
        scf_nscf_overrides["scf"] = overrides["scf"]
    if "nscf" in overrides:
        scf_nscf_overrides["nscf"] = overrides["nscf"]

    scf_nscf = PwScfNscfTask(
        code=codes["pw"],
        structure=structure,
        pseudo_family=pseudo_family,
        protocol=protocol,
        overrides=scf_nscf_overrides or None,
        metadata={"call_link_label": "scf_nscf"},
    )
    nscf_remote_folder = scf_nscf["nscf_remote_folder"]

    wannier_overrides = overrides.get("wannier90")

    # --- per-block Wannierisation via a Map zone keyed by block label ---
    block_source = blocks_to_map_source(blocks=blocks)
    with Map(block_source) as block_zone:
        item = block_zone.item.value
        unpacked = unpack_block_item(item=item)
        block_out = BlockWannierize(
            codes=codes,
            structure=structure,
            block=unpacked["block"],
            projection_type=unpacked["projection_type"],
            nscf_remote_folder=nscf_remote_folder,
            kpoints=kpoints,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides=wannier_overrides,
            electronic_type=electronic_type,
            spin_type=spin_type,
        )
        block_zone.gather(
            {
                "hr_retrieved": block_out["hr_retrieved"],
                "remote_folder": block_out["remote_folder"],
                "nnkp_file": block_out["nnkp_file"],
            }
        )

    return BlockWannierizeOutputs(
        nscf=PwOutputs(remote_folder=nscf_remote_folder),
        blocks=block_zone.outputs,
    )
