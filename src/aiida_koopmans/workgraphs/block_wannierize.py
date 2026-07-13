"""Block-by-block Wannierisation of a periodic system.

A single shared scf + nscf is run once (via :func:`RunScfNscf`), then each
projection block (occupied / empty manifold, per spin) is Wannierised in its
own ``Wannier90WorkChain`` that *skips* scf and nscf and reads the shared
nscf scratch directly. The per-block fan-out is a native ``for`` loop over
``blocks`` inside the ``@task.graph`` body -- do not convert it to a ``Map``
zone. Results are collected into a dict keyed by each block's stable
``label`` (e.g. ``"block_1"`` / ``"block_1_spin_up"``) and returned as a
dynamic output namespace.

Per-block file staging that the supercell fold consumes:

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
from aiida_workgraph import dynamic, task
from aiida_workgraph.utils import get_dict_from_builder

from aiida_koopmans.types import ProjectionBlock, block_w90_kwargs
from aiida_koopmans.workgraphs import Codes
from aiida_koopmans.workgraphs.pw import PwOutputs, RunScfNscf
from aiida_koopmans.workgraphs.wannier90 import Wannier90Step

# Force retrieval of the wannier90 checkpoint: ``aiida.chk`` is not in the
# default retrieve list, but the supercell fold needs it to unitarily
# rotate the per-block manifolds. ``aiida_hr.dat`` lands in ``retrieved``
# automatically once ``write_hr=True``.
_W90_RETRIEVE_SETTINGS: dict[str, list[str]] = {"additional_retrieve_list": ["aiida.chk"]}


class WannierizeBlockOutputs(TypedDict):
    """Per-block Wannierisation outputs that the supercell fold reads.

    * ``hr_retrieved`` -- wannier90 ``retrieved`` FolderData (holds
      ``aiida_hr.dat``).
    * ``remote_folder`` -- wannier90 ``RemoteData`` scratch (holds
      ``aiida.chk``).
    * ``nnkp_file`` -- ``aiida.nnkp`` SinglefileData from the ``-pp`` run.
    """

    hr_retrieved: orm.FolderData
    remote_folder: orm.RemoteData
    nnkp_file: orm.SinglefileData


class WannierizeBlocksOutputs(TypedDict):
    """Outputs of :func:`WannierizeBlocks`.

    * ``nscf`` -- the shared nscf :class:`PwOutputs` so the supercell fold
      can read ``nscf["remote_folder"]`` (the nscf scratch every block was
      built on).
    * ``blocks`` -- a dynamic namespace keyed by block label; each entry is
      a :class:`WannierizeBlockOutputs`, consumable downstream as a namespace.
    """

    nscf: PwOutputs
    blocks: Annotated[dict, dynamic(WannierizeBlockOutputs)]


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
    overrides: dict[str, Any] | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
) -> WannierizeBlockOutputs:
    """Wannierise a single projection block off the shared nscf scratch.

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
    * forces ``write_hr=True`` and ``aiida.chk`` retrieval.
    """
    overrides = overrides or {}

    builder = Wannier90WorkChain.get_builder_from_protocol(
        codes=codes,
        structure=structure,
        protocol=protocol,
        overrides=overrides,
        pseudo_family=pseudo_family,
        electronic_type=electronic_type,
        spin_type=spin_type,
        projection_type=projection_type,
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
    # ``write_hr`` is set by the retrieve_hamiltonian override above; pin it
    # explicitly so a stripped-down override dict can't silently drop it.
    w90_params["write_hr"] = True
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

    # Force ``aiida.chk`` into the wannier90 retrieve list (merged on top of
    # whatever ``settings`` the protocol set; the workchain only adds its own
    # ``postproc_setup`` key on top of this).
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

    data.setdefault("metadata", {})["call_link_label"] = "wannier90"
    outputs = Wannier90Step(**data)

    return WannierizeBlockOutputs(
        hr_retrieved=outputs["wannier90"]["retrieved"],
        remote_folder=outputs["wannier90"]["remote_folder"],
        nnkp_file=outputs["wannier90_pp"]["nnkp_file"],
    )


@task.graph
def WannierizeBlocks(
    codes: Codes,
    structure: orm.StructureData,
    blocks: list[ProjectionBlock],
    kpoints: orm.KpointsData,
    pseudo_family: str | None = None,
    protocol: str | None = None,
    overrides: dict[str, Any] | None = None,
    electronic_type: ElectronicType = ElectronicType.INSULATOR,
    spin_type: SpinType = SpinType.NONE,
) -> WannierizeBlocksOutputs:
    """Wannierise a periodic system block-by-block off one shared scf + nscf.

    A single :func:`RunScfNscf` runs scf + nscf once; every projection
    block is then Wannierised in its own ``Wannier90WorkChain`` that skips
    scf / nscf and reads the shared nscf scratch (``nscf["remote_folder"]``).
    The per-block fan-out is a native ``for`` loop over ``blocks`` inside this
    ``@task.graph`` body; the per-block outputs are collected into a dict
    keyed by block label and returned as the ``blocks`` dynamic namespace.

    Args:
        codes: code instances. Required keys: ``pw``, ``wannier90``,
            ``pw2wannier90``; ``projwfc`` is needed only for projection types
            that run projwfc (e.g. SCDM / energy-auto frozen window).
        structure: the periodic ``StructureData``.
        blocks: the resolved projection blocks; occupied and empty manifolds
            appear as separate blocks. Each is Wannierised independently.
        kpoints: the explicit k-mesh shared by the nscf and every block's
            wannier90 / pw2wannier90.
        pseudo_family: pseudopotential family label.
        protocol: protocol name passed to both builders.
        overrides: optional overrides. ``overrides["scf"]`` / ``["nscf"]``
            feed :func:`RunScfNscf`; ``overrides["wannier90"]`` feeds every
            per-block wannier builder.
        electronic_type / spin_type: forwarded to the wannier builder.

    Returns:
        A :class:`WannierizeBlocksOutputs`: the shared ``nscf`` outputs plus a
        ``blocks`` namespace keyed by block label.
    """
    overrides = overrides or {}

    # --- shared scf + nscf (run once) ---
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
        metadata={"call_link_label": "scf_nscf"},
    )
    nscf_remote_folder = scf_nscf["nscf_remote_folder"]

    wannier_overrides = overrides.get("wannier90")

    # --- per-block Wannierisation: native for-loop fan-out ---
    # Each iteration adds an independent ``WannierizeBlock`` (they share only
    # the read-only nscf scratch, so they run in parallel), collected into a
    # dict keyed by block label -> the ``blocks`` dynamic output namespace.
    block_outputs: dict[str, WannierizeBlockOutputs] = {}
    for block in blocks:
        block_outputs[block["label"]] = WannierizeBlock(
            codes=codes,
            structure=structure,
            block=block,
            projection_type=block["projection_type"],
            nscf_remote_folder=nscf_remote_folder,
            kpoints=kpoints,
            pseudo_family=pseudo_family,
            protocol=protocol,
            overrides=wannier_overrides,
            electronic_type=electronic_type,
            spin_type=spin_type,
        )

    return WannierizeBlocksOutputs(
        nscf=PwOutputs(remote_folder=nscf_remote_folder),
        blocks=block_outputs,
    )
