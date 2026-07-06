"""Fold per-block Wannier orbitals into supercell kcp.x wavefunctions.

Step "B2" of the periodic MLWF / projwfs Koopmans port — the AiiDA port of
legacy ``FoldToSupercellWorkflow`` (``koopmans/workflows/_folding.py``).
Consumes the per-block outputs of
:func:`~aiida_koopmans.workgraphs.block_wannierize.BlockWannierizeTask` and
produces the ``evc_occupied{n}.dat`` / ``evc0_empty{n}.dat`` files that seed
the supercell ``dft_init`` kcp.x run:

1. one ``wann2kcp.x`` per projection block (``wan_mode='wannier2kcp'``),
   reading the shared nscf scratch plus that block's ``.nnkp`` / ``.chk`` /
   ``_hr.dat``, and writing the folded ``evcw`` wavefunctions
   (``evcw1.dat`` + ``evcw2.dat``, or a single ``evcw.dat`` when the run is
   spin-resolved via ``spin_component``);
2. one ``merge_evc.x`` per (manifold, kcp spin index) that concatenates the
   same-named ``evcw`` file across every block of the merge group and names
   the result ``evc_occupied{n}.dat`` / ``evc0_empty{n}.dat``.

Deviation from legacy: single-block groups also run through ``merge_evc.x``
(legacy passes the lone ``evcw`` file along and renames it via the symlink
machinery). Concatenating one file is a plain copy, and it normalises the
output contract — every folded manifold file sits at the root of a
``merge_evc.x`` ``remote_folder`` under its final kcp.x name, which is what
``KcpCalculation``'s ``read_wavefunctions`` staging expects.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict, cast

from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans.calculations.merge_evc import MergeEvcCalculation
from aiida_koopmans.calculations.wann2kcp import Wann2kcpCalculation
from aiida_koopmans.types import MergeGroup, ProjectionBlock, SpinChannel, merge_dest_filename
from aiida_koopmans.workgraphs import Codes
from aiida_koopmans.workgraphs.block_wannierize import BlockWannierOutputs

Wann2kcpTask = task(Wann2kcpCalculation)
MergeEvcTask = task(MergeEvcCalculation)

# wann2kcp.x's seedname must match the names of the staged Wannier files;
# aiida-wannier90 hard-codes ``aiida``, and ``Wann2kcpCalculation`` copies the
# ``.nnkp`` / ``.chk`` / ``_hr.dat`` inputs to ``<seedname>.*``.
_W2K_SEEDNAME = "aiida"


class FoldToSupercellOutputs(TypedDict, total=False):
    """Folded manifold wavefunctions, one ``RemoteData`` per kcp.x file.

    Keys are the destination stems (``.dat`` filenames without the
    extension, matching ``KcpCalculation.read_wavefunctions``); each value
    is the ``merge_evc.x`` ``remote_folder`` whose root holds
    ``<stem>.dat``. ``total=False`` because the empty-manifold entries only
    exist when the projections include empty blocks (per spin channel, for
    spin-polarized runs). The stems are a fixed four-name vocabulary —
    kcp.x always runs nspin=2 in the DSCF flow — so this is a plain
    TypedDict rather than a dynamic namespace.
    """

    evc_occupied1: orm.RemoteData
    evc_occupied2: orm.RemoteData
    evc0_empty1: orm.RemoteData
    evc0_empty2: orm.RemoteData


class FoldTarget(TypedDict):
    """One ``merge_evc.x`` invocation derived from a merge group.

    * ``group_index`` — index into the ``merge_groups`` list.
    * ``stem`` — destination stem (``evc_occupied1`` etc.).
    * ``source_filename`` — the per-block ``evcw`` file to concatenate
      (``evcw{1,2}.dat`` for a spinless wann2kcp run, ``evcw.dat`` for a
      spin-resolved one).
    """

    group_index: int
    stem: str
    source_filename: str


def enumerate_fold_targets(
    merge_groups: list[MergeGroup], spin_polarized: bool
) -> list[FoldTarget]:
    """List the merge_evc.x runs (and their file names) for a set of merge groups.

    Ports the file-naming walk of legacy ``FoldToSupercellWorkflow._run``
    (``_folding.py:80-109``): a spin-polarized group folds its single
    ``evcw.dat`` into the kcp spin slot matching the group's spin channel,
    while a spinless group emits both kcp spin slots from ``evcw1.dat`` /
    ``evcw2.dat``. Shared between :func:`FoldToSupercell` (which creates the
    tasks) and the MLWF initialisation (which needs to know which stems will
    exist) so the two can never disagree.
    """
    targets: list[FoldTarget] = []
    for group_index, group in enumerate(merge_groups):
        if spin_polarized:
            spin_index = 2 if SpinChannel(group["spin"]) == SpinChannel.DOWN else 1
            pairs = [(spin_index, "evcw.dat")]
        else:
            pairs = [(1, "evcw1.dat"), (2, "evcw2.dat")]
        for spin_index, source_filename in pairs:
            stem = merge_dest_filename(group["filled"], spin_index).removesuffix(".dat")
            targets.append(
                FoldTarget(group_index=group_index, stem=stem, source_filename=source_filename)
            )
    return targets


@task.graph
def FoldToSupercell(
    codes: Codes,
    blocks: list[ProjectionBlock],
    merge_groups: list[MergeGroup],
    nscf_remote_folder: orm.RemoteData,
    block_wannier: Annotated[dict, dynamic(BlockWannierOutputs)],
    kgrid: list[int],
    gamma_only: bool = False,
    spin_polarized: bool = False,
    options: dict[str, Any] | None = None,
) -> FoldToSupercellOutputs:
    """Convert per-block Wannier orbitals into merged supercell kcp.x files.

    Args:
        codes: code instances; required keys ``wann2kcp`` and ``merge_evc``.
        blocks: the projection blocks, in the same order they were
            Wannierised.
        merge_groups: the per-(manifold, spin) grouping of ``blocks`` from
            :func:`~aiida_koopmans.types.group_blocks_to_merge`.
        nscf_remote_folder: scratch of the shared nscf every block was
            built on (``wann2kcp.x`` re-reads the Bloch states from it).
        block_wannier: per-block Wannierisation outputs keyed by block
            label — the ``blocks`` namespace of
            :func:`~aiida_koopmans.workgraphs.block_wannierize.BlockWannierizeTask`.
        kgrid: the primitive Monkhorst-Pack grid; its product is the
            ``-nr`` real-space fold count of ``merge_evc.x``.
        gamma_only: whether the primitive sampling is Γ-only. Forwarded as
            wann2kcp.x's ``gamma_trick`` — legacy enforces the two to be
            equal (``_folding.py:49-55``).
        spin_polarized: spin-resolved Wannierisation. Each block then runs
            wann2kcp.x with its own ``spin_component`` and writes a single
            ``evcw.dat``; spinless blocks write ``evcw1.dat`` + ``evcw2.dat``.
        options: ``metadata.options`` for the underlying CalcJobs.

    The per-block fan-out is a native ``for`` loop in this deferred body
    (the documented dynamic scatter-gather; see ``block_wannierize.py``).
    """
    # --- per-block wann2kcp.x fan-out ---
    w2k_remotes: dict[str, Any] = {}
    for block in blocks:
        label = block["label"]
        parameters: dict[str, Any] = {
            "wan_mode": "wannier2kcp",
            "seedname": _W2K_SEEDNAME,
            "gamma_trick": bool(gamma_only),
        }
        spin = SpinChannel(block["spin"])
        if spin != SpinChannel.NONE:
            parameters["spin_component"] = spin.value

        w2k_inputs: dict[str, Any] = {
            "code": codes["wann2kcp"],
            "parameters": parameters,
            "parent_folder": nscf_remote_folder,
            "nnkp_file": block_wannier[label]["nnkp_file"],
            "wannier_folder": block_wannier[label]["hr_retrieved"],
            "metadata": {"call_link_label": f"fold_{label}"},
        }
        if options:
            w2k_inputs["metadata"]["options"] = options
        w2k_remotes[label] = Wann2kcpTask(**w2k_inputs)["remote_folder"]

    # --- per-(manifold, spin slot) merge_evc.x ---
    merged: dict[str, Any] = {}
    for target in enumerate_fold_targets(merge_groups, spin_polarized):
        group = merge_groups[target["group_index"]]
        # Keys are zero-padded so the sorted-key order ``merge_evc.x``
        # symlinks ``input_{i}.dat`` in matches the band order of the
        # group's blocks.
        source_files = {f"b{i:02d}": w2k_remotes[b["label"]] for i, b in enumerate(group["blocks"])}
        merge_inputs: dict[str, Any] = {
            "code": codes["merge_evc"],
            "kgrid": list(kgrid),
            "dest_filename": f"{target['stem']}.dat",
            "source_files": source_files,
            "settings": {
                "source_filenames": dict.fromkeys(source_files, target["source_filename"])
            },
            "metadata": {"call_link_label": f"merge_{target['stem']}"},
        }
        if options:
            merge_inputs["metadata"]["options"] = options
        merged[target["stem"]] = MergeEvcTask(**merge_inputs)["remote_folder"]

    return cast("FoldToSupercellOutputs", merged)
