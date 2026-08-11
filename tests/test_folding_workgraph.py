"""Construction-level tests for the fold-to-supercell workgraph.

Mirrors ``test_block_wannierize.py``: the graphs are built (no daemon, no
real codes) and their task lists inspected, so a wiring regression surfaces
without a QE install. The merge-target bookkeeping
(:func:`enumerate_fold_targets`) is additionally unit-tested against the
expected wavefunction file-naming rules.
"""

from __future__ import annotations

import io

import pytest

from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.workgraphs.folding import FoldToSupercell, enumerate_fold_targets
from aiida_koopmans.workgraphs.utils.wannier_merge import group_blocks_to_merge
from tests.fixtures import explicit_block

# ----------------------------------------------------------------------
# Block / merge-group shapes
# ----------------------------------------------------------------------


def _spinless_blocks():
    """Silicon-like shape: two occupied blocks + one empty block, nspin=1."""
    return [
        explicit_block("block_1", range(1, 3), filled=True),
        explicit_block("block_2", range(3, 5), filled=True),
        explicit_block("block_3", range(5, 9), filled=False),
    ]


def _spin_polarized_blocks():
    """One occupied + one empty block per spin channel."""
    return [
        explicit_block("block_1_up", range(1, 8), spin=SpinChannel.UP, filled=True),
        explicit_block("block_2_up", range(8, 9), spin=SpinChannel.UP, filled=False),
        explicit_block("block_1_down", range(1, 6), spin=SpinChannel.DOWN, filled=True),
        explicit_block("block_2_down", range(6, 9), spin=SpinChannel.DOWN, filled=False),
    ]


# ----------------------------------------------------------------------
# enumerate_fold_targets — destination-filename semantics
# ----------------------------------------------------------------------


class TestEnumerateFoldTargets:
    def test_spinless_groups_emit_both_kcp_spin_slots(self):
        groups = group_blocks_to_merge(_spinless_blocks(), {SpinChannel.NONE: 4})
        targets = enumerate_fold_targets(groups, spin_polarized=False)
        assert [(t["stem"], t["source_port"]) for t in targets] == [
            ("evc_occupied1", "evcw1"),
            ("evc_occupied2", "evcw2"),
            ("evc0_empty1", "evcw1"),
            ("evc0_empty2", "evcw2"),
        ]

    def test_spin_polarized_groups_emit_one_slot_each(self):
        groups = group_blocks_to_merge(
            _spin_polarized_blocks(), {SpinChannel.UP: 7, SpinChannel.DOWN: 5}
        )
        targets = enumerate_fold_targets(groups, spin_polarized=True)
        assert [(t["stem"], t["source_port"]) for t in targets] == [
            ("evc_occupied1", "evcw"),
            ("evc0_empty1", "evcw"),
            ("evc_occupied2", "evcw"),
            ("evc0_empty2", "evcw"),
        ]

    def test_group_indices_point_into_the_group_list(self):
        groups = group_blocks_to_merge(_spinless_blocks(), {SpinChannel.NONE: 4})
        targets = enumerate_fold_targets(groups, spin_polarized=False)
        assert [t["group_index"] for t in targets] == [0, 0, 1, 1]


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


@pytest.fixture
def fold_codes(aiida_localhost):
    """Return stand-in InstalledCode nodes for wann2kcp.x and merge_evc.x."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    def _code(label: str, entry_point: str):
        try:
            return InstalledCode.collection.get(label=label)
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=aiida_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point,
            ).store()

    return {
        "wann2kcp": _code("fold-w2k", "koopmans.wann2kcp"),
        "merge_evc": _code("fold-merge", "koopmans.merge_evc"),
    }


def _fake_block_wannier(blocks, aiida_localhost):
    """Build the per-block output namespace with stand-in file/folder nodes."""
    from aiida.orm import FolderData, RemoteData, SinglefileData

    namespace = {}
    for block in blocks:
        folder = FolderData()
        folder.put_object_from_bytes(b"chk", "aiida.chk")
        folder.put_object_from_bytes(b"hr", "aiida_hr.dat")
        namespace[block["label"]] = {
            "retrieved": folder.store(),
            "remote_folder": RemoteData(
                computer=aiida_localhost, remote_path=f"/fake/{block['label']}"
            ).store(),
            "nnkp_file": SinglefileData(io.BytesIO(b"nnkp"), filename="aiida.nnkp").store(),
        }
    return namespace


class TestFoldToSupercellGraphBuild:
    @pytest.mark.parametrize(
        "blocks_factory,spin_polarized,num_occ,n_merges",
        [
            (_spinless_blocks, False, {SpinChannel.NONE: 4}, 4),
            (
                _spin_polarized_blocks,
                True,
                {SpinChannel.UP: 7, SpinChannel.DOWN: 5},
                4,
            ),
        ],
    )
    def test_one_wann2kcp_per_block_one_merge_per_target(
        self, fold_codes, aiida_localhost, blocks_factory, spin_polarized, num_occ, n_merges
    ):
        from aiida.orm import RemoteData

        blocks = blocks_factory()
        groups = group_blocks_to_merge(blocks, num_occ)
        wg = FoldToSupercell.build(
            codes=fold_codes,
            blocks=blocks,
            merge_groups=groups,
            nscf_remote_folder=RemoteData(
                computer=aiida_localhost, remote_path="/fake/nscf"
            ).store(),
            block_wannier=_fake_block_wannier(blocks, aiida_localhost),
            kgrid=[2, 2, 2],
            spin_polarized=spin_polarized,
        )
        # Task names inherit the per-call ``call_link_label``.
        names = [t.name for t in wg.tasks]
        fold_names = [name for name in names if name.startswith("fold_")]
        merge_names = [name for name in names if name.startswith("merge_")]
        extract_names = [name for name in names if name.startswith("extract_")]
        assert fold_names == [f"fold_{b['label']}" for b in blocks]
        # One chk/hr extraction per block lifts the wannier90 artefacts into
        # SinglefileData nodes.
        assert extract_names == [f"extract_{b['label']}" for b in blocks]
        assert len(merge_names) == n_merges
        assert names.count("map_zone") == 0


class TestEagerBuildMissingCodes:
    """A missing required code builds eagerly and is reported at run start.

    The body defers ``codes`` member access through
    ``workgraphs.utils.codes.get`` (interim for node-graph#169's ``ref()``),
    so ``check_before_run`` reports the unset socket instead of the build
    dying on a bare ``KeyError``.
    """

    def test_missing_merge_evc_surfaces_at_run(self, fold_codes, aiida_localhost):
        from aiida.orm import RemoteData
        from aiida_workgraph.errors import MissingRequiredInputsError

        blocks = _spinless_blocks()
        groups = group_blocks_to_merge(blocks, {SpinChannel.NONE: 4})

        def _build(codes):
            return FoldToSupercell.build(
                codes=codes,
                blocks=blocks,
                merge_groups=groups,
                nscf_remote_folder=RemoteData(
                    computer=aiida_localhost, remote_path="/fake/nscf"
                ).store(),
                block_wannier=_fake_block_wannier(blocks, aiida_localhost),
                kgrid=[2, 2, 2],
            )

        wg = _build({"wann2kcp": fold_codes["wann2kcp"]})
        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.run()
        entries = [entry for entry in excinfo.value.missing if ".codes." in entry.socket_path]
        assert [entry.socket_path for entry in entries] == ["graph_inputs.codes.merge_evc"]
        assert (entries[0].help or "").startswith("Needed ")

        # Control: with both codes supplied, the codes namespace is clean.
        try:
            _build(fold_codes).check_before_run()
        except MissingRequiredInputsError as exc:
            assert not [entry for entry in exc.missing if ".codes." in entry.socket_path]
