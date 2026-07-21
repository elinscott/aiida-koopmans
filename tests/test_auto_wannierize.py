"""Tests for the automated block-splitting Wannierisation.

Pure-function tests for the band-group detection/restriction helpers, plus
construction-level graph tests: the top-level ``WannierizeAndSplitBlocks``
build (shared scf+nscf, the bands step, the runtime detection task, one
nested per-block graph per block) and eager ``WannierizeAndSplitBlock``
builds with concrete groups, which execute the normally-deferred body and
expose both the unsplit and the split branches. Nothing runs — dummy codes
only.
"""

from __future__ import annotations

import numpy as np
import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.projections import (
    detect_band_blocks,
    groups_to_wannier_indices,
    restrict_groups_to_block,
)
from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
from aiida_koopmans.workgraphs.auto_wannierize import (
    WannierizeAndSplitBlock,
    WannierizeAndSplitBlocks,
    _subblock_w90_parameters,
)

# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


class TestDetectBandBlocks:
    def test_gap_splitting(self):
        """A gap wider than the threshold everywhere in the BZ opens a group."""
        # Bands (nkpts=2): 1 and 2 overlap; 3 sits > 2 eV above 2; 4 touches 3.
        energies = np.array(
            [
                [0.0, 0.5, 4.0, 4.6],
                [0.4, 0.9, 4.5, 4.9],
            ]
        )
        assert detect_band_blocks(energies, threshold=2.0) == [[1, 2], [3, 4]]

    def test_occupied_boundary_always_splits(self):
        """Band num_occ_bands + 1 opens a group even with no energy gap."""
        energies = np.array([[0.0, 0.1, 0.2, 0.3]])
        assert detect_band_blocks(energies, num_occ_bands=2) == [[1, 2], [3, 4]]

    def test_no_criteria_yields_one_group(self):
        energies = np.array([[0.0, 10.0, 20.0]])
        assert detect_band_blocks(energies) == [[1, 2, 3]]

    def test_gap_must_hold_across_the_whole_bz(self):
        """A gap at one k-point that closes at another does not split."""
        energies = np.array(
            [
                [0.0, 5.0],  # large gap here...
                [0.0, 0.5],  # ...but not here
            ]
        )
        assert detect_band_blocks(energies, threshold=2.0) == [[1, 2]]

    def test_boundary_and_gap_combine(self):
        energies = np.array([[0.0, 0.1, 5.0, 5.1, 20.0]])
        assert detect_band_blocks(energies, num_occ_bands=2, threshold=2.0) == [
            [1, 2],
            [3, 4],
            [5],
        ]


class TestGroupRestriction:
    def test_overlap_filtering(self):
        groups = [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert restrict_groups_to_block(groups, [5, 6, 7, 8]) == [[5, 6, 7, 8]]

    def test_group_spanning_two_blocks_is_split_between_them(self):
        groups = [[1, 2, 3, 4, 5, 6]]
        assert restrict_groups_to_block(groups, [1, 2, 3]) == [[1, 2, 3]]
        assert restrict_groups_to_block(groups, [4, 5, 6]) == [[4, 5, 6]]

    def test_uncovered_block_band_raises(self):
        with pytest.raises(ValueError, match="must span every"):
            restrict_groups_to_block([[1, 2]], [1, 2, 3])

    def test_wannier_index_rebasing(self):
        """Global band groups map to 1-based positions within the block."""
        assert groups_to_wannier_indices([[5, 6], [7, 8]], [5, 6, 7, 8]) == [[1, 2], [3, 4]]
        assert groups_to_wannier_indices([[1, 2]], [1, 2]) == [[1, 2]]


class TestSubblockParameters:
    def test_forced_keys_and_dis_stripping(self):
        params = _subblock_w90_parameters(
            4,
            [2, 2, 2],
            {"dis_froz_max": 10.0, "dis_num_iter": 200, "num_iter": 500, "exclude_bands": [9]},
        )
        assert params["num_wann"] == 4
        assert params["num_bands"] == 4
        assert params["mp_grid"] == [2, 2, 2]
        assert params["write_hr"] is True
        assert params["write_u_matrices"] is True
        assert params["write_xyz"] is True
        # User minimisation settings propagate; disentanglement and band
        # exclusion must not (the split files cover exactly the group).
        assert params["num_iter"] == 500
        assert "dis_froz_max" not in params
        assert "dis_num_iter" not in params
        assert "exclude_bands" not in params


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


@pytest.fixture
def auto_codes(aiida_localhost):
    """Stand-in codes for construction-only builds (never executed)."""
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
        "pw": _code("aw-pw", "quantumespresso.pw"),
        "wannier90": _code("aw-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("aw-p2w", "quantumespresso.pw2wannier90"),
        "wannierjl": _code("aw-wjl", "wannierjl.check_neighbors"),
    }


@pytest.fixture
def kpath(aiida_profile):
    """Return a short explicit k-path ``KpointsData``."""
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]])
    return kpts


@pytest.fixture
def nscf_scratch(aiida_localhost, tmp_path):
    """Return a stand-in ``RemoteData`` for the shared nscf scratch."""
    from aiida.orm import RemoteData

    return RemoteData(computer=aiida_localhost, remote_path=str(tmp_path))


def _explicit_block(label: str, include: range, projections: list[str]) -> ExplicitProjectionBlock:
    n = len(include)
    return ExplicitProjectionBlock(
        label=label,
        spin=SpinChannel.NONE,
        num_wann=n,
        num_bands=n,
        include_bands=list(include),
        projection_type=WannierProjectionType.ANALYTIC,
        projections=projections,
    )


class TestTopLevelGraphBuild:
    def test_shared_steps_and_per_block_fanout(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        """One scf+nscf, one bands step, one detection, one nested graph per block."""
        blocks = [
            _explicit_block("block_1", range(1, 5), ["Si: sp3"]),
            _explicit_block("block_2", range(5, 9), ["Si: sp3"]),
        ]
        wg = WannierizeAndSplitBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=blocks,
            kpoints=kmesh,
            bands_kpoints=kpath,
            num_occ_bands=4,
            threshold=1.5,
            pseudo_family=fake_cutoffs_family.label,
        )
        names = [t.name for t in wg.tasks]
        assert names.count("scf_nscf") == 1
        assert names.count("bands") == 1
        assert names.count("detect_band_groups") == 1
        # Nested graph tasks are named by their call_link_label.
        n_block_tasks = sum(1 for name in names if name.startswith("wannierize_split_block"))
        assert n_block_tasks == 2
        assert names.count("map_zone") == 0

        # The bands step runs a `bands` calculation off the scf density, on
        # the explicit path.
        bands_task = wg.tasks["bands"]
        params = bands_task.inputs["pw"]["parameters"].value.get_dict()
        assert params["CONTROL"]["calculation"] == "bands"
        assert bands_task.inputs["kpoints"].value.uuid == kpath.uuid

        # The detection is restricted to the Wannierised manifold and knows
        # the occupied boundary and the threshold.
        detect_task = wg.tasks["detect_band_groups"]
        assert detect_task.inputs["num_bands_total"].value == 8
        assert detect_task.inputs["num_occ_bands"].value == 4
        assert detect_task.inputs["threshold"].value == 1.5


class TestPerBlockGraphBuild:
    """Eager per-block builds: the deferred body runs with concrete groups."""

    def _build(self, codes, structure, kpoints, nscf_scratch, block, groups, pseudo_family):
        return WannierizeAndSplitBlock.build(
            codes=codes,
            structure=structure,
            block=block,
            groups=groups,
            nscf_remote_folder=nscf_scratch,
            kpoints=kpoints,
            mp_grid=[2, 2, 2],
            pseudo_family=pseudo_family,
        )

    def test_single_group_skips_the_split(
        self, auto_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """A block already isolated by the detection wannierises plainly."""
        block = _explicit_block("block_2", range(5, 9), ["Si: sp3"])
        wg = self._build(
            auto_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            fake_cutoffs_family.label,
        )
        names = [t.name for t in wg.tasks]
        assert "wannierize_whole_block" in names
        assert "extract_win_file" not in names
        assert "split_wannierization" not in names
        assert not any(name.startswith("Wannier90Calculation") for name in names)

    def test_split_branch_topology_and_wiring(
        self, auto_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """Two groups: whole-block wannierize, wjl split, nested re-wannierisation."""
        block = _explicit_block("block_1", range(1, 9), ["Si: sp3", "Si: sp3"])
        wg = self._build(
            auto_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            fake_cutoffs_family.label,
        )
        names = [t.name for t in wg.tasks]
        assert "wannierize_whole_block" in names
        assert "extract_win_file" in names
        assert "split_wannierization" in names
        assert "rewannierize_split_blocks" in names

        # The wjl split gets block-local 1-based Wannier indices and both
        # parent folders point at the whole-block wannier90 scratch (which
        # holds the chk and the staged amn/mmn/eig).
        split_task = wg.tasks["split_wannierization"]
        assert split_task.inputs["groups"].value == [[1, 2, 3, 4], [5, 6, 7, 8]]
        assert split_task.inputs["wjl_code"].value.uuid == auto_codes["wannierjl"].uuid
        assert split_task.inputs["pw2wannier90_code"].value.uuid == auto_codes["pw2wannier90"].uuid
        assert split_task.inputs["nscf_parent"].value.uuid == nscf_scratch.uuid

        # The nested re-Wannierisation knows the group sizes (its fan-out
        # cardinality) up front, even though the split folders are futures.
        rewann_task = wg.tasks["rewannierize_split_blocks"]
        assert rewann_task.inputs["group_sizes"].value == [4, 4]

    def test_groups_are_rebased_for_offset_blocks(
        self, auto_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """A block starting at band 5 hands 1-based local indices to the split."""
        block = _explicit_block("block_2", range(5, 13), ["Si: sp3", "Si: sp3"])
        wg = self._build(
            auto_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            fake_cutoffs_family.label,
        )
        split_task = wg.tasks["split_wannierization"]
        assert split_task.inputs["groups"].value == [[1, 2, 3, 4], [5, 6, 7, 8]]


class TestRewannierizeSplitBlocksBuild:
    """Eager build of the nested re-Wannierisation graph with concrete folders."""

    def test_one_wannier90_per_group_and_merge(
        self, auto_codes, silicon_structure, kmesh, aiida_profile
    ):
        from aiida.orm import FolderData

        from aiida_koopmans.workgraphs.auto_wannierize import RewannierizeSplitBlocks

        split_blocks = {
            "block_0": FolderData().store(),
            "block_1": FolderData().store(),
        }
        wg = RewannierizeSplitBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            split_blocks=split_blocks,
            group_sizes=[4, 4],
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            wannier90_overrides={"num_iter": 500, "dis_froz_max": 10.0},
        )
        names = [t.name for t in wg.tasks]
        assert "merge_split_block_products" in names
        w90_tasks = [t for t in wg.tasks if t.name.startswith("wannier90_split_block")]
        assert len(w90_tasks) == 2

        # Each re-wannierisation is preprocessing-free (local_input_folder
        # wired from the split's per-block folder) with num_bands == num_wann
        # and no disentanglement keys; user minimisation settings propagate.
        for i, w90_task in enumerate(sorted(w90_tasks, key=lambda t: t.name)):
            params = w90_task.inputs["parameters"].value
            params = params.get_dict() if hasattr(params, "get_dict") else dict(params)
            assert params["num_wann"] == 4
            assert params["num_bands"] == 4
            assert params["mp_grid"] == [2, 2, 2]
            assert params["write_hr"] is True
            assert params["write_u_matrices"] is True
            assert params["write_xyz"] is True
            assert params["num_iter"] == 500
            assert not any(key.startswith("dis_") for key in params)
            folder = w90_task.inputs["local_input_folder"].value
            assert folder.uuid == split_blocks[f"block_{i}"].uuid
