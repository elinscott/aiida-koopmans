"""Tests for the automated block-splitting Wannierisation.

Pure-function tests for the band-group detection/restriction helpers, plus
construction-level graph tests: the split-mode ``WannierizeBlocks`` build
(shared scf+nscf, the bands step, the runtime detection task, one nested
per-block graph per block) and eager ``WannierizeAndSplitBlock`` builds with
concrete groups, which execute the normally-deferred body and expose both
the unsplit and the split branches. Nothing runs — dummy codes only.
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
    _plain_options,
    _subblock_w90_parameters,
    detect_band_groups,
    extract_win_file,
    merge_split_block_products,
)
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks

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
        wg = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=blocks,
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            bands_kpoints=kpath,
            num_occ_bands=4,
            split_threshold=1.5,
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

    def test_parallelization_reaches_the_shared_pw_steps(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        """The pw mapping lands on the bands step and threads into scf+nscf."""
        blocks = [_explicit_block("block_1", range(1, 5), ["Si: sp3"])]
        wg = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=blocks,
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            bands_kpoints=kpath,
            num_occ_bands=4,
            pseudo_family=fake_cutoffs_family.label,
            parallelization={"pw": {"ntasks": 3, "npool": 2}},
        )

        # The bands step is a direct pw step in this graph, so the resources
        # and -npool flag are merged straight onto its pw namespace.
        bands_pw = wg.tasks["bands"].inputs["pw"]
        assert bands_pw["metadata"]["options"]["resources"].value["num_mpiprocs_per_machine"] == 3
        assert bands_pw["settings"].value["cmdline"] == ["-npool", "2"]

        # The scf+nscf pair runs inside a nested graph, which receives the
        # mapping as an input rather than dropping it.
        assert wg.tasks["scf_nscf"].inputs["parallelization"].value == {
            "pw": {"ntasks": 3, "npool": 2}
        }


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


# ----------------------------------------------------------------------
# Leaf calcfunctions (run in-process via ._callable) and helpers
# ----------------------------------------------------------------------


def _bands_data(array):
    """Wrap an eigenvalue array (2D or 3D) in a ``BandsData``."""
    from aiida.orm import BandsData, KpointsData

    array = np.asarray(array, dtype=float)
    nkpts = array.shape[-2]
    kpts = KpointsData()
    kpts.set_kpoints([[i / max(nkpts, 1), 0.0, 0.0] for i in range(nkpts)])
    bands = BandsData()
    bands.set_kpointsdata(kpts)
    bands.set_bands(array)
    return bands


class TestDetectBandGroupsCalcfunction:
    """The runtime wrapper reshapes the eigenvalues before grouping."""

    def test_truncates_to_the_wannierised_manifold(self, aiida_profile):
        """``num_bands_total`` drops the disentanglement pool above the manifold."""
        # Band 5 sits far above the manifold; it must not influence the groups.
        bands = _bands_data([[0.0, 0.1, 5.0, 5.1, 20.0]])
        groups = detect_band_groups._callable(
            bands=bands, num_occ_bands=2, threshold=2.0, num_bands_total=4
        )
        assert groups.get_list() == [[1, 2], [3, 4]]

    def test_selects_the_requested_spin_channel(self, aiida_profile):
        """A 3D (spin-resolved) bands array is indexed by ``spin_channel_index``."""
        spin_up = [[0.0, 0.5, 4.0, 4.6], [0.4, 0.9, 4.5, 4.9]]  # 2 eV gap -> two groups
        spin_down = [[0.0, 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]  # no gap -> one group
        bands = _bands_data([spin_up, spin_down])
        assert detect_band_groups._callable(
            bands=bands, threshold=2.0, spin_channel_index=0
        ).get_list() == [[1, 2], [3, 4]]
        assert detect_band_groups._callable(
            bands=bands, threshold=2.0, spin_channel_index=1
        ).get_list() == [[1, 2, 3, 4]]


class TestExtractWinFile:
    """Recovering the ``.win`` from the wannier90 calculation that wrote it."""

    def test_missing_creator_raises(self, aiida_profile):
        """A folder with no creating calculation cannot yield its ``.win``."""
        from aiida.orm import FolderData

        with pytest.raises(ValueError, match="no creating calculation"):
            extract_win_file._callable(retrieved=FolderData().store())

    def test_reads_the_win_from_the_creator(self, aiida_localhost):
        """The ``.win`` is read back off the creating calculation's repository."""
        from aiida.common.links import LinkType
        from aiida.orm import CalcJobNode, FolderData

        calc = CalcJobNode(
            computer=aiida_localhost,
            process_type="aiida.calculations:core.arithmetic.add",
        )
        calc.set_option("resources", {"num_machines": 1})
        calc.set_option("input_filename", "aiida.win")
        calc.base.repository.put_object_from_bytes(b"num_wann = 4\n", "aiida.win")
        calc.store()
        retrieved = FolderData()
        retrieved.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label="retrieved")
        retrieved.store()
        calc.seal()

        win = extract_win_file._callable(retrieved=retrieved)
        assert win.filename == "aiida.win"
        assert win.get_content() == "num_wann = 4\n"


class TestMergeSplitBlockProducts:
    """Per-sub-block products merge block-diagonally in band order."""

    def test_block_diagonal_merge(self, aiida_profile):
        """Two 2-WF sub-blocks merge into one 4-WF block-diagonal product set."""
        from aiida.orm import FolderData

        from aiida_koopmans.wannier_merge import (
            generate_wannier_centres_file_contents,
            generate_wannier_hr_file_contents,
            generate_wannier_u_file_contents,
            parse_wannier_centres_file_contents,
            parse_wannier_hr_file_contents,
            parse_wannier_u_file_contents,
        )

        rvect = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
        weights = [1, 2, 2]
        kpts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        atom_lines = ["Si       0.00000000      0.00000000      0.00000000"]

        def _folder(seed):
            rng = np.random.default_rng(seed)
            umat = rng.random((2, 2, 2)) + 1j * rng.random((2, 2, 2))
            ham = rng.random((3, 2, 2)) + 1j * rng.random((3, 2, 2))
            centres = [[0.1 * seed, 0.0, 0.0], [0.2 * seed, 0.0, 0.0]]
            folder = FolderData()
            folder.base.repository.put_object_from_bytes(
                generate_wannier_u_file_contents(umat, kpts).encode(), "aiida_u.mat"
            )
            folder.base.repository.put_object_from_bytes(
                generate_wannier_hr_file_contents(ham, rvect, weights).encode(), "aiida_hr.dat"
            )
            folder.base.repository.put_object_from_bytes(
                generate_wannier_centres_file_contents(centres, atom_lines).encode(),
                "aiida_centres.xyz",
            )
            return folder.store()

        merged = merge_split_block_products._callable(b00=_folder(1), b01=_folder(2))

        umat, _ = parse_wannier_u_file_contents(merged["u_file"].get_content())
        assert umat.shape == (2, 4, 4)
        # The two sub-blocks occupy the diagonal 2x2 blocks; the off-diagonal
        # blocks are exactly zero.
        np.testing.assert_allclose(umat[:, :2, 2:], 0.0)
        np.testing.assert_allclose(umat[:, 2:, :2], 0.0)

        ham, _, _ = parse_wannier_hr_file_contents(merged["hr_file"].get_content())
        assert ham.shape == (3, 4, 4)

        centres, atom_back = parse_wannier_centres_file_contents(
            merged["centres_file"].get_content()
        )
        assert len(centres) == 4  # 2 + 2 concatenated
        assert len(atom_back) == 1


class TestPlainOptions:
    """Rebuilding CalcJob ``metadata.options`` free of provenance proxies."""

    def test_defaults_when_absent(self):
        assert _plain_options(None) == {"resources": {"num_machines": 1}}
        assert _plain_options({}) == {"resources": {"num_machines": 1}}

    def test_rebuilds_nested_mapping_into_a_fresh_dict(self):
        opts = {"resources": {"num_machines": 2}, "max_wallclock_seconds": 60}
        rebuilt = _plain_options(opts)
        assert rebuilt == opts
        assert rebuilt is not opts
        assert rebuilt["resources"] is not opts["resources"]


class TestOverridesForwarding:
    """The scf/nscf override entries are split out to the shared pair."""

    def test_scf_and_nscf_overrides_reach_the_shared_pair(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        """``overrides['scf']`` / ``overrides['nscf']`` forward to RunScfNscf."""
        blocks = [_explicit_block("block_1", range(1, 5), ["Si: sp3"])]
        overrides = {
            "scf": {"pw": {"parameters": {"SYSTEM": {"ecutwfc": 30.0}}}},
            "nscf": {"pw": {"parameters": {"SYSTEM": {"nbnd": 12}}}},
        }
        wg = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=blocks,
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            bands_kpoints=kpath,
            num_occ_bands=4,
            pseudo_family=fake_cutoffs_family.label,
            overrides=overrides,
        )
        forwarded = wg.tasks["scf_nscf"].inputs["overrides"].value
        assert forwarded["scf"]["pw"]["parameters"]["SYSTEM"]["ecutwfc"] == 30.0
        assert forwarded["nscf"]["pw"]["parameters"]["SYSTEM"]["nbnd"] == 12
