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
    _parse_win_convergence,
    _plain_options,
    _subblock_w90_parameters,
    detect_band_groups,
    extract_win_file,
    merge_split_block_products,
    merge_win_convergence,
)
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks
from tests.fixtures import assert_graph_roundtrips

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


#: Keyword header of the ``.win`` that ``mrwf`` (WannierIO.jl) writes next to
#: each split sub-block, followed by a structure block carrying an ``=`` line
#: that must not be mistaken for a keyword.
_EMITTED_SUBBLOCK_WIN = """\
# Created by WannierIO.jl

conv_tol = 4.0e-7
conv_window = 3
dis_conv_tol = 4.0e-7
dis_num_iter = 0
fermi_energy = 9.2768886075
mp_grid = 4  4  4
num_cg_steps = 200
num_iter = 2000
num_wann = 2
write_hr = true
write_u_matrices = .true.
auto_projections = true

begin unit_cell_cart
angstrom
num_iter = 999
end unit_cell_cart
"""


class TestParseWinConvergence:
    """Harvesting the convergence keywords from the split-emitted ``.win``."""

    def test_extracts_convergence_keywords_with_types(self):
        harvested = _parse_win_convergence(_EMITTED_SUBBLOCK_WIN)
        assert harvested == {
            "num_iter": 2000,
            "num_cg_steps": 200,
            "conv_window": 3,
            "conv_tol": 4.0e-7,
        }
        assert isinstance(harvested["num_iter"], int)
        assert isinstance(harvested["num_cg_steps"], int)
        assert isinstance(harvested["conv_window"], int)
        assert isinstance(harvested["conv_tol"], float)

    def test_non_convergence_keywords_are_not_harvested(self):
        """Counts, toggles and dis_* stay owned by the parameter builder."""
        harvested = _parse_win_convergence(_EMITTED_SUBBLOCK_WIN)
        for key in ("num_wann", "mp_grid", "write_hr", "dis_num_iter", "fermi_energy"):
            assert key not in harvested

    def test_fortran_exponent_marker(self):
        assert _parse_win_convergence("conv_tol =   4.0000000000d-07\n") == {"conv_tol": 4.0e-7}

    def test_block_contents_are_skipped(self):
        """An ``=`` line inside a begin/end block is not a keyword."""
        assert _parse_win_convergence(_EMITTED_SUBBLOCK_WIN)["num_iter"] == 2000


class TestMergeWinConvergence:
    """Emitted convergence keywords merge under the built parameters."""

    @staticmethod
    def _win_file():
        import io as _io

        from aiida.orm import SinglefileData

        return SinglefileData(_io.BytesIO(_EMITTED_SUBBLOCK_WIN.encode()), filename="aiida.win")

    def test_emitted_keywords_fill_the_gaps(self, aiida_profile):
        from aiida.orm import Dict

        base = _subblock_w90_parameters(2, [4, 4, 4], None)
        merged = merge_win_convergence._callable(
            win_file=self._win_file(), parameters=Dict(base)
        ).get_dict()
        assert merged["num_iter"] == 2000
        assert merged["num_cg_steps"] == 200
        assert merged["conv_window"] == 3
        assert merged["conv_tol"] == 4.0e-7
        # The built parameters ride along untouched.
        assert merged["num_wann"] == 2
        assert merged["num_bands"] == 2
        assert merged["mp_grid"] == [4, 4, 4]

    def test_user_override_wins(self, aiida_profile):
        from aiida.orm import Dict

        base = _subblock_w90_parameters(2, [4, 4, 4], {"num_iter": 500})
        merged = merge_win_convergence._callable(
            win_file=self._win_file(), parameters=Dict(base)
        ).get_dict()
        assert merged["num_iter"] == 500
        assert merged["num_cg_steps"] == 200

    def test_without_the_merge_the_parameters_lack_convergence_keywords(self):
        """Negative control: the base builder alone emits no convergence set.

        This pins the defect the harvest fixes — dropping the merge would
        hand wannier90 its compiled-in ``num_iter``/``num_cg_steps``.
        """
        base = _subblock_w90_parameters(2, [4, 4, 4], None)
        for key in ("num_iter", "num_cg_steps", "conv_tol", "conv_window"):
            assert key not in base


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

        assert_graph_roundtrips(wg)

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
            split_threshold=1.5,
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
        # The unsplit branch emits the required contract from the
        # whole-block run — its gauge is final, so its parsed
        # ``output_parameters`` rides along; the folder keys stay
        # unpopulated (runtime ``None``, uniformly across the split
        # route's branches).
        for name in ("u_file", "hr_file", "centres_file", "nnkp_file", "output_parameters"):
            assert wg.outputs[name]._links, name
        for name in ("retrieved", "remote_folder"):
            assert not wg.outputs[name]._links, name
        assert_graph_roundtrips(wg)

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

        # Both dynamic namespaces of the split feed the nested graph: the
        # per-block folders and the emitted per-block ``.win`` files (whose
        # convergence keywords seed the re-wannierisations).
        for namespace in ("split_blocks", "split_win_files"):
            links = rewann_task.inputs[namespace]._links
            assert len(links) == 1, namespace
            assert links[0].from_task.name == "split_wannierization", namespace

        # The merged trio and the merged parsed Dict feed the outputs; the
        # plain-route-only folder keys stay unpopulated on the split route.
        for name in ("u_file", "hr_file", "centres_file", "nnkp_file", "output_parameters"):
            assert wg.outputs[name]._links, name
        for name in ("retrieved", "remote_folder"):
            assert not wg.outputs[name]._links, name
        assert_graph_roundtrips(wg)

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
        import io as _io

        from aiida.orm import FolderData, SinglefileData

        from aiida_koopmans.workgraphs.auto_wannierize import RewannierizeSplitBlocks

        split_blocks = {
            "block_0": FolderData().store(),
            "block_1": FolderData().store(),
        }
        split_win_files = {
            f"block_{i}": SinglefileData(
                _io.BytesIO(_EMITTED_SUBBLOCK_WIN.encode()), filename="aiida.win"
            ).store()
            for i in range(2)
        }
        wg = RewannierizeSplitBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            split_blocks=split_blocks,
            split_win_files=split_win_files,
            group_sizes=[4, 4],
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            wannier90_overrides={"num_iter": 500, "dis_froz_max": 10.0},
        )
        names = [t.name for t in wg.tasks]
        assert "merge_split_block_products" in names
        assert "merge_wannier_output_parameters" in names
        w90_tasks = [t for t in wg.tasks if t.name.startswith("wannier90_split_block")]
        assert len(w90_tasks) == 2

        # Each re-wannierisation is preprocessing-free (local_input_folder
        # wired from the split's per-block folder). Its parameters are the
        # merge of the split-emitted convergence keywords under the built
        # set: each wannier90 ``parameters`` is wired from the per-group
        # ``merge_win_convergence`` task, which reads the group's emitted
        # ``.win`` and the built base parameters (num_bands == num_wann, no
        # disentanglement keys, user minimisation settings propagating).
        for i, w90_task in enumerate(sorted(w90_tasks, key=lambda t: t.name)):
            merge_task = wg.tasks[f"merge_win_convergence_{i}"]
            links = w90_task.inputs["parameters"]._links
            assert len(links) == 1
            assert links[0].from_task.name == merge_task.name
            assert merge_task.inputs["win_file"].value.uuid == split_win_files[f"block_{i}"].uuid

            params = merge_task.inputs["parameters"].value
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
        assert_graph_roundtrips(wg)


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


class TestMergeWannierOutputParameters:
    """Per-group parsed outputs concatenate block-wide in group order."""

    @staticmethod
    def _group_parameters(spreads, start=1):
        from aiida.orm import Dict

        return Dict(
            {
                "number_wfs": len(spreads),
                "wannier_functions_output": [
                    {"wf_ids": i, "wf_centres": [0.1 * i, 0.0, 0.0], "wf_spreads": spread}
                    for i, spread in enumerate(spreads, start=start)
                ],
            }
        )

    def test_group_order_concatenation_and_wf_id_rebasing(self, aiida_profile):
        from aiida_koopmans.workgraphs.auto_wannierize import merge_wannier_output_parameters

        merged = merge_wannier_output_parameters._callable(
            b00=self._group_parameters([1.1, 2.2]),
            b01=self._group_parameters([3.3]),
        ).get_dict()
        assert merged["number_wfs"] == 3
        assert [wf["wf_ids"] for wf in merged["wannier_functions_output"]] == [1, 2, 3]
        assert [wf["wf_spreads"] for wf in merged["wannier_functions_output"]] == [1.1, 2.2, 3.3]

    def test_swapped_group_keys_swap_the_band_order(self, aiida_profile):
        """Negative control: the keys, not insertion order, define the order.

        Assigning the groups to swapped keys yields the swapped
        concatenation — proving the merge would mis-order bands if the
        caller mislabelled the groups.
        """
        from aiida_koopmans.workgraphs.auto_wannierize import merge_wannier_output_parameters

        merged = merge_wannier_output_parameters._callable(
            b01=self._group_parameters([1.1, 2.2]),
            b00=self._group_parameters([3.3]),
        ).get_dict()
        assert [wf["wf_spreads"] for wf in merged["wannier_functions_output"]] == [3.3, 1.1, 2.2]
        assert [wf["wf_ids"] for wf in merged["wannier_functions_output"]] == [1, 2, 3]

    def test_out_of_order_entries_are_sorted_before_rebasing(self, aiida_profile):
        from aiida_koopmans.workgraphs.auto_wannierize import merge_wannier_output_parameters

        params = self._group_parameters([1.1, 2.2])
        shuffled = params.get_dict()
        shuffled["wannier_functions_output"].reverse()
        from aiida.orm import Dict

        merged = merge_wannier_output_parameters._callable(b00=Dict(shuffled)).get_dict()
        assert [wf["wf_spreads"] for wf in merged["wannier_functions_output"]] == [1.1, 2.2]

    def test_wf_count_mismatch_raises(self, aiida_profile):
        from aiida_koopmans.workgraphs.auto_wannierize import merge_wannier_output_parameters

        params = self._group_parameters([1.1])
        broken = params.get_dict()
        broken["number_wfs"] = 2
        from aiida.orm import Dict

        with pytest.raises(ValueError, match="declares"):
            merge_wannier_output_parameters._callable(b00=Dict(broken))


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
            split_threshold=1.5,
            pseudo_family=fake_cutoffs_family.label,
            overrides=overrides,
        )
        forwarded = wg.tasks["scf_nscf"].inputs["overrides"].value
        assert forwarded["scf"]["pw"]["parameters"]["SYSTEM"]["ecutwfc"] == 30.0
        assert forwarded["nscf"]["pw"]["parameters"]["SYSTEM"]["nbnd"] == 12
