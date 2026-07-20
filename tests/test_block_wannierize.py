"""Construction-level unit tests for the block-by-block Wannierize workgraph.

These build the ``WannierizeBlocks`` graph (no daemon, no real codes
execution) and introspect its task list. The per-block fan-out is a native
``for`` loop in the (top-level) graph body, which runs at build time over the
concrete ``blocks`` list -- so the built graph shows one ``WannierizeBlock``
per block plus a single shared ``scf_nscf`` task.
"""

import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlock, WannierizeBlocks

# ----------------------------------------------------------------------
# Fixtures: codes, structures, block shapes
# ----------------------------------------------------------------------


@pytest.fixture
def wannier_codes(aiida_localhost):
    """Return the ``Codes`` dict of stand-in InstalledCode nodes.

    The codes never execute (these are construction-only tests); they exist
    only so the ``Codes`` input namespace is populated with real
    ``AbstractCode`` nodes, which the builder / namespace validators require.
    """
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
        "pw": _code("bw-pw", "quantumespresso.pw"),
        "wannier90": _code("bw-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("bw-p2w", "quantumespresso.pw2wannier90"),
        "projwfc": _code("bw-pjw", "quantumespresso.projwfc"),
    }


@pytest.fixture
def zno_structure(aiida_profile):
    """Return a 4-atom periodic wurtzite-ish ZnO ``StructureData``."""
    from aiida.orm import StructureData

    cell = [[3.25, 0.0, 0.0], [-1.625, 2.814, 0.0], [0.0, 0.0, 5.2]]
    struct = StructureData(cell=cell, pbc=True)
    struct.append_atom(position=(0.0, 0.0, 0.0), symbols="Zn", name="Zn")
    struct.append_atom(position=(1.625, 0.938, 2.6), symbols="Zn", name="Zn")
    struct.append_atom(position=(0.0, 0.0, 1.95), symbols="O", name="O")
    struct.append_atom(position=(1.625, 0.938, 4.55), symbols="O", name="O")
    return struct


def _explicit_block(label: str, include: range) -> ExplicitProjectionBlock:
    """Build a minimal nspin=1 explicit (ANALYTIC) block over ``include`` bands."""
    n = len(include)
    return ExplicitProjectionBlock(
        label=label,
        spin=SpinChannel.NONE,
        num_wann=n,
        num_bands=n,
        include_bands=list(include),
        projection_type=WannierProjectionType.ANALYTIC,
        projections=[],
    )


def _silicon_blocks() -> list[ExplicitProjectionBlock]:
    """tutorial_2 silicon shape: 1 occupied block + 1 empty block, nspin=1."""
    return [
        _explicit_block("block_1", range(1, 5)),  # 4 occupied
        _explicit_block("block_2", range(5, 9)),  # 4 empty
    ]


def _zno_blocks() -> list[ExplicitProjectionBlock]:
    """ZnO shape: 4 occupied blocks + 1 empty block, nspin=1."""
    return [
        _explicit_block("block_1", range(1, 6)),  # Zn 3d-ish
        _explicit_block("block_2", range(6, 9)),
        _explicit_block("block_3", range(9, 13)),
        _explicit_block("block_4", range(13, 17)),
        _explicit_block("block_5", range(17, 21)),  # empty
    ]


# ----------------------------------------------------------------------
# Graph construction: shared scf+nscf once, one WannierizeBlock per block
# ----------------------------------------------------------------------


def _build(codes, structure, blocks, kpoints):
    return WannierizeBlocks.build(
        codes=codes,
        structure=structure,
        blocks=blocks,
        kpoints=kpoints,
        pseudo_family="SSSP/1.3/PBE/efficiency",
    )


class TestBlockWannierizeGraphBuild:
    @pytest.mark.parametrize(
        "structure_fixture,blocks_factory,n_blocks",
        [("silicon_structure", _silicon_blocks, 2), ("zno_structure", _zno_blocks, 5)],
    )
    def test_graph_builds_one_block_per_block(
        self, request, wannier_codes, kmesh, structure_fixture, blocks_factory, n_blocks
    ):
        structure = request.getfixturevalue(structure_fixture)
        wg = _build(wannier_codes, structure, blocks_factory(), kmesh)
        names = [t.name for t in wg.tasks]

        # Shared scf+nscf appears exactly once.
        assert names.count("scf_nscf") == 1
        # The native for-loop unrolls at build time over the concrete blocks
        # list: one independent WannierizeBlock per block (aiida-workgraph
        # auto-suffixes the repeats: WannierizeBlock, BlockWannierize1, ...).
        # No Map zone.
        n_block_tasks = sum(1 for name in names if name.startswith("WannierizeBlock"))
        assert n_block_tasks == n_blocks
        assert names.count("map_zone") == 0

    def test_scf_nscf_overrides_reach_the_shared_pair(
        self, wannier_codes, silicon_structure, kmesh
    ):
        """`scf` / `nscf` override entries feed the shared pair, nothing else."""
        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            overrides={
                "scf": {"pw": {"parameters": {"SYSTEM": {"ecutwfc": 70.0}}}},
                "nscf": {"pw": {"parameters": {"SYSTEM": {"nbnd": 20}}}},
            },
        )
        pw_overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        assert pw_overrides["scf"]["pw"]["parameters"]["SYSTEM"]["ecutwfc"] == 70.0
        assert pw_overrides["nscf"]["pw"]["parameters"]["SYSTEM"]["nbnd"] == 20
        assert "wannier90" not in pw_overrides


# ----------------------------------------------------------------------
# Eager per-block build: the flat WannierizeOverrides -> builder translation
# ----------------------------------------------------------------------


class TestWannierizeBlockBuild:
    """Build ``WannierizeBlock`` directly so its (normally deferred) body runs.

    Inside ``WannierizeBlocks`` the per-block graph is a deferred subgraph
    task, so the construction tests above never execute its body. Building it
    directly exercises the translation of the flat :class:`WannierizeOverrides`
    keys (``wannier90`` / ``pw2wannier90``) into the upstream
    namespace-mirroring builder shape, plus the per-block parameter edits.
    """

    @pytest.fixture
    def nscf_scratch(self, aiida_localhost, tmp_path):
        """Return a stand-in ``RemoteData`` for the shared nscf scratch."""
        from aiida.orm import RemoteData

        return RemoteData(computer=aiida_localhost, remote_path=str(tmp_path))

    def _build_block(
        self,
        codes,
        structure,
        kpoints,
        nscf_scratch,
        block,
        pseudo_family,
        overrides=None,
        mp_grid=None,
    ):
        return WannierizeBlock.build(
            codes=codes,
            structure=structure,
            block=block,
            projection_type=WannierProjectionType.ANALYTIC,
            nscf_remote_folder=nscf_scratch,
            kpoints=kpoints,
            mp_grid=mp_grid,
            pseudo_family=pseudo_family,
            overrides=overrides,
        )

    @staticmethod
    def _wannier_task(wg):
        matches = [t for t in wg.tasks if "annier90" in t.name]
        assert matches, f"no wannier90 task among {[t.name for t in wg.tasks]}"
        return matches[0]

    def test_flat_overrides_reach_the_builder_namespaces(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """`wannier90` / `pw2wannier90` land in `parameters` / `INPUTPP`; `scf` is ignored."""
        block = ExplicitProjectionBlock(
            label="block_1",
            spin=SpinChannel.NONE,
            num_wann=4,
            num_bands=6,
            projection_type=WannierProjectionType.ANALYTIC,
            projections=["Si: sp3"],
        )
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
            overrides={
                "wannier90": {"dis_froz_max": 10.6, "dis_num_iter": 200},
                "pw2wannier90": {"write_unk": True},
                # Belongs to the shared scf+nscf pair; the block-level graph
                # must not wrap it into the builder overrides.
                "scf": {"pw": {"parameters": {"SYSTEM": {"ecutwfc": 999.0}}}},
            },
            mp_grid=[2, 2, 2],
        )
        task = self._wannier_task(wg)

        params = task.inputs["wannier90"]["wannier90"]["parameters"].value.get_dict()
        assert params["dis_froz_max"] == 10.6
        # A disentangling block keeps a user-supplied iteration budget instead
        # of the 5000-iteration default.
        assert params["dis_num_iter"] == 200
        assert params["num_wann"] == 4
        assert params["num_bands"] == 6
        assert params["write_hr"] is True
        assert params["mp_grid"] == [2, 2, 2]

        inputpp = task.inputs["pw2wannier90"]["pw2wannier90"]["parameters"].value.get_dict()
        assert inputpp["INPUTPP"]["write_unk"] is True

        # scf and nscf are skipped (their namespaces stay unpopulated, which
        # is how the workchain decides to skip the steps); the shared scratch
        # is the pw2wannier90 parent. In particular the `scf` override above
        # must not have leaked in.
        assert task.inputs["scf"]["pw"]["parameters"].value is None
        assert task.inputs["nscf"]["pw"]["parameters"].value is None
        # `.value` arrives as a provenance-tagged proxy, so compare identity
        # via the node uuid rather than `is`.
        parent = task.inputs["pw2wannier90"]["pw2wannier90"]["parent_folder"].value
        assert parent.uuid == nscf_scratch.uuid

        # aiida.chk is forced into the wannier90 retrieve list.
        settings = task.inputs["wannier90"]["wannier90"]["settings"].value.get_dict()
        assert "aiida.chk" in settings["additional_retrieve_list"]

        projections = task.inputs["wannier90"]["wannier90"]["projections"].value
        assert list(projections) == ["Si: sp3"]

    def test_no_overrides_defaults(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """num_bands == num_wann strips the windows; mp_grid=None drops the key."""
        block = _explicit_block("block_1", range(1, 5))
        block["exclude_bands"] = [9, 10]
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
        )
        task = self._wannier_task(wg)

        params = task.inputs["wannier90"]["wannier90"]["parameters"].value.get_dict()
        assert params["num_wann"] == 4
        assert params["num_bands"] == 4
        assert params["exclude_bands"] == [9, 10]
        for key in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"):
            assert key not in params
        assert params["write_hr"] is True
        assert "mp_grid" not in params

    def test_disentangling_block_gets_the_default_iteration_budget(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """num_bands > num_wann without a user dis_num_iter unfreezes the subspace."""
        block = ExplicitProjectionBlock(
            label="block_1",
            spin=SpinChannel.NONE,
            num_wann=4,
            num_bands=6,
            projection_type=WannierProjectionType.ANALYTIC,
            projections=["Si: sp3"],
        )
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
        )
        task = self._wannier_task(wg)
        params = task.inputs["wannier90"]["wannier90"]["parameters"].value.get_dict()
        assert params["dis_num_iter"] == 5000
