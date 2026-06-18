"""Construction-level unit tests for the block-by-block Wannierize workgraph.

These build the ``BlockWannierizeTask`` graph (no daemon, no real codes
execution) and introspect its task list. The per-block fan-out is a native
``for`` loop in the (top-level) graph body, which runs at build time over the
concrete ``blocks`` list -- so the built graph shows one ``BlockWannierize``
per block plus a single shared ``scf_nscf`` task.
"""

import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
from aiida_koopmans.workgraphs.block_wannierize import BlockWannierizeTask

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
def silicon_structure(aiida_profile):
    """Return a 2-atom periodic silicon ``StructureData``."""
    from aiida.orm import StructureData

    cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
    struct = StructureData(cell=cell, pbc=True)
    struct.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
    struct.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si", name="Si")
    return struct


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


@pytest.fixture
def kmesh(aiida_profile):
    """Return a coarse explicit k-mesh shared by nscf and every block."""
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints_mesh([2, 2, 2])
    return kpts


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
# Graph construction: shared scf+nscf once, one BlockWannierize per block
# ----------------------------------------------------------------------


def _build(codes, structure, blocks, kpoints):
    return BlockWannierizeTask.build(
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
        # list: one independent BlockWannierize per block (aiida-workgraph
        # auto-suffixes the repeats: BlockWannierize, BlockWannierize1, ...).
        # No Map zone.
        n_block_tasks = sum(1 for name in names if name.startswith("BlockWannierize"))
        assert n_block_tasks == n_blocks
        assert names.count("map_zone") == 0
