"""Construction-level unit tests for the block-by-block Wannierize workgraph.

These build the ``WannierizeBlocks`` graph (no daemon, no real codes
execution) and introspect its task list. The per-block fan-out is a native
``for`` loop in the (top-level) graph body, which runs at build time over the
concrete ``blocks`` list -- so the built graph shows one ``WannierizeBlock``
per block plus a single shared ``scf_nscf`` task.
"""

import pytest
from aiida_quantumespresso.common.types import ElectronicType
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
from aiida_koopmans.workgraphs.block_wannierize import (
    UnconstrainedDisentanglementWarning,
    WannierizeBlock,
    WannierizeBlocks,
)
from tests.fixtures import (
    assert_graph_roundtrips,
    automatic_block,
    bands_data,
    explicit_block,
    si_external_projector_tables,
)

# ----------------------------------------------------------------------
# Fixtures: structures, block shapes (``wannier_codes`` is shared, see fixtures.py)
# ----------------------------------------------------------------------


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


def _silicon_blocks() -> list[ExplicitProjectionBlock]:
    """tutorial_2 silicon shape: 1 occupied block + 1 empty block, nspin=1."""
    return [
        explicit_block("block_1", range(1, 5), filled=True),
        explicit_block("block_2", range(5, 9), filled=False),
    ]


def _zno_blocks() -> list[ExplicitProjectionBlock]:
    """ZnO shape: 4 occupied blocks + 1 empty block, nspin=1."""
    return [
        explicit_block("block_1", range(1, 6), filled=True),  # Zn 3d-ish
        explicit_block("block_2", range(6, 9), filled=True),
        explicit_block("block_3", range(9, 13), filled=True),
        explicit_block("block_4", range(13, 17), filled=True),
        explicit_block("block_5", range(17, 21), filled=False),
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
        # list: one independent WannierizeBlock per block, labelled after the
        # block. No Map zone.
        n_block_tasks = sum(1 for name in names if name.startswith("wannierize_block"))
        assert n_block_tasks == n_blocks
        assert names.count("map_zone") == 0
        # The parsed per-block outputs are unified once, in band order.
        assert names.count("collect_wannier_functions") == 1
        output_names = [socket._name for socket in wg.outputs]
        for expected in ("blocks", "centres", "spreads", "nscf"):
            assert expected in output_names, output_names

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

    def test_shared_nscf_bands_reach_every_block(self, wannier_codes, silicon_structure, kmesh):
        """The internal nscf's eigenvalues are linked into each per-block graph.

        This default is the only way a block's frozen window is ever checked
        on the plain wannierize route: no caller there passes ``nscf_bands``
        explicitly, so a per-block test that supplies it by hand exercises
        the check and skips the wiring. Assert the link itself rather than a
        value — at build time the socket carries an unresolved output of
        ``scf_nscf``.
        """
        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )
        block_tasks = [t.name for t in wg.tasks if t.name.startswith("wannierize_block")]
        assert len(block_tasks) == 2
        for name in block_tasks:
            links = wg.tasks[name].inputs["nscf_bands"]._links
            assert [str(link) for link in links] == [
                f'TaskLink(from="scf_nscf.nscf_output_band", to="{name}.nscf_bands")'
            ]

    def test_external_scratch_leaves_the_bands_socket_to_the_caller(
        self, wannier_codes, silicon_structure, kmesh, nscf_remote
    ):
        """With no internal nscf there are no eigenvalues to default to."""
        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            nscf_remote_folder=nscf_remote,
        )
        socket = wg.tasks["wannierize_block_1"].inputs["nscf_bands"]
        assert not socket._links
        assert socket.value is None

    def test_external_scratch_skips_the_internal_scf_nscf(
        self, wannier_codes, silicon_structure, kmesh, nscf_remote
    ):
        """An ``nscf_remote_folder`` input replaces the internal ground state."""
        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            nscf_remote_folder=nscf_remote,
        )
        names = [t.name for t in wg.tasks]
        assert "scf_nscf" not in names
        assert names.count("collect_wannier_functions") == 1
        assert sum(1 for name in names if name.startswith("wannierize_block")) == 2

    def test_rejects_a_pool_below_the_uppermost_block(
        self, wannier_codes, silicon_structure, kmesh
    ):
        """Every route Wannierises through here, so the pool rule is checked here.

        A pool on anything but the channel's top block moves the blocks
        above it up the Wannier ordering while their own bookkeeping keeps
        saying where they were, so the graph builds and runs and every
        consumer of the ordering — the u/hr merge, the screening fan-out —
        addresses the wrong Wannier functions.
        """
        blocks = [
            explicit_block("block_1", range(1, 5), filled=True, num_bands=8),
            explicit_block("block_2", range(5, 9), filled=False),
        ]
        with pytest.raises(ValueError, match="Only a channel's uppermost block"):
            _build(wannier_codes, silicon_structure, blocks, kmesh)

    def test_external_scratch_rejects_scf_nscf_overrides(
        self, wannier_codes, silicon_structure, kmesh, nscf_remote
    ):
        """scf/nscf overrides would be silently ignored alongside an external scratch."""
        with pytest.raises(ValueError, match="external"):
            WannierizeBlocks.build(
                codes=wannier_codes,
                structure=silicon_structure,
                blocks=_silicon_blocks(),
                kpoints=kmesh,
                pseudo_family="SSSP/1.3/PBE/efficiency",
                nscf_remote_folder=nscf_remote,
                overrides={"scf": {"pw": {"parameters": {"SYSTEM": {"ecutwfc": 70.0}}}}},
            )


# ----------------------------------------------------------------------
# Split mode: loud validation and the unified-output gate
# ----------------------------------------------------------------------


class TestSplitMode:
    """Split-mode triggering, loud validation, and the uniform block contract."""

    def _build_split(self, codes, structure, kmesh, kpath=None, **kwargs):
        defaults: dict = {
            "codes": codes,
            "structure": structure,
            "blocks": _silicon_blocks(),
            "kpoints": kmesh,
            "mp_grid": [2, 2, 2],
            "pseudo_family": "SSSP/1.3/PBE/efficiency",
            "split_threshold": 1.5,
            "bands_kpoints": kpath,
            "num_occ_bands": 4,
        }
        defaults.update(kwargs)
        return WannierizeBlocks.build(**defaults)

    def test_split_without_bands_kpoints_raises(self, auto_codes, silicon_structure, kmesh):
        """The trigger is the threshold; the k-path is a requirement, not the trigger."""
        with pytest.raises(ValueError, match="bands_kpoints"):
            self._build_split(auto_codes, silicon_structure, kmesh, bands_kpoints=None)

    def test_disentangling_block_cannot_be_split(self, auto_codes, silicon_structure, kmesh, kpath):
        """A parent block with a pool is rejected before the split chain is built.

        The per-group re-Wannierisation reads only the parent's gauge
        products, so the parent's disentanglement matrix would be dropped
        silently. Without this guard the chain assembles and runs: the
        rejection used to fall out of the group restriction refusing bands
        that reached into the pool, and confining the block's bands to the
        Wannier manifold removed that side effect.
        """
        blocks = [explicit_block("block_1", range(1, 9), projections=["Si: sp3"], num_bands=12)]
        with pytest.raises(NotImplementedError, match="splitting a disentangled block"):
            self._build_split(
                auto_codes, silicon_structure, kmesh, kpath=kpath, blocks=blocks, num_occ_bands=4
            )

    def test_automatic_block_triggers_the_split_path(self, auto_codes, silicon_structure, kmesh):
        """An automatic-projections block alone selects split mode.

        No threshold is set, yet the build must go down the split path — and
        therefore demand its ``bands_kpoints`` requirement.
        """
        blocks = [explicit_block("block_1", range(1, 5)), automatic_block("block_2", range(5, 9))]
        with pytest.raises(ValueError, match="bands_kpoints"):
            WannierizeBlocks.build(
                codes=auto_codes,
                structure=silicon_structure,
                blocks=blocks,
                kpoints=kmesh,
                mp_grid=[2, 2, 2],
                pseudo_family="SSSP/1.3/PBE/efficiency",
                num_occ_bands=4,
            )

    def test_split_without_num_occ_bands_raises(self, auto_codes, silicon_structure, kmesh, kpath):
        """The detection always splits at the occupied/empty boundary."""
        with pytest.raises(ValueError, match="num_occ_bands"):
            self._build_split(auto_codes, silicon_structure, kmesh, kpath, num_occ_bands=None)

    def test_split_without_wannierjl_code_raises(
        self, wannier_codes, silicon_structure, kmesh, kpath
    ):
        """The detected groups are split with Wannier.jl, so its code is required."""
        with pytest.raises(ValueError, match="wannierjl"):
            self._build_split(wannier_codes, silicon_structure, kmesh, kpath)

    def test_split_with_external_scratch_raises(
        self, auto_codes, silicon_structure, kmesh, kpath, nscf_remote
    ):
        """The bands step needs the internal scf; an external nscf scratch has none."""
        with pytest.raises(ValueError, match="external"):
            self._build_split(
                auto_codes, silicon_structure, kmesh, kpath, nscf_remote_folder=nscf_remote
            )

    def test_split_without_mp_grid_raises(self, auto_codes, silicon_structure, kmesh, kpath):
        """The per-group re-wannierisation writes ``mp_grid`` into each sub-block."""
        with pytest.raises(ValueError, match="mp_grid"):
            self._build_split(auto_codes, silicon_structure, kmesh, kpath, mp_grid=None)

    def test_split_only_inputs_without_a_trigger_raise(
        self, wannier_codes, silicon_structure, kmesh, kpath
    ):
        """Split-only knobs without a trigger would be silently ignored."""
        with pytest.raises(ValueError, match="Split-only"):
            WannierizeBlocks.build(
                codes=wannier_codes,
                structure=silicon_structure,
                blocks=_silicon_blocks(),
                kpoints=kmesh,
                pseudo_family="SSSP/1.3/PBE/efficiency",
                bands_kpoints=kpath,
                num_occ_bands=4,
            )

    def test_uniform_block_contract_and_output_gating(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        """Split and plain entries expose identical sockets; unified outputs gate.

        Every entry carries a final-gauge ``output_parameters``, so the
        unified ``centres`` / ``spreads`` are collected in both modes;
        split mode additionally wires ``bands`` / ``groups``. Every
        declared socket shows up in ``wg.outputs`` either way, so
        populated-ness is read off the links.
        """
        # Every entry declares the full flat contract; the plain-route-only
        # keys (retrieved / remote_folder / wannier90_parameters) stay
        # unpopulated at runtime on split entries.
        expected_entry = {
            "u_file",
            "hr_file",
            "centres_file",
            "retrieved",
            "remote_folder",
            "nnkp_file",
            "output_parameters",
            "wannier90_parameters",
        }

        wg = self._build_split(
            auto_codes, silicon_structure, kmesh, kpath, pseudo_family=fake_cutoffs_family.label
        )
        names = [t.name for t in wg.tasks]
        assert names.count("collect_wannier_functions") == 1
        # Each entry receives its whole products namespace through a single
        # handle link (per-key links into a dynamic entry do not survive the
        # run-start round-trip), so the link sits on the entry itself.
        for label in ("block_1", "block_2"):
            entry = wg.outputs["blocks"][label]
            assert {socket._name for socket in entry} == expected_entry
            assert entry._links, label
        assert wg.outputs["nscf"]["remote_folder"]._links
        for populated in ("bands", "groups", "centres", "spreads"):
            assert wg.outputs[populated]._links, populated
        assert_graph_roundtrips(wg)

        plain = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )
        assert "collect_wannier_functions" in [t.name for t in plain.tasks]
        for label in ("block_1", "block_2"):
            entry = plain.outputs["blocks"][label]
            assert {socket._name for socket in entry} == expected_entry
            assert entry._links, label
        assert plain.outputs["centres"]._links
        assert plain.outputs["spreads"]._links
        assert not plain.outputs["bands"]._links
        assert not plain.outputs["groups"]._links
        assert_graph_roundtrips(plain)

    def test_automatic_block_split_topology(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        """A single atomic-projector block routes through the split machinery.

        No threshold is set: the automatic block is the trigger on its own,
        and the detection still runs (it always opens a group at the
        occupied/empty boundary).
        """
        block = automatic_block(
            "block_1", range(1, 9), projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE
        )
        wg = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=[block],
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kpath,
            num_occ_bands=4,
        )
        names = [t.name for t in wg.tasks]
        assert names.count("bands") == 1
        assert names.count("detect_band_groups") == 1
        assert "wannierize_split_block_1" in names
        detect_task = wg.tasks["detect_band_groups"]
        # The detection covers the block's Wannierised manifold; with no
        # threshold only the occupied/empty boundary splits it.
        assert detect_task.inputs["num_bands_total"].value == 8
        assert detect_task.inputs["threshold"].value is None
        for populated in ("bands", "groups", "centres", "spreads"):
            assert wg.outputs[populated]._links, populated
        assert_graph_roundtrips(wg)

    def test_external_block_split_topology_forwards_projector_inputs(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family, tmp_path
    ):
        """An external-projector block splits and carries its projector inputs.

        Like any automatic block it is a split trigger on its own, and the
        directory path plus orbital tables must reach the nested per-block
        graph (only the whole-block wannierisation consumes them).
        """
        block = automatic_block(
            "block_1",
            range(1, 9),
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        )
        wg = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=[block],
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kpath,
            num_occ_bands=4,
            external_projectors_path=str(tmp_path),
            external_projectors=si_external_projector_tables(),
        )
        names = [t.name for t in wg.tasks]
        assert names.count("detect_band_groups") == 1
        assert "wannierize_split_block_1" in names
        split_task = wg.tasks["wannierize_split_block_1"]
        assert split_task.inputs["external_projectors_path"].value == str(tmp_path)
        assert split_task.inputs["external_projectors"].value == si_external_projector_tables()
        assert_graph_roundtrips(wg)


# ----------------------------------------------------------------------
# Initial orbital partition emission
# ----------------------------------------------------------------------


class TestOrbitalPartitionEmission:
    """The ``orbitals`` socket: occupancy-stamp gating and both-mode wiring."""

    def test_plain_mode_emits_the_partition(self, wannier_codes, silicon_structure, kmesh):
        wg = _build(wannier_codes, silicon_structure, _silicon_blocks(), kmesh)
        names = [t.name for t in wg.tasks]
        assert names.count("initial_orbital_partition") == 1
        assert wg.outputs["orbitals"]._links
        # The task receives the reduced, JSON-pure block records in
        # input-list order.
        specs = wg.tasks["initial_orbital_partition"].inputs["blocks"].value
        assert [s["label"] for s in specs] == ["block_1", "block_2"]
        assert [s["filled"] for s in specs] == [True, False]
        assert [s["num_wann"] for s in specs] == [4, 4]
        assert_graph_roundtrips(wg)

    def test_split_mode_emits_the_partition(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        wg = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family=fake_cutoffs_family.label,
            split_threshold=1.5,
            bands_kpoints=kpath,
            num_occ_bands=4,
        )
        names = [t.name for t in wg.tasks]
        assert names.count("initial_orbital_partition") == 1
        assert wg.outputs["orbitals"]._links
        assert_graph_roundtrips(wg)

    def test_unstamped_blocks_skip_the_emission(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        """Blocks of unknown occupancy Wannierise; only the partition waits.

        The automatic-projections shape: the pseudopotentials fix how many
        Wannier functions the block has, but which of them are occupied is
        settled by the band-group detection this build only schedules.
        """
        wg = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=[automatic_block("block_1", range(1, 9))],
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kpath,
            num_occ_bands=4,
        )
        assert "initial_orbital_partition" not in [t.name for t in wg.tasks]
        assert not wg.outputs["orbitals"]._links

    def test_partially_stamped_blocks_raise(self, wannier_codes, silicon_structure, kmesh):
        blocks = _silicon_blocks()
        del blocks[1]["filled"]
        with pytest.raises(ValueError, match="block_2"):
            _build(wannier_codes, silicon_structure, blocks, kmesh)

    def test_partition_follows_the_stamps_not_the_bands(
        self, wannier_codes, silicon_structure, kmesh
    ):
        """The emitted partition splits where the stamps say, not where the bands do.

        A magnetized layout: the channels' occupied counts differ (four up,
        two down), so no single band boundary reproduces the stamps — a
        band-position reading with the up channel's boundary calls
        ``emp_dw``'s lowest bands occupied.
        """
        blocks = [
            explicit_block("occ_up", range(1, 5), spin=SpinChannel.UP, filled=True),
            explicit_block("emp_up", range(5, 7), spin=SpinChannel.UP, filled=False),
            explicit_block("occ_dw", range(1, 3), spin=SpinChannel.DOWN, filled=True),
            explicit_block("emp_dw", range(3, 7), spin=SpinChannel.DOWN, filled=False),
        ]
        wg = _build(wannier_codes, silicon_structure, blocks, kmesh)
        specs = wg.tasks["initial_orbital_partition"].inputs["blocks"].value
        assert [(s["label"], s["filled"]) for s in specs] == [
            ("occ_up", True),
            ("emp_up", False),
            ("occ_dw", True),
            ("emp_dw", False),
        ]

    def test_blocks_out_of_emitted_order_raise(self, wannier_codes, silicon_structure, kmesh):
        """Non-channel-contiguous input would mis-pair `orbitals` against `spreads`.

        The partition walks channels contiguously (up, occupied-then-empty,
        then down) while `spreads` / `centres` concatenate in input-list
        order; interleaving the channels makes the two orders diverge, so
        the build must refuse rather than let a positional consumer
        mis-align silently.
        """
        blocks = [
            explicit_block("occ_up", range(1, 3), spin=SpinChannel.UP, filled=True),
            explicit_block("occ_dw", range(1, 3), spin=SpinChannel.DOWN, filled=True),
            explicit_block("emp_up", range(3, 4), spin=SpinChannel.UP, filled=False),
            explicit_block("emp_dw", range(3, 4), spin=SpinChannel.DOWN, filled=False),
        ]
        with pytest.raises(ValueError, match="emitted orbital order"):
            _build(wannier_codes, silicon_structure, blocks, kmesh)

    def test_partition_task_runs_through_the_engine(self, aiida_profile):
        """The reduced specs and the emitted substrate survive engine storage."""
        from aiida import orm
        from aiida_workgraph import WorkGraph

        from aiida_koopmans.workgraphs.variational_orbitals import initial_orbital_partition

        wg = WorkGraph("initial_orbital_partition_unit")
        wg.add_task(
            initial_orbital_partition,
            name="partition",
            blocks=[
                {"label": "occ", "spin": SpinChannel.NONE, "filled": True, "num_wann": 2},
                {"label": "emp", "spin": SpinChannel.NONE, "filled": False, "num_wann": 1},
            ],
        )
        wg.run()
        node = wg.tasks.partition.outputs.result.value
        assert isinstance(node, orm.List)
        orbitals = node.get_list()
        assert [o["index"] for o in orbitals] == [1, 2, 3]
        assert [o["group_id"] for o in orbitals] == [1, 1, 2]
        # Storage degrades the str-enum spin to a plain str; `==` still
        # holds (`is` would not), which is the comparison consumers use.
        assert all(o["spin"] == SpinChannel.NONE for o in orbitals)


# ----------------------------------------------------------------------
# collect_wannier_functions (raw callable, no engine)
# ----------------------------------------------------------------------


def _w90_output_parameters(spreads: list[float]) -> dict:
    """Build a synthetic wannier90 ``output_parameters`` payload.

    Shape-faithful to aiida-wannier90's parser
    (``aiida_wannier90/parsers/wannier90.py``): the final-state WF table
    lands in ``wannier_functions_output`` as a per-WF list of
    ``{wf_ids, wf_centres, wf_spreads}`` dicts with 1-based ``wf_ids``,
    ``number_wfs`` comes from the MAIN table, and the manifold-total
    Omega decomposition sits in separate ``Omega_*`` scalars.
    """
    return {
        "number_wfs": len(spreads),
        "wannier_functions_output": [
            {"wf_ids": i, "wf_centres": (0.1 * i, 0.0, 0.0), "wf_spreads": spread}
            for i, spread in enumerate(spreads, start=1)
        ],
        "Omega_total": sum(spreads),
    }


class TestCollectWannierFunctions:
    """Unit tests of the unify task via its raw ``._callable``.

    The inputs are plain dicts because that is what the task body sees at
    runtime: aiida-pythonjob's built-in ``Dict`` deserializer hands
    ``orm.Dict`` inputs over as their ``get_dict()`` payload.
    """

    def test_concatenates_in_key_order(self):
        """Blocks concatenate in (zero-padded) key order = input-list order."""
        from aiida_koopmans.workgraphs.block_wannierize import collect_wannier_functions

        collected = collect_wannier_functions._callable(
            output_parameters={
                "b01": _w90_output_parameters([3.3]),
                "b00": _w90_output_parameters([1.1, 2.2]),
            }
        )
        assert collected["spreads"] == [1.1, 2.2, 3.3]
        assert collected["centres"] == [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.0, 0.0]]

    def test_entries_are_ordered_by_wf_ids(self):
        """Out-of-order ``wannier_functions_output`` entries are re-sorted."""
        from aiida_koopmans.workgraphs.block_wannierize import collect_wannier_functions

        parameters = _w90_output_parameters([1.1, 2.2, 3.3])
        parameters["wannier_functions_output"].reverse()
        collected = collect_wannier_functions._callable(output_parameters={"b00": parameters})
        assert collected["spreads"] == [1.1, 2.2, 3.3]

    def test_unparsed_centre_coordinates_stay_none(self):
        """The upstream parser None-pads unreadable coordinates; keep them."""
        from aiida_koopmans.workgraphs.block_wannierize import collect_wannier_functions

        parameters = _w90_output_parameters([1.1])
        parameters["wannier_functions_output"][0]["wf_centres"] = (0.5, None, 0.7)
        collected = collect_wannier_functions._callable(output_parameters={"b00": parameters})
        assert collected["centres"] == [[0.5, None, 0.7]]
        assert collected["spreads"] == [1.1]

    def test_missing_wf_table_raises(self):
        """Parameters without ``wannier_functions_output`` (e.g. a -pp run) raise."""
        from aiida_koopmans.workgraphs.block_wannierize import collect_wannier_functions

        with pytest.raises(ValueError, match="lists 0 final-state Wannier functions"):
            collect_wannier_functions._callable(output_parameters={"b00": {"number_wfs": 2}})

    def test_wf_count_mismatch_raises(self):
        """A WF table shorter than the declared ``number_wfs`` is rejected."""
        from aiida_koopmans.workgraphs.block_wannierize import collect_wannier_functions

        parameters = _w90_output_parameters([1.1])
        parameters["number_wfs"] = 2
        with pytest.raises(ValueError, match="declares number_wfs = 2"):
            collect_wannier_functions._callable(output_parameters={"b00": parameters})

    def test_entries_without_spreads_raise(self):
        """Restart-for-plotting entries (only wf_ids + im_re_ratio) are rejected."""
        from aiida_koopmans.workgraphs.block_wannierize import collect_wannier_functions

        parameters = {
            "number_wfs": 1,
            "wannier_functions_output": [{"wf_ids": 1, "im_re_ratio": 1.0}],
        }
        with pytest.raises(ValueError, match="no ``wf_spreads``"):
            collect_wannier_functions._callable(output_parameters={"b00": parameters})


# ----------------------------------------------------------------------
# extract_wannier_output_files (raw callable, no engine)
# ----------------------------------------------------------------------


class TestExtractWannierProducts:
    """The gauge-product trio is read back off a wannier90 retrieved folder."""

    @staticmethod
    def _folder(names):
        from aiida.orm import FolderData

        folder = FolderData()
        for name in names:
            folder.base.repository.put_object_from_bytes(f"<{name}>".encode(), name)
        return folder.store()

    def test_extracts_the_trio(self, aiida_profile):
        from aiida_koopmans.workgraphs.block_wannierize import extract_wannier_output_files

        folder = self._folder(["aiida_u.mat", "aiida_hr.dat", "aiida_centres.xyz"])
        products = extract_wannier_output_files._callable(retrieved=folder)
        assert products["u_file"].filename == "aiida_u.mat"
        assert products["u_file"].get_content() == "<aiida_u.mat>"
        assert products["hr_file"].get_content() == "<aiida_hr.dat>"
        assert products["centres_file"].get_content() == "<aiida_centres.xyz>"

    def test_missing_file_raises(self, aiida_profile):
        from aiida_koopmans.workgraphs.block_wannierize import extract_wannier_output_files

        folder = self._folder(["aiida_u.mat"])
        with pytest.raises(ValueError, match=r"aiida_hr\.dat"):
            extract_wannier_output_files._callable(retrieved=folder)


# ----------------------------------------------------------------------
# Eager per-block build: the flat WannierizeOverrides -> builder translation
# ----------------------------------------------------------------------


#: Two k-points, six bands, for a block that Wannierises the lowest four:
#: the fifth band drops from 8.0 to 7.5 across the pair, so a window that
#: freezes it is over the limit at both k-points and the ceiling comes from
#: the second.
_POOL_BANDS = [[0.0, 1.0, 2.0, 3.0, 8.0, 9.0], [0.5, 1.5, 2.5, 3.5, 7.5, 9.5]]


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
        nscf_bands=None,
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
            nscf_bands=nscf_bands,
        )

    @staticmethod
    def _wannier_task(wg):
        matches = [
            t for t in wg.tasks if "annier90" in t.name and t.name != "emit_wannier90_parameters"
        ]
        assert matches, f"no wannier90 task among {[t.name for t in wg.tasks]}"
        return matches[0]

    @staticmethod
    def _w90_parameters(wg):
        """Return the merged parameters the emit task feeds the wannier90 step."""
        value = wg.tasks["emit_wannier90_parameters"].inputs["parameters"].value
        return value.get_dict() if hasattr(value, "get_dict") else dict(value)

    def test_parallelization_reaches_wannier_and_pw2wannier_steps(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """wannier90 takes metadata.options only (no flags); pw2wannier90 takes -pd."""
        block = explicit_block(
            "block_1", range(1, 5), projections=["Si: sp3"], filled=True, num_bands=4
        )
        wg = WannierizeBlock.build(
            codes=wannier_codes,
            structure=silicon_structure,
            block=block,
            projection_type=WannierProjectionType.ANALYTIC,
            nscf_remote_folder=nscf_scratch,
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family=fake_cutoffs_family.label,
            parallelization={
                "wannier90": {"ntasks": 4},
                "pw2wannier90": {"ntasks": 2, "pd": True},
            },
        )
        task = self._wannier_task(wg)

        w90 = task.inputs["wannier90"]["wannier90"]
        assert w90["metadata"]["options"]["resources"].value["num_mpiprocs_per_machine"] == 4
        # wannier90 has no pool/pd concept, so no cmdline flag is injected.
        w90_settings = w90["settings"].value
        assert w90_settings is None or "cmdline" not in w90_settings
        p2w = task.inputs["pw2wannier90"]["pw2wannier90"]
        assert p2w["metadata"]["options"]["resources"].value["num_mpiprocs_per_machine"] == 2
        assert p2w["settings"].value["cmdline"] == ["-pd", "true"]

    def test_flat_overrides_reach_the_builder_namespaces(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """`wannier90` / `pw2wannier90` land in `parameters` / `INPUTPP`; `scf` is ignored."""
        block = explicit_block(
            "block_1", range(1, 5), projections=["Si: sp3"], filled=True, num_bands=6
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

        params = self._w90_parameters(wg)
        assert params["dis_froz_max"] == 10.6
        # A disentangling block keeps a user-supplied iteration budget instead
        # of the 5000-iteration default.
        assert params["dis_num_iter"] == 200
        assert params["num_wann"] == 4
        assert params["num_bands"] == 6
        assert params["write_hr"] is True
        assert params["write_u_matrices"] is True
        assert params["write_xyz"] is True
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

        # Only aiida.chk is force-retrieved; the U matrices, centres and hr file
        # ride upstream's default retrieve suffixes once the write_* pins above
        # cause them to be written.
        settings = task.inputs["wannier90"]["wannier90"]["settings"].value.get_dict()
        assert settings["additional_retrieve_list"] == ["aiida.chk"]

        projections = task.inputs["wannier90"]["wannier90"]["projections"].value
        assert list(projections) == ["Si: sp3"]

    def test_parameters_ride_an_explicit_socket(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """One emitted Dict both feeds the wannier90 step and exits the graph."""
        block = explicit_block("block_1", range(1, 5))
        wg = self._build_block(
            wannier_codes, silicon_structure, kmesh, nscf_scratch, block, fake_cutoffs_family.label
        )
        task = self._wannier_task(wg)
        links = task.inputs["wannier90"]["wannier90"]["parameters"]._links
        assert len(links) == 1
        assert links[0].from_task.name == "emit_wannier90_parameters"
        assert wg.outputs["wannier90_parameters"]._links
        assert_graph_roundtrips(wg)

    def test_no_overrides_defaults(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """num_bands == num_wann strips the windows; mp_grid=None drops the key."""
        block = explicit_block("block_1", range(1, 5))
        block["exclude_bands"] = [9, 10]
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
        )
        params = self._w90_parameters(wg)
        assert params["num_wann"] == 4
        assert params["num_bands"] == 4
        assert params["exclude_bands"] == [9, 10]
        for key in ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max"):
            assert key not in params
        assert params["write_hr"] is True
        assert params["write_u_matrices"] is True
        assert params["write_xyz"] is True
        assert "mp_grid" not in params

    def test_automatic_block_relies_on_the_projection_type(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """An atomic-projector block sets ``auto_projections``, no orbital list.

        The block's counts stay authoritative: no protocol-seeded
        ``exclude_bands`` survives, and the pw2wannier90 namelist carries the
        matching ``atom_proj`` so the amn width equals ``num_wann``.
        """
        block = automatic_block(
            "block_1", range(1, 9), projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE
        )
        wg = WannierizeBlock.build(
            codes=wannier_codes,
            structure=silicon_structure,
            block=block,
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
            nscf_remote_folder=nscf_scratch,
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family=fake_cutoffs_family.label,
        )
        task = self._wannier_task(wg)
        params = self._w90_parameters(wg)
        assert params["auto_projections"] is True
        assert params["num_wann"] == 8
        assert params["num_bands"] == 8
        assert "exclude_bands" not in params
        assert task.inputs["wannier90"]["wannier90"]["projections"].value is None
        inputpp = task.inputs["pw2wannier90"]["pw2wannier90"]["parameters"].value.get_dict()
        assert inputpp["INPUTPP"]["atom_proj"] is True
        assert_graph_roundtrips(wg)

    def test_external_block_stages_the_projector_inputs(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family, tmp_path
    ):
        """An external-projector block stages the directory and the namelist.

        The pw2wannier90 step must carry the projector-directory
        ``RemoteData`` plus the kind→element file map, and its namelist the
        ``atom_proj_ext`` switches pointing at the staged copy; the ``.win``
        side is plain ``auto_projections`` with the block's counts.
        """
        block = automatic_block(
            "block_1",
            range(1, 9),
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        )
        wg = WannierizeBlock.build(
            codes=wannier_codes,
            structure=silicon_structure,
            block=block,
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
            nscf_remote_folder=nscf_scratch,
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family=fake_cutoffs_family.label,
            external_projectors_path=str(tmp_path),
            external_projectors=si_external_projector_tables(),
        )
        task = self._wannier_task(wg)
        params = self._w90_parameters(wg)
        assert params["auto_projections"] is True
        assert params["num_wann"] == 8
        assert params["num_bands"] == 8
        p2w = task.inputs["pw2wannier90"]["pw2wannier90"]
        inputpp = p2w["parameters"].value.get_dict()["INPUTPP"]
        assert inputpp["atom_proj"] is True
        assert inputpp["atom_proj_ext"] is True
        assert inputpp["atom_proj_dir"] == "external_projectors/"
        assert p2w["external_projectors_path"].value.get_remote_path() == str(tmp_path)
        assert p2w["external_projectors_list"].value.get_dict() == {"Si": "Si"}
        # No entry triggers upstream's frozen-list selection, so the
        # namelist never carries atom_proj_frozen.
        assert "atom_proj_frozen" not in inputpp
        assert_graph_roundtrips(wg)

    def test_external_block_without_projector_inputs_raises(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, tmp_path
    ):
        """An external block missing either projector input fails naming it."""
        block = automatic_block(
            "block_1",
            range(1, 9),
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        )
        with pytest.raises(ValueError, match=r"missing: \['external_projectors'\]"):
            WannierizeBlock.build(
                codes=wannier_codes,
                structure=silicon_structure,
                block=block,
                projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
                nscf_remote_folder=nscf_scratch,
                kpoints=kmesh,
                external_projectors_path=str(tmp_path),
            )

    def test_semicore_exclusions_stay_out(
        self,
        monkeypatch,
        wannier_codes,
        silicon_structure,
        kmesh,
        nscf_scratch,
        fake_cutoffs_family,
    ):
        """A curated semicore table must not leak into the block's inputs.

        With semicore handling active, the upstream protocol would exclude
        the semicore bands and shrink the pw2wannier90 atomic-projector set
        (``atom_proj_exclude``) — both behind the block bookkeeping's back.
        The bundled tables match pseudos by md5, so a curated entry is
        simulated by patching the lookup.
        """
        import aiida_wannier90_workflows.utils.pseudo as upstream_pseudo

        monkeypatch.setattr(
            upstream_pseudo,
            "get_pseudo_orbitals",
            lambda *args, **kwargs: {"Si": {"pswfcs": ["3S", "3P"], "semicores": ["3S"]}},
        )
        block = automatic_block(
            "block_1", range(1, 9), projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE
        )
        wg = WannierizeBlock.build(
            codes=wannier_codes,
            structure=silicon_structure,
            block=block,
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
            nscf_remote_folder=nscf_scratch,
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            pseudo_family=fake_cutoffs_family.label,
        )
        task = self._wannier_task(wg)
        params = self._w90_parameters(wg)
        assert "exclude_bands" not in params
        assert params["num_wann"] == 8
        inputpp = task.inputs["pw2wannier90"]["pw2wannier90"]["parameters"].value.get_dict()
        assert "atom_proj_exclude" not in inputpp["INPUTPP"]

    def test_no_exclusion_block_drops_global_exclude_bands(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """A global ``exclude_bands`` override must not stick to a no-exclusion block.

        The flat ``wannier90`` overrides apply to every block, but the block
        bookkeeping is the exclusion authority: a block that excludes
        nothing drops the seeded key.
        """
        block = explicit_block("block_1", range(1, 5))
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
            overrides={"wannier90": {"exclude_bands": [9, 10]}},
        )
        params = self._w90_parameters(wg)
        assert "exclude_bands" not in params

    def test_disentangling_block_gets_the_default_iteration_budget(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """num_bands > num_wann without a user dis_num_iter unfreezes the subspace."""
        block = explicit_block(
            "block_1", range(1, 5), projections=["Si: sp3"], filled=True, num_bands=6
        )
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
        )
        params = self._w90_parameters(wg)
        assert params["dis_num_iter"] == 5000

    def test_frozen_window_freezing_too_many_bands_is_rejected(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """A window over the limit names the block, the count and the ceiling.

        wannier90 would stop mid-run complaining about the window alone. The
        block Wannierises four bands, and ``dis_froz_max = 8.0`` freezes five
        at both k-points, so the message has to carry enough for the user to
        pick a new value without reading any source: which block, how many
        bands, where, and the largest value that fits (7.5 eV, the fifth band
        at the second k-point).
        """
        block = explicit_block("block_1", range(1, 5), projections=["Si: sp3"], num_bands=6)
        with pytest.raises(ValueError) as excinfo:
            self._build_block(
                wannier_codes,
                silicon_structure,
                kmesh,
                nscf_scratch,
                block,
                fake_cutoffs_family.label,
                overrides={"wannier90": {"dis_froz_max": 8.0}},
                nscf_bands=bands_data(_POOL_BANDS),
            )
        message = str(excinfo.value)
        assert "block_1" in message
        assert "num_wann = 4" in message
        assert "dis_froz_max = 8.0" in message
        assert "freezes 5" in message
        assert "k-point 1 of 2" in message
        assert "7.500000" in message

    def test_frozen_window_that_fits_is_left_untouched(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """A window within the limit reaches wannier90 exactly as written.

        The user picks ``dis_froz_max`` against the band structure, so a
        value that fits is theirs to keep: nothing here may adjust it, and
        the base workchain must not be handed the eigenvalues either, since
        given them it would lower the window on its own.
        """
        block = explicit_block("block_1", range(1, 5), projections=["Si: sp3"], num_bands=6)
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
            overrides={"wannier90": {"dis_froz_max": 4.0}},
            nscf_bands=bands_data(_POOL_BANDS),
        )
        assert self._w90_parameters(wg)["dis_froz_max"] == 4.0
        w90_inputs = wg.tasks["wannier90"].inputs["wannier90"]
        assert w90_inputs["bands"].value is None
        assert w90_inputs["current_spin"].value is None

    def test_frozen_window_is_checked_against_the_blocks_own_spin_channel(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """A collinear nscf's two channels are judged separately.

        The eigenvalues arrive as one array per channel. Here the same
        window fits the up channel and freezes a fifth band in the down one,
        so reading the wrong half would either miss a real failure or invent
        one.
        """
        bands = bands_data(
            [
                [[0.0, 1.0, 2.0, 3.0, 8.0, 9.0]],
                [[0.0, 1.0, 2.0, 3.0, 3.2, 9.0]],
            ]
        )

        def _build(spin):
            return self._build_block(
                wannier_codes,
                silicon_structure,
                kmesh,
                nscf_scratch,
                explicit_block(
                    f"occ_{spin.value}_1",
                    range(1, 5),
                    projections=["Si: sp3"],
                    spin=spin,
                    num_bands=6,
                ),
                fake_cutoffs_family.label,
                overrides={"wannier90": {"dis_froz_max": 4.0}},
                nscf_bands=bands,
            )

        _build(SpinChannel.UP)
        with pytest.raises(ValueError, match=r"occ_down_1.*freezes 5.*3\.200000"):
            _build(SpinChannel.DOWN)

    def test_spin_resolved_bands_on_an_unstamped_block_raise(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """Neither channel is a defensible guess, so the pairing is refused."""
        block = explicit_block("block_1", range(1, 5), projections=["Si: sp3"], num_bands=6)
        bands = bands_data([[[0.0, 1.0, 2.0, 3.0, 8.0, 9.0]], [[0.1, 1.1, 2.1, 3.1, 8.1, 9.1]]])
        with pytest.raises(ValueError, match="names no spin channel"):
            self._build_block(
                wannier_codes,
                silicon_structure,
                kmesh,
                nscf_scratch,
                block,
                fake_cutoffs_family.label,
                overrides={"wannier90": {"dis_froz_max": 4.0}},
                nscf_bands=bands,
            )

    def test_frozen_window_counts_only_the_bands_the_block_reads(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """Excluded bands are outside the count, however low they sit.

        This block excludes the two lowest bands and Wannierises two of the
        four it reads. Counting the excluded pair as frozen would put the
        window at four bands and reject a value that wannier90 accepts.
        """
        block = explicit_block(
            "block_1",
            range(3, 5),
            projections=["Si: sp3"],
            num_bands=4,
            exclude_bands=[1, 2],
        )
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
            overrides={"wannier90": {"dis_froz_max": 6.5}},
            nscf_bands=bands_data([[0.0, 1.0, 5.0, 6.0, 7.0, 20.0]]),
        )
        assert self._w90_parameters(wg)["dis_froz_max"] == 6.5

    def test_non_disentangling_block_has_its_window_stripped_unchecked(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """num_bands == num_wann cannot disentangle, so no window survives to check.

        The window here would freeze four bands for two Wannier functions.
        It is dropped rather than rejected: a globally supplied
        ``dis_froz_max`` is meant for whichever block disentangles, and
        reaching a block that cannot is not a user error.
        """
        block = explicit_block("block_1", range(1, 3), projections=["Si: sp3"])
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
            overrides={"wannier90": {"dis_froz_max": 4.0}},
            nscf_bands=bands_data([[0.0, 1.0, 2.0, 3.0]]),
        )
        assert "dis_froz_max" not in self._w90_parameters(wg)

    def test_pool_block_accounts_for_every_nscf_band(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """``len(exclude_bands) + num_bands`` reproduces the pw.x band count.

        wann2kcp.x reads the ``.chk`` against that count and refuses a
        mismatch, and the same two keys say which bands this block reads —
        so a block with spare bands must still account for every nscf band,
        just inside ``num_bands`` rather than inside the exclusions.
        """
        nbnd = 6
        block = explicit_block(
            "block_1",
            range(3, 5),
            projections=["Si: sp3"],
            num_bands=4,
            exclude_bands=[1, 2],
        )
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
            nscf_bands=bands_data([[0.0, 1.0, 2.0, 3.0, 8.0, 9.0]]),
        )
        params = self._w90_parameters(wg)
        assert params["num_wann"] == 2
        assert params["num_bands"] == 4
        assert params["exclude_bands"] == [1, 2]
        assert len(params["exclude_bands"]) + params["num_bands"] == nbnd

    def test_disentangling_block_without_a_window_still_warns(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """Wiring the bands does not itself count as a disentanglement constraint.

        A block with no window at all is a different case from one whose
        window is too high: there is nothing to reject, but the manifold is
        still chosen by free spread minimization, so the warning must keep
        firing.
        """
        block = explicit_block("block_1", range(1, 5), projections=["Si: sp3"], num_bands=6)
        with pytest.warns(UnconstrainedDisentanglementWarning, match="block_1"):
            self._build_block(
                wannier_codes,
                silicon_structure,
                kmesh,
                nscf_scratch,
                block,
                fake_cutoffs_family.label,
                nscf_bands=bands_data([[0.0, 1.0, 2.0, 3.0, 8.0, 9.0]]),
            )

    def test_gauge_products_are_extracted(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """The block graph runs the extract step and wires the uniform trio."""
        block = explicit_block("block_1", range(1, 5))
        wg = self._build_block(
            wannier_codes,
            silicon_structure,
            kmesh,
            nscf_scratch,
            block,
            fake_cutoffs_family.label,
        )
        assert "extract_wannier_output_files" in [t.name for t in wg.tasks]
        # The plain block populates the full contract, optional keys included.
        for name in (
            "u_file",
            "hr_file",
            "centres_file",
            "nnkp_file",
            "retrieved",
            "remote_folder",
            "output_parameters",
        ):
            assert wg.outputs[name]._links, name
        assert_graph_roundtrips(wg)


@pytest.mark.parametrize(
    "projection_type",
    [WannierProjectionType.SCDM, WannierProjectionType.RANDOM],
)
def test_unsupported_projection_types_raise(
    wannier_codes, silicon_structure, kmesh, projection_type
):
    """Only explicit projections and pseudoatomic projectors are supported.

    SCDM and random starting guesses are out of scope, so both fail loudly
    before any graph is built.
    """
    block = automatic_block("block_1", range(1, 9), projection_type=projection_type)
    with pytest.raises(ValueError, match=f"'{projection_type.value}' is not supported"):
        WannierizeBlocks.build(
            codes=wannier_codes, structure=silicon_structure, blocks=[block], kpoints=kmesh
        )


def test_projector_inputs_without_external_block_raise(
    wannier_codes, silicon_structure, kmesh, tmp_path
):
    """External projector inputs without an external block are rejected.

    No other projection source consumes them, so accepting them here would
    silently ignore them.
    """
    with pytest.raises(ValueError, match="without any 'atomic_projectors_external' block"):
        WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            external_projectors_path=str(tmp_path),
            external_projectors=si_external_projector_tables(),
        )


def test_second_external_block_raises(auto_codes, silicon_structure, kmesh, kpath, tmp_path):
    """Two external blocks per call are rejected naming the limitation.

    Each block would receive the full orbital tables — they are not split
    per block — so a second external block would wannierize the whole
    projector manifold instead of its own bands.
    """
    blocks = [
        automatic_block(
            "block_1",
            range(1, 5),
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        ),
        automatic_block(
            "block_2",
            range(5, 9),
            projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        ),
    ]
    with pytest.raises(ValueError, match="only one is supported per call"):
        WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=blocks,
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            bands_kpoints=kpath,
            num_occ_bands=4,
            external_projectors_path=str(tmp_path),
            external_projectors=si_external_projector_tables(),
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"external_projectors": {}}, "`external_projectors` is empty"),
        ({"external_projectors_path": "  "}, "`external_projectors_path` is blank"),
    ],
)
def test_degenerate_projector_inputs_raise(
    auto_codes, silicon_structure, kmesh, kpath, tmp_path, overrides, match
):
    """An empty table dict or a blank path fails before any graph is built."""
    block = automatic_block(
        "block_1",
        range(1, 9),
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
    )
    kwargs = {
        "external_projectors_path": str(tmp_path),
        "external_projectors": si_external_projector_tables(),
        **overrides,
    }
    with pytest.raises(ValueError, match=match):
        WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=[block],
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            bands_kpoints=kpath,
            num_occ_bands=4,
            **kwargs,
        )


def test_unknown_parallelization_code_raises(wannier_codes, silicon_structure, kmesh):
    """A typo'd parallelization code name fails loudly at build time."""
    with pytest.raises(ValueError, match="unknown parallelization code name"):
        WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            parallelization={"pww": {"npool": 2}},
        )


def test_bands_seed_does_not_mutate_the_nscf_override(
    auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
):
    """The bands step's calculation stamp must not leak into the caller's dict.

    The bands step is seeded from the nscf override; stamping its own
    calculation type through shared nested dicts previously rewrote the
    captured nscf CONTROL to 'bands', which the nscf step's own enforcement
    then rejected at run start.
    """
    from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks

    nscf_override = {"pw": {"parameters": {"CONTROL": {"tstress": True}, "SYSTEM": {"nbnd": 12}}}}
    overrides = {"nscf": nscf_override}
    wg = WannierizeBlocks.build(
        codes=auto_codes,
        structure=silicon_structure,
        blocks=_silicon_blocks(),
        kpoints=kmesh,
        mp_grid=[2, 2, 2],
        bands_kpoints=kpath,
        num_occ_bands=4,
        split_threshold=2.0,
        pseudo_family=fake_cutoffs_family.label,
        overrides=overrides,
    )
    # The caller's dict is untouched...
    assert "calculation" not in nscf_override["pw"]["parameters"]["CONTROL"]
    # ...and the captured nscf override of the scf_nscf task carries no
    # bands leak (the deferred nscf enforcement would raise on it).
    captured = wg.tasks.scf_nscf.inputs.overrides.value
    control = captured["nscf"]["pw"]["parameters"]["CONTROL"]
    assert control.get("calculation") != "bands"


class TestUnconstrainedDisentanglementWarning:
    """The pre-dispatch warning fires at parent-graph build, where users see it.

    The nested per-block body only runs daemon-side, so these assert around
    ``WannierizeBlocks.build`` — the path the dispatcher takes.
    """

    @staticmethod
    def _pool_block():
        """Build a block whose 4 Wannier functions sit under 6 read bands."""
        return explicit_block(
            "block_1", range(1, 5), projections=["Si: sp3"], filled=True, num_bands=6
        )

    def _build(self, codes, structure, blocks, kpoints, **kwargs):
        return WannierizeBlocks.build(
            codes=codes,
            structure=structure,
            blocks=blocks,
            kpoints=kpoints,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            **kwargs,
        )

    def test_pool_carrying_block_warns_at_build(self, wannier_codes, silicon_structure, kmesh):
        """num_bands > num_wann without any window/frozen key warns at build time."""
        with pytest.warns(
            UnconstrainedDisentanglementWarning,
            match=r"Block 'block_1' includes num_bands = 6 .* num_wann = 4",
        ):
            self._build(wannier_codes, silicon_structure, [self._pool_block()], kmesh)

    def test_supplied_window_silences_the_warning(
        self, wannier_codes, silicon_structure, kmesh, recwarn
    ):
        """A dis_froz_max in the flat wannier90 overrides counts as a constraint."""
        self._build(
            wannier_codes,
            silicon_structure,
            [self._pool_block()],
            kmesh,
            overrides={"wannier90": {"dis_froz_max": 10.6}},
        )
        assert not [
            w for w in recwarn if issubclass(w.category, UnconstrainedDisentanglementWarning)
        ]

    def test_non_disentangling_block_does_not_warn(
        self, wannier_codes, silicon_structure, kmesh, recwarn
    ):
        """num_bands == num_wann cannot disentangle: no warning."""
        self._build(wannier_codes, silicon_structure, _silicon_blocks(), kmesh)
        assert not [
            w for w in recwarn if issubclass(w.category, UnconstrainedDisentanglementWarning)
        ]

    def test_metal_protocol_window_silences_the_warning(
        self, wannier_codes, silicon_structure, kmesh, recwarn
    ):
        """METAL resolves the ENERGY_FIXED frozen type, whose protocol sets dis_froz_max."""
        self._build(
            wannier_codes,
            silicon_structure,
            [self._pool_block()],
            kmesh,
            electronic_type=ElectronicType.METAL,
        )
        assert not [
            w for w in recwarn if issubclass(w.category, UnconstrainedDisentanglementWarning)
        ]
