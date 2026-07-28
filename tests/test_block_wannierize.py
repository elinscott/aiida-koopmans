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
from tests.fixtures import assert_graph_roundtrips, automatic_block, explicit_block

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
        explicit_block("block_1", range(1, 5)),  # 4 occupied
        explicit_block("block_2", range(5, 9)),  # 4 empty
    ]


def _zno_blocks() -> list[ExplicitProjectionBlock]:
    """ZnO shape: 4 occupied blocks + 1 empty block, nspin=1."""
    return [
        explicit_block("block_1", range(1, 6)),  # Zn 3d-ish
        explicit_block("block_2", range(6, 9)),
        explicit_block("block_3", range(9, 13)),
        explicit_block("block_4", range(13, 17)),
        explicit_block("block_5", range(17, 21)),  # empty
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
        # folder keys (retrieved / remote_folder) stay unpopulated at
        # runtime on split entries.
        expected_entry = {
            "u_file",
            "hr_file",
            "centres_file",
            "retrieved",
            "remote_folder",
            "nnkp_file",
            "output_parameters",
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

    def test_parallelization_reaches_wannier_and_pw2wannier_steps(
        self, wannier_codes, silicon_structure, kmesh, nscf_scratch, fake_cutoffs_family
    ):
        """wannier90 takes metadata.options only (no flags); pw2wannier90 takes -pd."""
        block = ExplicitProjectionBlock(
            label="block_1",
            spin=SpinChannel.NONE,
            num_wann=4,
            num_bands=4,
            projection_type=WannierProjectionType.ANALYTIC,
            projections=["Si: sp3"],
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
        task = self._wannier_task(wg)

        params = task.inputs["wannier90"]["wannier90"]["parameters"].value.get_dict()
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
        params = task.inputs["wannier90"]["wannier90"]["parameters"].value.get_dict()
        assert params["auto_projections"] is True
        assert params["num_wann"] == 8
        assert params["num_bands"] == 8
        assert "exclude_bands" not in params
        assert task.inputs["wannier90"]["wannier90"]["projections"].value is None
        inputpp = task.inputs["pw2wannier90"]["pw2wannier90"]["parameters"].value.get_dict()
        assert inputpp["INPUTPP"]["atom_proj"] is True
        assert_graph_roundtrips(wg)

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
