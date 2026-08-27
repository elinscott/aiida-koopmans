"""Construction-level tests for the Koopmans DFPT workgraphs.

Build the ``RunDFPT`` and ``SinglepointDFPTWorkflow`` graphs (no daemon, no
real code execution) and introspect their task lists / wiring, mirroring the
style of ``test_block_wannierize.py``. Also unit-tests the
``prepare_kcw_wannier_files`` calcfunction via its raw ``._callable``.
"""

from __future__ import annotations

import pytest
from wannier90_input.models.parameters import Projection

from aiida_koopmans.projections import ExplicitProjectionBlock, get_wannier_indices
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.workgraphs.dfpt import (
    RunDFPT,
    SinglepointDFPTWorkflow,
    prepare_kcw_wannier_files,
)
from tests.fixtures import assert_graph_roundtrips, explicit_block

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def bands_path(aiida_profile):
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints([[0.0, 0.0, 0.0], [0.25, 0.0, 0.25], [0.5, 0.0, 0.5]])
    return kpts


@pytest.fixture
def occ_retrieved(aiida_profile):
    return _retrieved_folder(("aiida_u.mat", "aiida_hr.dat", "aiida_centres.xyz"))


@pytest.fixture
def emp_retrieved(aiida_profile):
    return _retrieved_folder(
        ("aiida_u.mat", "aiida_u_dis.mat", "aiida_hr.dat", "aiida_centres.xyz")
    )


def _retrieved_folder(names):
    from aiida.orm import FolderData

    folder = FolderData()
    for name in names:
        folder.base.repository.put_object_from_bytes(f"contents of {name}".encode(), name)
    return folder.store()


def _wannier_block_folder(num_wann: int, num_bands: int, u_dis: bool = False):
    """Build a stored FolderData mimicking one block's wannier90 ``retrieved``.

    The files hold synthetic but *parseable* Wannier90 products (the merge
    path re-reads them), sharing one R-vector list / k-point set across all
    blocks so they are mergeable. ``u_dis=True`` adds a ``num_wann x
    num_bands`` disentanglement matrix.
    """
    import numpy as np
    from aiida.orm import FolderData

    from aiida_koopmans.workgraphs.utils.wannier_merge import (
        generate_wannier_centres_file_contents,
        generate_wannier_hr_file_contents,
        generate_wannier_u_file_contents,
    )

    rng = np.random.default_rng(100 * num_wann + num_bands)
    rvect = np.array([[0, 0, 0], [1, 0, 0]])
    weights = [1, 1]
    kpts = np.array([[0.0, 0.0, 0.0]])
    ham = rng.random((2, num_wann, num_wann)) + 1j * rng.random((2, num_wann, num_wann))
    umat = rng.random((1, num_wann, num_wann)) + 1j * rng.random((1, num_wann, num_wann))
    centres = [[float(i), 0.0, 0.0] for i in range(num_wann)]
    atom_lines = [
        "Si       0.00000000      0.00000000      0.00000000",
        "Si       1.35750000      1.35750000      1.35750000",
    ]

    folder = FolderData()
    put = folder.base.repository.put_object_from_bytes
    put(generate_wannier_hr_file_contents(ham, rvect, weights).encode(), "aiida_hr.dat")
    put(generate_wannier_u_file_contents(umat, kpts).encode(), "aiida_u.mat")
    put(generate_wannier_centres_file_contents(centres, atom_lines).encode(), "aiida_centres.xyz")
    if u_dis:
        udis = rng.random((1, num_wann, num_bands)) + 1j * rng.random((1, num_wann, num_bands))
        put(generate_wannier_u_file_contents(udis, kpts).encode(), "aiida_u_dis.mat")
    return folder.store()


def _block(label: str, include: range) -> ExplicitProjectionBlock:
    return explicit_block(label, include, projections=["Si:sp3"])


# ----------------------------------------------------------------------
# prepare_kcw_wannier_files (raw callable, no engine)
# ----------------------------------------------------------------------


class TestPrepareKcwWannierFiles:
    def test_occ_only(self, aiida_profile, occ_retrieved):
        outputs = prepare_kcw_wannier_files._callable(occ_b00=occ_retrieved)
        names = sorted(outputs["wannier_files"].base.repository.list_object_names())
        assert names == ["aiida_centres.xyz", "aiida_hr.dat", "aiida_u.mat"]

    def test_emp_files_are_renamed(self, aiida_profile, occ_retrieved, emp_retrieved):
        outputs = prepare_kcw_wannier_files._callable(occ_b00=occ_retrieved, emp_b00=emp_retrieved)
        merged = outputs["wannier_files"]
        names = sorted(merged.base.repository.list_object_names())
        assert names == [
            "aiida_centres.xyz",
            "aiida_emp_centres.xyz",
            "aiida_emp_hr.dat",
            "aiida_emp_u.mat",
            "aiida_emp_u_dis.mat",
            "aiida_hr.dat",
            "aiida_u.mat",
        ]
        # Contents come from the right manifold despite the rename.
        content = merged.base.repository.get_object_content("aiida_emp_u.mat", mode="rb")
        assert content == b"contents of aiida_u.mat"

    def test_missing_required_file_raises(self, aiida_profile, emp_retrieved):
        from aiida.orm import FolderData

        incomplete = FolderData()
        incomplete.base.repository.put_object_from_bytes(b"x", "aiida_hr.dat")
        incomplete.store()
        with pytest.raises(ValueError, match="write_u_matrices"):
            prepare_kcw_wannier_files._callable(occ_b00=incomplete)

    def test_no_occupied_folder_raises(self, aiida_profile, emp_retrieved):
        with pytest.raises(ValueError, match="at least one occupied"):
            prepare_kcw_wannier_files._callable(emp_b00=emp_retrieved)


class TestPrepareKcwWannierFilesMultiBlock:
    """Multi-block manifolds are merged before staging (see test_wannier_merge)."""

    def test_occ_blocks_are_merged(self, aiida_profile):
        from aiida_koopmans.workgraphs.utils.wannier_merge import (
            parse_wannier_centres_file_contents,
            parse_wannier_hr_file_contents,
            parse_wannier_u_file_shape,
        )

        outputs = prepare_kcw_wannier_files._callable(
            occ_b00=_wannier_block_folder(num_wann=2, num_bands=2),
            occ_b01=_wannier_block_folder(num_wann=3, num_bands=3),
        )
        merged = outputs["wannier_files"]
        assert sorted(merged.base.repository.list_object_names()) == [
            "aiida_centres.xyz",
            "aiida_hr.dat",
            "aiida_u.mat",
        ]
        ham, _, _ = parse_wannier_hr_file_contents(
            merged.base.repository.get_object_content("aiida_hr.dat")
        )
        assert ham.shape[1:] == (5, 5)
        assert parse_wannier_u_file_shape(
            merged.base.repository.get_object_content("aiida_u.mat")
        ) == (1, 5, 5)
        centres, atom_lines = parse_wannier_centres_file_contents(
            merged.base.repository.get_object_content("aiida_centres.xyz")
        )
        assert len(centres) == 5
        assert len(atom_lines) == 2

    def test_disentangled_emp_blocks_extend_u_dis(self, aiida_profile):
        from aiida_koopmans.workgraphs.utils.wannier_merge import parse_wannier_u_file_shape

        # Empty manifold: 2 + 2 Wannier functions over 6 empty bands; only
        # the last block is disentangled (u_dis 2 x 4).
        outputs = prepare_kcw_wannier_files._callable(
            nbnd_emp=6,
            occ_b00=_wannier_block_folder(num_wann=2, num_bands=2),
            emp_b00=_wannier_block_folder(num_wann=2, num_bands=2),
            emp_b01=_wannier_block_folder(num_wann=2, num_bands=4, u_dis=True),
        )
        merged = outputs["wannier_files"]
        assert parse_wannier_u_file_shape(
            merged.base.repository.get_object_content("aiida_emp_u_dis.mat")
        ) == (1, 4, 6)

    def test_disentangled_emp_blocks_without_u_dis_raise(self, aiida_profile):
        with pytest.raises(ValueError, match="u_dis"):
            prepare_kcw_wannier_files._callable(
                nbnd_emp=6,
                occ_b00=_wannier_block_folder(num_wann=2, num_bands=2),
                emp_b00=_wannier_block_folder(num_wann=2, num_bands=2),
                emp_b01=_wannier_block_folder(num_wann=2, num_bands=4, u_dis=False),
            )


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


class TestKoopmansDFPTTaskBuild:
    def test_full_chain_with_screening_and_bands(
        self, dfpt_codes, nscf_remote, occ_retrieved, emp_retrieved, bands_path
    ):
        wg = RunDFPT.build(
            kcw_code=dfpt_codes["kcw"],
            nscf_remote_folder=nscf_remote,
            block_wannier={
                "occ": {"retrieved": occ_retrieved},
                "emp": {"retrieved": emp_retrieved},
            },
            occ_labels=["occ"],
            emp_labels=["emp"],
            num_wann_occ=4,
            num_wann_emp=4,
            kgrid=[2, 2, 2],
            bands_kpoints=bands_path,
            eps_inf=5.3,
            has_disentangle=True,
        )
        # Task names come from the ``call_link_label`` each step is given.
        names = [t.name for t in wg.tasks]
        assert "prepare_kcw_wannier_files" in names
        assert "wann2kc" in names
        assert "screen" in names
        assert "ham" in names

    @pytest.mark.parametrize("check_spread", [True, False])
    def test_check_spread_input_controls_the_namelist(
        self, dfpt_codes, nscf_remote, occ_retrieved, check_spread
    ):
        wg = RunDFPT.build(
            kcw_code=dfpt_codes["kcw"],
            nscf_remote_folder=nscf_remote,
            block_wannier={"occ": {"retrieved": occ_retrieved}},
            occ_labels=["occ"],
            num_wann_occ=4,
            num_wann_emp=0,
            kgrid=[2, 2, 2],
            check_spread=check_spread,
        )
        screen_params = wg.tasks["screen"].inputs["parameters"].value
        assert screen_params["SCREEN"]["check_spread"] is check_spread

    def test_parallelization_reaches_every_kcw_step(self, dfpt_codes, nscf_remote, occ_retrieved):
        """Ntasks + -pd reach every kcw step; -npool reaches wann2kc/screen but not ham."""
        wg = RunDFPT.build(
            kcw_code=dfpt_codes["kcw"],
            nscf_remote_folder=nscf_remote,
            block_wannier={"occ": {"retrieved": occ_retrieved}},
            occ_labels=["occ"],
            num_wann_occ=4,
            num_wann_emp=0,
            kgrid=[2, 2, 2],
            parallelization={"kcw": {"ntasks": 8, "npool": 4, "pd": True}},
        )
        # wann2kc and screen take both -npool and -pd (legacy KCWWannier /
        # KCWScreen); ham takes only -pd (legacy KCWHam has no pool option).
        for name in ("wann2kc", "screen"):
            task = wg.tasks[name]
            assert task.inputs["settings"].value == {"cmdline": ["-npool", "4", "-pd", "true"]}
            assert (
                task.inputs["metadata"]["options"]["resources"].value["num_mpiprocs_per_machine"]
                == 8
            )
        ham = wg.tasks["ham"]
        assert ham.inputs["settings"].value == {"cmdline": ["-pd", "true"]}
        assert ham.inputs["metadata"]["options"]["resources"].value["num_mpiprocs_per_machine"] == 8

    def test_kcw_overrides_land_in_the_right_step_and_nowhere_else(
        self, dfpt_codes, nscf_remote, occ_retrieved
    ):
        """A control/screen/ham override reaches only the steps that read that namelist.

        ``control`` reaches every step (wann2kcw, screen, ham); ``screen``
        only the screen step; ``ham`` only the ham step.
        """
        wg = RunDFPT.build(
            kcw_code=dfpt_codes["kcw"],
            nscf_remote_folder=nscf_remote,
            block_wannier={"occ": {"retrieved": occ_retrieved}},
            occ_labels=["occ"],
            num_wann_occ=4,
            num_wann_emp=0,
            kgrid=[2, 2, 2],
            kcw_overrides={
                "control": {"lrpa": True},
                "screen": {"tr2": 1.0e-16},
                "ham": {"on_site_only": True},
            },
        )
        wann2kc_params = wg.tasks["wann2kc"].inputs["parameters"].value
        screen_params = wg.tasks["screen"].inputs["parameters"].value
        ham_params = wg.tasks["ham"].inputs["parameters"].value

        # The control override reaches every step.
        assert wann2kc_params["CONTROL"]["lrpa"] is True
        assert screen_params["CONTROL"]["lrpa"] is True
        assert ham_params["CONTROL"]["lrpa"] is True

        # The screen override reaches only the screen step.
        assert screen_params["SCREEN"]["tr2"] == pytest.approx(1.0e-16)
        assert "SCREEN" not in wann2kc_params
        assert "SCREEN" not in ham_params

        # The ham override reaches only the ham step.
        assert ham_params["HAM"]["on_site_only"] is True
        assert "HAM" not in wann2kc_params
        assert "HAM" not in screen_params

    def test_the_route_keeps_the_keywords_it_owns(self, dfpt_codes, nscf_remote, occ_retrieved):
        """An owned keyword takes the route's value, not the caller's.

        The seeded neighbours in the same namelists take the caller's, so a
        blanket "overrides are ignored" would not pass this.
        """
        wg = RunDFPT.build(
            kcw_code=dfpt_codes["kcw"],
            nscf_remote_folder=nscf_remote,
            block_wannier={"occ": {"retrieved": occ_retrieved}},
            occ_labels=["occ"],
            num_wann_occ=4,
            num_wann_emp=0,
            kgrid=[2, 2, 2],
            kcw_overrides={
                "control": {"mp1": 7, "kcw_at_ks": True, "kcw_iverbosity": 2},
                "wannier": {"num_wann_occ": 99, "check_ks": False},
                "ham": {"do_bands": True, "write_hr": False},
            },
        )
        control = wg.tasks["wann2kc"].inputs["parameters"].value["CONTROL"]
        wannier = wg.tasks["wann2kc"].inputs["parameters"].value["WANNIER"]
        ham_params = wg.tasks["ham"].inputs["parameters"].value

        # Owned: the route's value stands.
        assert control["mp1"] == 2
        assert control["kcw_at_ks"] is False
        assert wannier["num_wann_occ"] == 4
        # No band path was given, so no interpolation runs.
        assert ham_params["HAM"]["do_bands"] is False

        # Seeded, in those same namelists: the caller's value stands.
        assert control["kcw_iverbosity"] == 2
        assert wannier["check_ks"] is False
        assert ham_params["HAM"]["write_hr"] is False

    def test_alpha_guess_skips_screening(
        self, dfpt_codes, nscf_remote, occ_retrieved, emp_retrieved
    ):
        wg = RunDFPT.build(
            kcw_code=dfpt_codes["kcw"],
            nscf_remote_folder=nscf_remote,
            block_wannier={
                "occ": {"retrieved": occ_retrieved},
                "emp": {"retrieved": emp_retrieved},
            },
            occ_labels=["occ"],
            emp_labels=["emp"],
            num_wann_occ=4,
            num_wann_emp=4,
            kgrid=[2, 2, 2],
            alpha_guess=[0.3] * 8,
        )
        names = [t.name for t in wg.tasks]
        assert "screen" not in names
        assert "alphas_from_guess" in names
        assert "ham" in names


class TestRunDFPTMaterialization:
    """Materialize ``RunDFPT`` itself with resolved (non-socket) inputs.

    ``.build()`` (used throughout this file otherwise) only exercises
    ``RunDFPT`` at graph-*construction* time, where its inputs are still
    socket references. A ``@task.graph`` body runs again, with its inputs
    already resolved to plain values, whenever the workgraph is actually
    materialized (submitted, or reconstructed via ``WorkGraph.from_dict``
    for execution) — a distinct code path ``assert_graph_roundtrips``
    (a ``to_dict``/``from_dict`` round trip of the *unmaterialized* graph)
    does not exercise either. ``node_graph.utils.graph.materialize_graph``
    is what the engine actually calls at that point; calling it directly
    with real stored nodes reproduces exactly what a live run hits.

    This caught a real bug: ``wannierize_bands`` / ``projwfc`` are
    ``PwOutputs`` / ``ProjwfcOutputs`` namespaces whose ``output_parameters``
    field (and, for ``PwOutputs``, ``output_atomic_occupations`` — pw.x
    DFT+U) is declared plain ``dict`` (not ``orm.Dict``) — at materialization
    every ``dict``-typed field arrives fully deserialized, so echoing the
    whole namespace straight into ``RunDFPT``'s own output failed with
    ``Invalid graph return payload`` at ``outputs.wannierize_bands.output_parameters.<key>``
    (a raw Python value where a socket was required). Fixed by
    :func:`~aiida_koopmans.workgraphs.dfpt.emit_namespace_dict_field`, applied
    to every ``dict``-typed field :func:`~aiida_koopmans.workgraphs.dfpt._dict_typed_field_names`
    reads off each namespace's own TypedDict — not hard-coded to
    ``output_parameters`` alone, which would have missed
    ``output_atomic_occupations`` the same way the original bug missed both.
    """

    def test_wannierize_bands_and_projwfc_survive_materialization(
        self, aiida_localhost, tmp_path, dfpt_codes, nscf_remote, occ_retrieved
    ):
        from aiida.orm import BandsData, Dict, KpointsData, ProjectionData, RemoteData, XyData
        from aiida_workgraph import WorkGraph
        from node_graph.utils.graph import materialize_graph

        kpts = KpointsData()
        kpts.set_kpoints([[0.0, 0.0, 0.0]])
        bands = BandsData()
        bands.set_kpointsdata(kpts)
        bands.set_bands([[0.0, 1.0]])
        bands.store()

        # ``output_parameters`` needs a real pw.x-shaped key (``lkpoint_dir``
        # is the one the live run actually tripped on) so the reproduction
        # is exact, not just any dict. ``output_atomic_occupations`` (DFT+U)
        # is a second, independent dict-typed field on the same namespace,
        # not populated by the current quality-check bands run, but exactly
        # the shape a caller with Hubbard corrections would supply.
        wannierize_bands = {
            "remote_folder": RemoteData(
                computer=aiida_localhost, remote_path=str(tmp_path / "bands")
            ).store(),
            "output_parameters": Dict({"lkpoint_dir": False, "wall_time": "1.0s"}).store(),
            "output_atomic_occupations": Dict({"atom_1": {"3d": 1.5}}).store(),
            "output_band": bands,
        }
        projwfc = {
            "remote_folder": RemoteData(
                computer=aiida_localhost, remote_path=str(tmp_path / "projwfc")
            ).store(),
            "output_parameters": Dict({"lkpoint_dir": False}).store(),
            "Dos": XyData().store(),
            "projections": ProjectionData().store(),
            "bands": bands,
        }

        graph = materialize_graph(
            RunDFPT._callable,
            RunDFPT._inputs_spec,
            RunDFPT._outputs_spec,
            "dfpt_materialize",
            WorkGraph,
            args=(),
            kwargs={
                "kcw_code": dfpt_codes["kcw"],
                "nscf_remote_folder": nscf_remote,
                "block_wannier": {"occ": {"retrieved": occ_retrieved}},
                "occ_labels": ["occ"],
                "num_wann_occ": 4,
                "num_wann_emp": 0,
                "kgrid": [2, 2, 2],
                "wannierize_bands": wannierize_bands,
                "projwfc": projwfc,
            },
        )
        assert graph is not None


class TestSinglepointDFPTBuild:
    def test_occ_and_emp_manifolds(
        self, dfpt_codes, silicon_structure, kmesh, bands_path, fake_cutoffs_family
    ):
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={
                "none": {
                    "occ": [_block("occ", range(1, 5))],
                    "emp": [_block("emp", range(5, 9))],
                }
            },
            kpoints=kmesh,
            bands_kpoints=bands_path,
            # A real installed family: bands_kpoints unlocks WannierizeBlocks'
            # quality-check bands run, which always evaluates
            # projected_dos_supported(...) now (no "projwfc" in codes
            # short-circuit) — that call resolves the family.
            pseudo_family=fake_cutoffs_family.label,
            eps_inf=11.7,
        )
        names = [t.name for t in wg.tasks]
        assert names.count("scf_nscf") == 1
        # One WannierizeBlocks per channel covers both manifolds' blocks.
        assert names.count("wannierize") == 1
        assert "dfpt" in names

        # The single chain's results sit under channels.none in the dynamic
        # output namespace.
        channel_keys = [ns._name for ns in wg.outputs.channels]
        assert channel_keys == ["none"]
        result_keys = [s._name for s in wg.outputs.channels.none]
        for expected in ("alphas", "screen_parameters", "ham_parameters", "bands"):
            assert expected in result_keys

        # kcw.x needs an nspin=2 scratch even for closed-shell systems (the
        # DFPT perturbations are spin-dependent): both PW runs are forced to
        # nspin=2 / tot_magnetization=0, and the nscf drops symmetry.
        pw_overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        scf_system = pw_overrides["scf"]["pw"]["parameters"]["SYSTEM"]
        nscf_system = pw_overrides["nscf"]["pw"]["parameters"]["SYSTEM"]
        assert scf_system["nspin"] == 2

        # `wg.run()` reconstructs the graph from its serialized form before
        # executing; the nested WannierizeBlocks wiring must survive that.
        assert_graph_roundtrips(wg)
        assert scf_system["tot_magnetization"] == 0
        assert nscf_system["nspin"] == 2
        assert nscf_system["tot_magnetization"] == 0
        assert nscf_system["nosym"] is True
        assert nscf_system["noinv"] is True

        # pw2wannier90 must read the up channel of the nspin=2 scratch, and
        # the wannier90 runs must write the files kcw.x consumes.
        w90_overrides = wg.tasks["wannierize"].inputs["overrides"]
        inputpp = w90_overrides["pw2wannier90"].value
        assert inputpp["spin_component"] == "up"
        w90_params = w90_overrides["wannier90"].value
        assert w90_params["write_u_matrices"] is True
        assert w90_params["write_xyz"] is True

        # The wannierization reuses the shared scratch (no internal scf+nscf)
        # and sees the channel's blocks in band order: occupied then empty.
        block_labels = [b["label"] for b in wg.tasks["wannierize"].inputs["blocks"].value]
        assert block_labels == ["occ", "emp"]

        # RunDFPT gets the whole blocks namespace plus the band-ordered
        # manifold label lists it partitions by in its deferred body.
        assert wg.tasks["dfpt"].inputs["occ_labels"].value == ["occ"]
        assert wg.tasks["dfpt"].inputs["emp_labels"].value == ["emp"]

    def test_bands_kpoints_unlocks_the_wannierize_quality_check(
        self, dfpt_codes, silicon_structure, kmesh, bands_path, fake_cutoffs_family
    ):
        """A bands path threads through to WannierizeBlocks' quality check.

        The shared scf's scratch (not a fresh one) feeds the quality-check
        bands step, and its ``bands`` output reaches ``RunDFPT`` as
        ``wannierize_bands``. No projwfc code was configured, but the
        family supports the projected DOS
        (:func:`~aiida_koopmans.workgraphs.wannier90.projected_dos_supported`,
        the sole gate now), so ``RunDFPT``'s ``projwfc`` input still wires —
        the missing code surfaces as a structural error only when the graph
        actually runs.
        """
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            bands_kpoints=bands_path,
            pseudo_family=fake_cutoffs_family.label,
        )
        wannierize_inputs = wg.tasks["wannierize"].inputs
        scf_links = wannierize_inputs["scf_remote_folder"]._links
        assert [link.from_task.name for link in scf_links] == ["scf_nscf"]
        interp_links = wannierize_inputs["interpolation_kpoints"]._links
        assert [link.from_socket._name for link in interp_links] == ["bands_kpoints"]

        dfpt_inputs = wg.tasks["dfpt"].inputs
        assert dfpt_inputs["wannierize_bands"]._links
        assert dfpt_inputs["projwfc"]._links
        assert_graph_roundtrips(wg)

    def test_no_bands_kpoints_skips_the_wannierize_quality_check(
        self, dfpt_codes, silicon_structure, kmesh
    ):
        """Negative control: without a bands path, no quality-check wiring exists.

        ``scf_remote_folder`` is still handed to ``WannierizeBlocks`` (it is
        cheap and unconditional), but with no ``interpolation_kpoints`` the
        quality check itself never runs, so ``RunDFPT`` gets no
        ``wannierize_bands``.
        """
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )
        wannierize_inputs = wg.tasks["wannierize"].inputs
        assert not wannierize_inputs["interpolation_kpoints"]._links
        dfpt_inputs = wg.tasks["dfpt"].inputs
        assert not dfpt_inputs["wannierize_bands"]._links
        assert not dfpt_inputs["projwfc"]._links

    @pytest.mark.parametrize("bands_kpoints_given", [True, False])
    def test_cutoffless_family_needs_scf_and_nscf_cutoffs_threaded(
        self,
        dfpt_codes,
        silicon_structure,
        kmesh,
        bands_path,
        fake_cutoffless_family,
        bands_kpoints_given,
    ):
        """Both the shared scf's and nscf's cutoffs reach the nested wannier builder.

        ``WannierizeBlocks``' per-block ``Wannier90WorkChain.get_builder_from_protocol``
        call asks ``fake_cutoffless_family`` for cutoffs it does not
        recommend unless the caller's ``ecutwfc``/``ecutrho`` reach it
        through ``overrides["scf"]``/``overrides["nscf"]`` — regardless of
        whether a ``bands_kpoints`` path unlocks the quality-check bands
        step, since that construction happens either way (issue #98).
        """
        cutoffs = {"SYSTEM": {"ecutwfc": 30.0, "ecutrho": 240.0}}
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            bands_kpoints=bands_path if bands_kpoints_given else None,
            pseudo_family=fake_cutoffless_family.label,
            overrides={
                "scf": {"pw": {"parameters": dict(cutoffs)}},
                "nscf": {"pw": {"parameters": dict(cutoffs)}},
            },
        )
        overrides = wg.tasks["wannierize"].inputs["overrides"]
        scf = overrides["scf"].value
        nscf = overrides["nscf"].value
        assert scf["pw"]["parameters"]["SYSTEM"]["ecutwfc"] == 30.0
        assert scf["pw"]["parameters"]["SYSTEM"]["ecutrho"] == 240.0
        assert nscf["pw"]["parameters"]["SYSTEM"]["ecutwfc"] == 30.0
        assert nscf["pw"]["parameters"]["SYSTEM"]["ecutrho"] == 240.0

    def test_projwfc_code_chains_the_projected_dos_into_each_channel(
        self, dfpt_pdos_codes, silicon_structure, kmesh, bands_path, fake_cutoffs_family
    ):
        """A projwfc code, over pseudos it supports, reaches WannierizeBlocks and RunDFPT."""
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_pdos_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            bands_kpoints=bands_path,
            pseudo_family=fake_cutoffs_family.label,
        )
        codes_socket = wg.tasks["wannierize"].inputs["codes"]["projwfc"]
        assert [link.from_socket._name for link in codes_socket._links] == ["projwfc"]
        assert wg.tasks["dfpt"].inputs["projwfc"]._links

    def test_incapable_pseudos_leave_dfpt_projwfc_unwired(
        self, dfpt_pdos_codes, silicon_structure, kmesh, bands_path, fake_family_without_pswfc
    ):
        """A configured projwfc code over pseudos it does not support wires nothing.

        Mirrors :func:`WannierizeBlocks`' own gate
        (:func:`~aiida_koopmans.workgraphs.wannier90.projected_dos_supported`):
        wiring ``RunDFPT``'s ``projwfc`` input on the code's presence alone
        — without checking the pseudos too — would hand it a socket the
        inner projwfc step never populates. No warning is asserted here:
        this construction-level ``.build()`` never executes WannierizeBlocks'
        own (nested, nested-graph-deferred) body, so only this gate's own
        (deliberately silent) check runs.
        """
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_pdos_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            bands_kpoints=bands_path,
            pseudo_family=fake_family_without_pswfc.label,
        )
        assert not wg.tasks["dfpt"].inputs["projwfc"]._links
        # The quality-check bands run is unaffected: it does not depend on
        # the pseudos' atomic wavefunctions.
        assert wg.tasks["dfpt"].inputs["wannierize_bands"]._links

    def test_occ_only(self, dfpt_codes, silicon_structure, kmesh):
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )
        names = [t.name for t in wg.tasks]
        assert names.count("wannierize") == 1
        assert "dfpt" in names
        block_labels = [b["label"] for b in wg.tasks["wannierize"].inputs["blocks"].value]
        assert block_labels == ["occ"]

    def test_multi_block_manifolds_reach_one_wannierization(
        self, dfpt_codes, silicon_structure, kmesh
    ):
        """All of a channel's blocks feed one WannierizeBlocks; the kcw chain sees the totals."""
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={
                "none": {
                    "occ": [_block("occ_1", range(1, 3)), _block("occ_2", range(3, 5))],
                    "emp": [_block("emp_1", range(5, 7)), _block("emp_2", range(7, 9))],
                }
            },
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )
        names = [t.name for t in wg.tasks]
        assert names.count("wannierize") == 1
        assert "dfpt" in names
        assert names.count("scf_nscf") == 1
        # The per-block fan-out lives inside WannierizeBlocks (covered by its
        # own tests); here the channel hands it every block in band order.
        block_labels = [b["label"] for b in wg.tasks["wannierize"].inputs["blocks"].value]
        assert block_labels == ["occ_1", "occ_2", "emp_1", "emp_2"]

        dfpt_inputs = wg.tasks["dfpt"].inputs
        assert dfpt_inputs["num_wann_occ"].value == 4
        assert dfpt_inputs["num_wann_emp"].value == 4
        assert dfpt_inputs["nbnd_emp"].value == 4
        assert dfpt_inputs["occ_labels"].value == ["occ_1", "occ_2"]
        assert dfpt_inputs["emp_labels"].value == ["emp_1", "emp_2"]
        assert dfpt_inputs["check_spread"].value == True  # noqa: E712 — TaggedValue breaks `is`

    def test_check_spread_reaches_the_channel_chain(self, dfpt_codes, silicon_structure, kmesh):
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            check_spread=False,
        )
        assert wg.tasks["dfpt"].inputs["check_spread"].value == False  # noqa: E712 — TaggedValue breaks `is`

    def test_user_overrides_cannot_disable_nspin2(self, dfpt_codes, silicon_structure, kmesh):
        """The nspin=2 forcing is physics, so it wins over caller overrides."""
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            overrides={"scf": {"pw": {"parameters": {"SYSTEM": {"nspin": 1}}}}},
        )
        pw_overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        assert pw_overrides["scf"]["pw"]["parameters"]["SYSTEM"]["nspin"] == 2

    def test_collinear_fans_out_per_channel(self, dfpt_codes, silicon_structure, kmesh):
        from aiida_quantumespresso.common.types import SpinType

        magnetization = {"pw": {"parameters": {"SYSTEM": {"tot_magnetization": 2}}}}
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={
                "up": {
                    "occ": [_block("occ_up", range(1, 6))],
                    "emp": [_block("emp_up", range(6, 9))],
                },
                "down": {
                    "occ": [_block("occ_down", range(1, 4))],
                    "emp": [_block("emp_down", range(4, 9))],
                },
            },
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            spin=SpinType.COLLINEAR,
            overrides={"scf": magnetization, "nscf": magnetization},
        )
        names = [t.name for t in wg.tasks]
        assert names.count("scf_nscf") == 1
        for expected in (
            "wannierize_up",
            "dfpt_up",
            "wannierize_down",
            "dfpt_down",
        ):
            assert expected in names, names

        # Each channel gathers its own results namespace under channels.<key>.
        channel_keys = sorted(ns._name for ns in wg.outputs.channels)
        assert channel_keys == ["down", "up"]
        for key in ("up", "down"):
            result_keys = [s._name for s in wg.outputs.channels[key]]
            assert "alphas" in result_keys
            assert "ham_parameters" in result_keys

        # nspin=2 is still forced, but the magnetization is the caller's.
        pw_overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        scf_system = pw_overrides["scf"]["pw"]["parameters"]["SYSTEM"]
        assert scf_system["nspin"] == 2
        assert scf_system["tot_magnetization"] == 2

        # Each channel's wannierization selects its spin in both wannier90
        # and pw2wannier90, and each kcw chain reads its channel.
        for suffix, channel, component in (("_up", "up", 1), ("_down", "down", 2)):
            w90_overrides = wg.tasks[f"wannierize{suffix}"].inputs["overrides"]
            w90_params = w90_overrides["wannier90"].value
            assert w90_params["spin"] == channel
            inputpp = w90_overrides["pw2wannier90"].value
            assert inputpp["spin_component"] == channel
            assert wg.tasks[f"dfpt{suffix}"].inputs["spin_component"].value == component

    def test_collinear_requires_both_channels(self, dfpt_codes, silicon_structure, kmesh):
        from aiida_quantumespresso.common.types import SpinType

        with pytest.raises(ValueError, match="manifolds keyed by"):
            SinglepointDFPTWorkflow.build(
                codes=dfpt_codes,
                structure=silicon_structure,
                manifolds={"up": {"occ": [_block("occ_up", range(1, 5))]}},
                kpoints=kmesh,
                pseudo_family="SSSP/1.3/PBE/efficiency",
                spin=SpinType.COLLINEAR,
            )

    def test_spinor_single_chain(self, dfpt_codes, silicon_structure, kmesh):
        from aiida_quantumespresso.common.types import SpinType

        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            # Spinor manifold: counts doubled; single "none" channel.
            manifolds={"none": {"occ": [_block("occ", range(1, 9))]}},
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            spin=SpinType.SPIN_ORBIT,
        )
        names = [t.name for t in wg.tasks]
        assert "wannierize" in names
        assert "dfpt" in names
        assert "dfpt_down" not in names

        # Spinor scratch: noncolin + lspinorb instead of nspin=2.
        pw_overrides = wg.tasks["scf_nscf"].inputs["overrides"].value
        scf_system = pw_overrides["scf"]["pw"]["parameters"]["SYSTEM"]
        assert scf_system["noncolin"] is True
        assert scf_system["lspinorb"] is True
        assert "nspin" not in scf_system
        assert "tot_magnetization" not in scf_system

        # Spinor wannierization: spinors on, no channel selection anywhere.
        w90_overrides = wg.tasks["wannierize"].inputs["overrides"]
        w90_params = w90_overrides["wannier90"].value
        assert w90_params["spinors"] is True
        assert "spin" not in w90_params
        assert not w90_overrides["pw2wannier90"].value

    def test_spinor_user_magnetization_wins(self, dfpt_codes, silicon_structure, kmesh):
        """A caller-supplied starting_magnetization survives the domag nudge."""
        from aiida_quantumespresso.common.types import SpinType

        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 9))]}},
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            spin=SpinType.NON_COLLINEAR,
            overrides={
                "scf": {"pw": {"parameters": {"SYSTEM": {"starting_magnetization": [0.7]}}}}
            },
        )
        scf_task = next(t for t in wg.tasks if t.name == "scf_nscf")
        scf_overrides = scf_task.inputs.overrides.value
        system = scf_overrides["scf"]["pw"]["parameters"]["SYSTEM"]
        assert system["starting_magnetization"] == [0.7]
        assert system["noncolin"] is True
        # The nscf, with no user value, keeps the domag nudge.
        nscf_system = scf_overrides["nscf"]["pw"]["parameters"]["SYSTEM"]
        assert nscf_system["starting_magnetization"] == [0.001]


# ----------------------------------------------------------------------
# derive_dfpt_manifolds / normalize_alpha_guess (pure helpers)
# ----------------------------------------------------------------------


class TestDeriveDfptManifolds:
    def test_silicon_like_split(self, silicon_structure):
        from aiida_koopmans.projections import derive_dfpt_manifolds

        # Si-like valence: occupied s + p (2 atoms x (1 + 3) = 8 Wannier
        # functions; nelec=16 makes them all filled), empty p restricted to
        # mr=1 (pz -> 2 atoms x 1 = 2). The mr restriction is what pins the
        # len(m_r) multiplicity: counting 2l+1 regardless would give the
        # empty block 6 Wannier functions and overrun nbnd.
        occ = [
            Projection(site="Si", ang_mtm="l=0"),
            Projection(site="Si", ang_mtm="l=1"),
        ]
        emp = [Projection(site="Si", ang_mtm="l=1,mr=1")]
        occ_blocks, emp_blocks, has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=[occ, emp],
            nelec=16,
            nbnd=12,
        )
        (occ_block,) = occ_blocks
        assert occ_block["label"] == "occ"
        assert occ_block["num_wann"] == 8
        assert occ_block["num_bands"] == 8
        assert occ_block["exclude_bands"] == [9, 10, 11, 12]
        assert occ_block["projections"] == [
            "Si:l=0:0,0,1:1,0,0:1:1.0",
            "Si:l=1:0,0,1:1,0,0:1:1.0",
        ]
        (emp_block,) = emp_blocks
        assert emp_block["label"] == "emp"
        assert emp_block["num_wann"] == 2
        assert emp_block["num_bands"] == 4
        assert emp_block["projections"] == ["Si:l=1,mr=1:0,0,1:1,0,0:1:1.0"]
        # The empty block disentangles, so the two Wannier-function
        # indices it takes (9-10 of a 12-band nscf) are the only thing left
        # that could be read as an occupancy; the stamp says it is empty.
        assert occ_block["filled"] is True
        assert emp_block["filled"] is False
        assert emp_block["exclude_bands"] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert has_disentangle is True
        assert n_orbitals == 10

    def test_hybrid_multiplicity_and_no_empty(self, silicon_structure):
        from aiida_koopmans.projections import derive_dfpt_manifolds

        # sp3 hybrids: l=-3 -> 4 orbitals per atom, 2 atoms -> 8.
        occ = [Projection(site="Si", ang_mtm="l=-3")]
        occ_blocks, emp_blocks, has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=[occ],
            nelec=16,
            nbnd=None,
        )
        (occ_block,) = occ_blocks
        assert occ_block["num_wann"] == 8
        assert occ_block["exclude_bands"] is None
        assert emp_blocks == []
        assert has_disentangle is False
        assert n_orbitals == 8

    def test_straddling_block_raises(self, silicon_structure):
        from aiida_koopmans.projections import derive_dfpt_manifolds

        with pytest.raises(ValueError, match="straddles"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[[Projection(site="Si", ang_mtm="l=-3")]],  # 8 wann
                nelec=12,  # nocc = 6: block spans bands 1-8
                nbnd=8,
            )

    def test_no_projection_blocks_raises(self, silicon_structure):
        """DFPT screening cannot derive manifolds without explicit w90 projections."""
        from aiida_koopmans.projections import derive_dfpt_manifolds

        with pytest.raises(NotImplementedError, match="explicit Wannier90 projections"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[],
                nelec=16,
                nbnd=None,
            )

    def test_too_few_empty_bands_for_empty_projections_raises(self, silicon_structure):
        """``nbnd`` must leave at least as many empty bands as the empty blocks need."""
        from aiida_koopmans.projections import derive_dfpt_manifolds

        occ = [
            Projection(site="Si", ang_mtm="l=0"),
            Projection(site="Si", ang_mtm="l=1"),
        ]  # 8 wann occ (nocc=8)
        emp = [
            Projection(site="Si", ang_mtm="l=1")
        ]  # full p triplet: 3 orbitals x 2 atoms = 6 wann
        with pytest.raises(ValueError, match="leaves only 2 empty bands"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[occ, emp],
                nelec=16,
                nbnd=10,  # 10 - 8 = 2 empty bands, short of the 6 the empty block needs
            )

    def test_multi_block_manifolds(self, silicon_structure):
        """Multi-block band layout: consecutive windows, extras on the last block."""
        from aiida_koopmans.projections import derive_dfpt_manifolds

        blocks = [
            [Projection(site="Si", ang_mtm="l=0")],  # 2 wann: bands 1-2 (occ)
            [Projection(site="Si", ang_mtm="l=1")],  # 6 wann: bands 3-8 (occ)
            [Projection(site="Si", ang_mtm="l=0")],  # 2 wann: bands 9-10 (emp)
            [Projection(site="Si", ang_mtm="l=0")],  # 2 wann: bands 11-14 (emp + 2 extra)
        ]
        occ_blocks, emp_blocks, has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=blocks,
            nelec=16,
            nbnd=14,
        )
        assert [b["label"] for b in occ_blocks] == ["occ_1", "occ_2"]
        assert [b["label"] for b in emp_blocks] == ["emp_1", "emp_2"]
        assert [b["num_wann"] for b in occ_blocks + emp_blocks] == [2, 6, 2, 2]
        # Every block spans its own window out of 1..nbnd, except the last
        # empty block, which absorbs the extra disentanglement bands and
        # only excludes the bands below it.
        assert occ_blocks[0]["exclude_bands"] == [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        assert occ_blocks[1]["exclude_bands"] == [1, 2, 9, 10, 11, 12, 13, 14]
        assert emp_blocks[0]["exclude_bands"] == [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14]
        assert emp_blocks[0]["num_bands"] == 2
        assert emp_blocks[1]["exclude_bands"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert emp_blocks[1]["num_bands"] == 4
        # The two extra disentanglement bands show up in num_bands and in
        # the exclusions that stop at band 10 -- never in the derived
        # Wannier-function indices, which stay the map onto the block's own
        # two functions.
        assert get_wannier_indices(emp_blocks[1]) == [11, 12]
        assert [b["filled"] for b in occ_blocks] == [True, True]
        assert [b["filled"] for b in emp_blocks] == [False, False]
        for block in occ_blocks + emp_blocks:
            assert len(block["exclude_bands"] or []) + block["num_bands"] == 14
        assert has_disentangle is True
        assert n_orbitals == 12

    def test_incomplete_occupied_coverage_raises(self, silicon_structure):
        from aiida_koopmans.projections import derive_dfpt_manifolds

        blocks = [
            [Projection(site="Si", ang_mtm="l=0")],
            [Projection(site="Si", ang_mtm="l=0")],
        ]  # 2 + 2 occ
        with pytest.raises(ValueError, match="occupied projection blocks span"):
            derive_dfpt_manifolds(
                structure=silicon_structure, projection_blocks=blocks, nelec=12, nbnd=6
            )

    def test_odd_electron_count_raises(self, silicon_structure):
        from aiida_koopmans.projections import derive_dfpt_manifolds

        with pytest.raises(ValueError, match="Odd electron count"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[[Projection(site="Si", ang_mtm="l=0")]],
                nelec=7,
                nbnd=None,
            )

    def test_collinear_channel_requires_explicit_nocc(self, silicon_structure):
        from aiida_koopmans.projections import derive_dfpt_manifolds
        from aiida_koopmans.spin import SpinChannel

        with pytest.raises(ValueError, match="per-channel"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[[Projection(site="Si", ang_mtm="l=-3")]],
                nelec=16,
                nbnd=None,
                spin_channel=SpinChannel.UP,
            )

    def test_collinear_channels_use_given_nocc(self, silicon_structure):
        from aiida_koopmans.projections import derive_dfpt_manifolds
        from aiida_koopmans.spin import SpinChannel

        # A magnetic system: nelec=14, tot_magnetization=2 -> nocc 8 up / 6 down.
        up_blocks = [[Projection(site="Si", ang_mtm="l=-3")]]  # 8 wann
        dn_blocks = [[Projection(site="Si", ang_mtm="l=1")]]  # 6 wann
        occ_up_blocks, emp_up_blocks, _, n_up = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=up_blocks,
            nelec=14,
            nbnd=8,
            spin_channel=SpinChannel.UP,
            nocc=8,
        )
        occ_dn_blocks, emp_dn_blocks, _, n_dn = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=dn_blocks,
            nelec=14,
            nbnd=6,
            spin_channel=SpinChannel.DOWN,
            nocc=6,
        )
        (occ_up,) = occ_up_blocks
        (occ_dn,) = occ_dn_blocks
        assert occ_up["label"] == "occ_up"
        assert occ_up["spin"] == SpinChannel.UP
        assert (occ_up["num_wann"], n_up) == (8, 8)
        assert occ_dn["label"] == "occ_down"
        assert occ_dn["spin"] == SpinChannel.DOWN
        assert (occ_dn["num_wann"], n_dn) == (6, 6)
        assert emp_up_blocks == [] and emp_dn_blocks == []

    def test_spinor_doubles_num_wann_and_uses_nelec_occupations(self, silicon_structure):
        from aiida_koopmans.projections import derive_dfpt_manifolds
        from aiida_koopmans.spin import SpinChannel

        # KCW example05.1 nspin4: the same sp3 block that gives num_wann=8
        # in a collinear run spans 16 spinor Wannier functions, and all
        # nelec=16 bands are singly occupied.
        occ = [Projection(site="Si", ang_mtm="l=-3")]  # 8 orbitals -> 16 spinor WFs
        emp = [Projection(site="Si", ang_mtm="l=0")]  # 2 orbitals -> 4 spinor WFs
        occ_blocks, emp_blocks, has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=[occ, emp],
            nelec=16,
            nbnd=22,
            spin_channel=SpinChannel.SPINOR,
        )
        (occ_block,) = occ_blocks
        assert occ_block["label"] == "occ"
        assert occ_block["spin"] == SpinChannel.SPINOR
        assert occ_block["num_wann"] == 16
        assert occ_block["num_bands"] == 16
        assert occ_block["exclude_bands"] == list(range(17, 23))
        (emp_block,) = emp_blocks
        assert emp_block["num_wann"] == 4
        assert emp_block["num_bands"] == 6
        assert has_disentangle is True
        assert n_orbitals == 20


class TestNormalizeAlphaGuess:
    def test_uniform_float(self):
        from aiida_koopmans.workgraphs.dfpt import normalize_alpha_guess

        assert normalize_alpha_guess(0.3, 4) == [0.3, 0.3, 0.3, 0.3]

    def test_flat_list(self):
        from aiida_koopmans.workgraphs.dfpt import normalize_alpha_guess

        assert normalize_alpha_guess([0.1, 0.2], 2) == [0.1, 0.2]

    def test_nested_per_spin_list_takes_first_channel(self):
        from aiida_koopmans.workgraphs.dfpt import normalize_alpha_guess

        assert normalize_alpha_guess([[0.1, 0.2]], 2) == [0.1, 0.2]

    def test_nested_per_spin_list_selects_channel(self):
        from aiida_koopmans.spin import SpinChannel
        from aiida_koopmans.workgraphs.dfpt import normalize_alpha_guess

        nested = [[0.1, 0.2], [0.3, 0.4]]
        assert normalize_alpha_guess(nested, 2, SpinChannel.UP) == [0.1, 0.2]
        assert normalize_alpha_guess(nested, 2, SpinChannel.DOWN) == [0.3, 0.4]


# ----------------------------------------------------------------------
# Workflow-level orbital grouping (group_orbitals_tol)
# ----------------------------------------------------------------------


def _orbital(index: int, *, filled: bool, group_id: int, representative: bool) -> dict:
    return {
        "spin": SpinChannel.NONE.value,
        "index": index,
        "filled": filled,
        "group_id": group_id,
        "representative": representative,
    }


class TestSpreadsMetricRow:
    """Unit tests of the metric-row wrapper via its raw ``._callable``."""

    def test_wraps_one_row(self):
        from aiida_koopmans.workgraphs.variational_orbitals import spreads_metric_row

        assert spreads_metric_row._callable([1.1, 2.2, 3.3]) == [[1.1, 2.2, 3.3]]

    def test_expected_count_passes(self):
        from aiida_koopmans.workgraphs.variational_orbitals import spreads_metric_row

        assert spreads_metric_row._callable([0.5, 0.7], expected_count=2) == [[0.5, 0.7]]

    def test_count_mismatch_raises(self):
        """A spread list not covering every variational orbital is rejected."""
        from aiida_koopmans.workgraphs.variational_orbitals import spreads_metric_row

        with pytest.raises(ValueError, match="3 Wannier spreads for 4 variational orbitals"):
            spreads_metric_row._callable([1.1, 2.2, 3.3], expected_count=4)


class TestSingleOrbitalAlpha:
    def test_unwraps_the_single_entry(self):
        from aiida_koopmans.workgraphs.dfpt import single_orbital_alpha

        assert single_orbital_alpha._callable([0.25]) == 0.25

    def test_multi_entry_list_raises(self):
        from aiida_koopmans.workgraphs.dfpt import single_orbital_alpha

        with pytest.raises(ValueError, match="exactly one alpha"):
            single_orbital_alpha._callable([0.25, 0.3])


class TestAlphasInOrbitalOrder:
    def test_occupied_then_empty_ascending(self):
        from aiida_koopmans.workgraphs.dfpt import alphas_in_orbital_order

        orbitals = [
            _orbital(2, filled=True, group_id=1, representative=False),
            _orbital(1, filled=True, group_id=1, representative=True),
            _orbital(3, filled=False, group_id=2, representative=True),
        ]
        ordered = alphas_in_orbital_order._callable(
            orbitals=orbitals,
            filled_alphas={"orb_1": 0.1, "orb_2": 0.2},
            empty_alphas={"orb_3": 0.3},
        )
        assert ordered == [0.1, 0.2, 0.3]

    def test_no_empty_manifold(self):
        from aiida_koopmans.workgraphs.dfpt import alphas_in_orbital_order

        orbitals = [_orbital(1, filled=True, group_id=1, representative=True)]
        assert alphas_in_orbital_order._callable(
            orbitals=orbitals, filled_alphas={"orb_1": 0.4}
        ) == [0.4]

    def test_uncovered_orbital_raises(self):
        """An orbital the group broadcast never populated raises a named error."""
        from aiida_koopmans.workgraphs.dfpt import alphas_in_orbital_order

        orbitals = [
            _orbital(1, filled=True, group_id=1, representative=True),
            _orbital(2, filled=True, group_id=1, representative=False),
        ]
        with pytest.raises(ValueError, match=r"No alpha for orbital orb_2 .* did not cover it"):
            alphas_in_orbital_order._callable(orbitals=orbitals, filled_alphas={"orb_1": 0.1})


class TestGroupedKcwScreeningBuild:
    """Eager build of the fan-out graph on concrete (synthetic) orbitals."""

    def _build(self, dfpt_codes, nscf_remote, occ_retrieved, orbitals):
        from aiida_koopmans.workgraphs.dfpt import GroupedKcwScreening

        return GroupedKcwScreening.build(
            kcw_code=dfpt_codes["kcw"],
            control={"kcw_iverbosity": 1},
            wannier={"seedname": "aiida"},
            screen_namelist={"tr2": 1.0e-18},
            parent_folder=nscf_remote,
            wannier_files=occ_retrieved,
            orbitals=orbitals,
        )

    def test_one_screen_per_representative(self, dfpt_codes, nscf_remote, occ_retrieved):
        """Representatives fan out; group members don't run."""
        orbitals = [
            _orbital(1, filled=True, group_id=1, representative=True),
            _orbital(2, filled=True, group_id=1, representative=False),
            _orbital(3, filled=True, group_id=2, representative=True),
            _orbital(4, filled=False, group_id=3, representative=True),
        ]
        wg = self._build(dfpt_codes, nscf_remote, occ_retrieved, orbitals)
        names = [t.name for t in wg.tasks]
        for expected in ("screen_orb_1", "screen_orb_3", "screen_orb_4"):
            assert expected in names, names
        assert "screen_orb_2" not in names
        assert "expand_alphas_by_group" in names
        assert "alphas_in_orbital_order" in names

    def test_i_orb_and_check_spread_in_the_namelist(self, dfpt_codes, nscf_remote, occ_retrieved):
        """Each representative's run solves only its orbital, without kcw's internal grouping."""
        orbitals = [
            _orbital(1, filled=True, group_id=1, representative=True),
            _orbital(2, filled=True, group_id=2, representative=True),
        ]
        wg = self._build(dfpt_codes, nscf_remote, occ_retrieved, orbitals)
        for index in (1, 2):
            screen_params = wg.tasks[f"screen_orb_{index}"].inputs["parameters"].value
            assert screen_params["SCREEN"]["i_orb"] == index
            assert screen_params["SCREEN"]["check_spread"] == False  # noqa: E712 — TaggedValue breaks `is`
            assert screen_params["SCREEN"]["tr2"] == 1.0e-18


class TestRunDFPTGrouping:
    def test_grouping_replaces_the_single_screen(
        self, dfpt_codes, nscf_remote, occ_retrieved, emp_retrieved
    ):
        """With a tolerance set: spread extraction + clustering + deferred fan-out."""
        wg = RunDFPT.build(
            kcw_code=dfpt_codes["kcw"],
            nscf_remote_folder=nscf_remote,
            block_wannier={
                "occ": {"retrieved": occ_retrieved},
                "emp": {"retrieved": emp_retrieved},
            },
            occ_labels=["occ"],
            emp_labels=["emp"],
            spreads=[0.5] * 4 + [0.7] * 4,
            num_wann_occ=4,
            num_wann_emp=4,
            kgrid=[2, 2, 2],
            has_disentangle=True,
            group_orbitals_tol=0.05,
        )
        names = [t.name for t in wg.tasks]
        assert "screen" not in names
        for expected in ("spreads_metric_row", "assign_orbital_groups", "grouped_screen", "ham"):
            assert expected in names, names
        # The unified spreads cover occ + emp, guarded against the totals.
        metric_inputs = wg.tasks["spreads_metric_row"].inputs
        assert metric_inputs["expected_count"].value == 8
        # The clustering sees one metric row covering occ + emp.
        group_inputs = wg.tasks["assign_orbital_groups"].inputs
        assert group_inputs["nelup"].value == 4
        assert group_inputs["nbnd"].value == 8
        assert group_inputs["tol"].value == 0.05
        assert group_inputs["spin_polarized"].value == False  # noqa: E712 — TaggedValue breaks `is`

    def test_grouping_without_spreads_raises(self, dfpt_codes, nscf_remote, occ_retrieved):
        """The spread clustering depends on the unified wannier90 spreads."""
        with pytest.raises(ValueError, match="requires the channel's per-orbital"):
            RunDFPT.build(
                kcw_code=dfpt_codes["kcw"],
                nscf_remote_folder=nscf_remote,
                block_wannier={"occ": {"retrieved": occ_retrieved}},
                occ_labels=["occ"],
                num_wann_occ=4,
                num_wann_emp=0,
                kgrid=[2, 2, 2],
                group_orbitals_tol=0.05,
            )

    def test_alpha_guess_wins_over_grouping(self, dfpt_codes, nscf_remote, occ_retrieved):
        """A caller guess skips screening entirely, grouping included."""
        wg = RunDFPT.build(
            kcw_code=dfpt_codes["kcw"],
            nscf_remote_folder=nscf_remote,
            block_wannier={"occ": {"retrieved": occ_retrieved}},
            occ_labels=["occ"],
            num_wann_occ=4,
            num_wann_emp=0,
            kgrid=[2, 2, 2],
            alpha_guess=[0.3] * 4,
            group_orbitals_tol=0.05,
        )
        names = [t.name for t in wg.tasks]
        assert "alphas_from_guess" in names
        for absent in ("screen", "spreads_metric_row", "grouped_screen"):
            assert absent not in names, names


class TestSinglepointDFPTGrouping:
    def test_tol_reaches_the_channel_chain(self, dfpt_codes, silicon_structure, kmesh):
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={
                "none": {
                    "occ": [_block("occ", range(1, 5))],
                    "emp": [_block("emp", range(5, 9))],
                }
            },
            kpoints=kmesh,
            group_orbitals_tol=0.05,
        )
        dfpt_inputs = wg.tasks["dfpt"].inputs
        assert dfpt_inputs["group_orbitals_tol"].value == 0.05
        # The channel's unified WannierizeBlocks spreads are threaded to the
        # kcw chain alongside the retrieved folders (the spread clustering
        # consumes them).
        assert "wannierize" in [t.name for t in wg.tasks]
        assert "spreads" in [socket._name for socket in dfpt_inputs]

    def test_default_keeps_the_single_screen(self, dfpt_codes, silicon_structure, kmesh):
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
        )
        assert wg.tasks["dfpt"].inputs["group_orbitals_tol"].value is None
