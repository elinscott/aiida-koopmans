"""Construction-level tests for the Koopmans DFPT workgraphs.

Build the ``RunDFPT`` and ``SinglepointDFPTWorkflow`` graphs (no daemon, no
real code execution) and introspect their task lists / wiring, mirroring the
style of ``test_block_wannierize.py``. Also unit-tests the
``prepare_kcw_wannier_files`` calcfunction via its raw ``._callable``.
"""

from __future__ import annotations

import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
from aiida_koopmans.workgraphs.dfpt import (
    RunDFPT,
    SinglepointDFPTWorkflow,
    prepare_kcw_wannier_files,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def dfpt_codes(aiida_localhost):
    """Stand-in ``Codes`` dict (construction-only; never executed)."""
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
        "pw": _code("dfpt-pw", "quantumespresso.pw"),
        "wannier90": _code("dfpt-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("dfpt-p2w", "quantumespresso.pw2wannier90"),
        "kcw": _code("dfpt-kcw", "koopmans.kcw_wann2kc"),
    }


@pytest.fixture
def bands_path(aiida_profile):
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints([[0.0, 0.0, 0.0], [0.25, 0.0, 0.25], [0.5, 0.0, 0.5]])
    return kpts


@pytest.fixture
def nscf_remote(aiida_localhost, tmp_path):
    from aiida.orm import RemoteData

    return RemoteData(computer=aiida_localhost, remote_path=str(tmp_path)).store()


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

    from aiida_koopmans.wannier_merge import (
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
    n = len(include)
    return ExplicitProjectionBlock(
        label=label,
        spin=SpinChannel.NONE,
        num_wann=n,
        num_bands=n,
        include_bands=list(include),
        projection_type=WannierProjectionType.ANALYTIC,
        projections=["Si:sp3"],
    )


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
        from aiida_koopmans.wannier_merge import (
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
        from aiida_koopmans.wannier_merge import parse_wannier_u_file_shape

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
            codes=dfpt_codes,
            nscf_remote_folder=nscf_remote,
            occ_retrieved={"b00": occ_retrieved},
            emp_retrieved={"b00": emp_retrieved},
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
            codes=dfpt_codes,
            nscf_remote_folder=nscf_remote,
            occ_retrieved={"b00": occ_retrieved},
            num_wann_occ=4,
            num_wann_emp=0,
            kgrid=[2, 2, 2],
            check_spread=check_spread,
        )
        screen_params = wg.tasks["screen"].inputs["parameters"].value
        assert screen_params["SCREEN"]["check_spread"] is check_spread

    def test_alpha_guess_skips_screening(
        self, dfpt_codes, nscf_remote, occ_retrieved, emp_retrieved
    ):
        wg = RunDFPT.build(
            codes=dfpt_codes,
            nscf_remote_folder=nscf_remote,
            occ_retrieved={"b00": occ_retrieved},
            emp_retrieved={"b00": emp_retrieved},
            num_wann_occ=4,
            num_wann_emp=4,
            kgrid=[2, 2, 2],
            alpha_guess=[0.3] * 8,
        )
        names = [t.name for t in wg.tasks]
        assert "screen" not in names
        assert "alphas_from_guess" in names
        assert "ham" in names


class TestSinglepointDFPTBuild:
    def test_occ_and_emp_manifolds(self, dfpt_codes, silicon_structure, kmesh, bands_path):
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
            kgrid=[2, 2, 2],
            bands_kpoints=bands_path,
            pseudo_family="SSSP/1.3/PBE/efficiency",
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

    def test_occ_only(self, dfpt_codes, silicon_structure, kmesh):
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            kgrid=[2, 2, 2],
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
            kgrid=[2, 2, 2],
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
        assert dfpt_inputs["check_spread"].value == True  # noqa: E712 — TaggedValue breaks `is`

    def test_check_spread_reaches_the_channel_chain(self, dfpt_codes, silicon_structure, kmesh):
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            kgrid=[2, 2, 2],
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
            kgrid=[2, 2, 2],
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
            kgrid=[2, 2, 2],
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
                kgrid=[2, 2, 2],
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
            kgrid=[2, 2, 2],
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
            kgrid=[2, 2, 2],
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


class _FakeQuantumNumbers:
    def __init__(self, l_value, m_r=None):
        self.angular = type("A", (), {"value": l_value})()
        self.m_r = m_r

    def __str__(self):
        return f"l={self.angular.value}"


class _FakeProjection:
    def __init__(self, site, l_value, m_r=None):
        self.site = site
        self.ang_mtm = _FakeQuantumNumbers(l_value, m_r)


class TestDeriveDfptManifolds:
    def test_silicon_like_split(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        # Occupied block: 2 Si atoms x (l=1 -> 3 orbitals) + 2 x (m_r-restricted
        # l=0 -> 1 orbital) = 8 Wannier functions; nelec=16 makes them all filled.
        occ = [_FakeProjection("Si", 1), _FakeProjection("Si", 0, m_r=[1])]
        emp = [_FakeProjection("Si", 0)]
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
        assert occ_block["projections"] == ["Si:l=1", "Si:l=0"]
        (emp_block,) = emp_blocks
        assert emp_block["label"] == "emp"
        assert emp_block["num_wann"] == 2
        assert emp_block["num_bands"] == 4
        assert emp_block["exclude_bands"] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert has_disentangle is True
        assert n_orbitals == 10

    def test_hybrid_multiplicity_and_no_empty(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        # sp3 hybrids: l=-3 -> 4 orbitals per atom, 2 atoms -> 8.
        occ = [_FakeProjection("Si", -3)]
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
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        with pytest.raises(ValueError, match="straddles"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[[_FakeProjection("Si", -3)]],  # 8 wann
                nelec=12,  # nocc = 6: block spans bands 1-8
                nbnd=8,
            )

    def test_multi_block_manifolds(self, silicon_structure):
        """Multi-block band layout: consecutive windows, extras on the last block."""
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        blocks = [
            [_FakeProjection("Si", 0)],  # 2 wann: bands 1-2 (occ)
            [_FakeProjection("Si", 1)],  # 6 wann: bands 3-8 (occ)
            [_FakeProjection("Si", 0, m_r=[1])],  # 2 wann: bands 9-10 (emp)
            [_FakeProjection("Si", 0, m_r=[1])],  # 2 wann: bands 11-14 (emp + 2 extra)
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
        assert emp_blocks[1]["include_bands"] == [11, 12, 13, 14]
        assert has_disentangle is True
        assert n_orbitals == 12

    def test_incomplete_occupied_coverage_raises(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        blocks = [[_FakeProjection("Si", 0)], [_FakeProjection("Si", 0)]]  # 2 + 2 occ
        with pytest.raises(ValueError, match="occupied projection blocks span"):
            derive_dfpt_manifolds(
                structure=silicon_structure, projection_blocks=blocks, nelec=12, nbnd=6
            )

    def test_odd_electron_count_raises(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        with pytest.raises(ValueError, match="Odd electron count"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[[_FakeProjection("Si", 0)]],
                nelec=7,
                nbnd=None,
            )

    def test_collinear_channel_requires_explicit_nocc(self, silicon_structure):
        from aiida_koopmans.types import SpinChannel
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        with pytest.raises(ValueError, match="per-channel"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[[_FakeProjection("Si", -3)]],
                nelec=16,
                nbnd=None,
                spin_channel=SpinChannel.UP,
            )

    def test_collinear_channels_use_given_nocc(self, silicon_structure):
        from aiida_koopmans.types import SpinChannel
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        # A magnetic system: nelec=14, tot_magnetization=2 -> nocc 8 up / 6 down.
        up_blocks = [[_FakeProjection("Si", -3)]]  # 8 wann
        dn_blocks = [[_FakeProjection("Si", 1)]]  # 6 wann
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
        from aiida_koopmans.types import SpinChannel
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        # KCW example05.1 nspin4: the same sp3 block that gives num_wann=8
        # in a collinear run spans 16 spinor Wannier functions, and all
        # nelec=16 bands are singly occupied.
        occ = [_FakeProjection("Si", -3)]  # 8 orbitals -> 16 spinor WFs
        emp = [_FakeProjection("Si", 0)]  # 2 orbitals -> 4 spinor WFs
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
        from aiida_koopmans.types import SpinChannel
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
            code=dfpt_codes["kcw"],
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
            codes=dfpt_codes,
            nscf_remote_folder=nscf_remote,
            occ_retrieved={"b00": occ_retrieved},
            emp_retrieved={"b00": emp_retrieved},
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
                codes=dfpt_codes,
                nscf_remote_folder=nscf_remote,
                occ_retrieved={"b00": occ_retrieved},
                num_wann_occ=4,
                num_wann_emp=0,
                kgrid=[2, 2, 2],
                group_orbitals_tol=0.05,
            )

    def test_alpha_guess_wins_over_grouping(self, dfpt_codes, nscf_remote, occ_retrieved):
        """A caller guess skips screening entirely, grouping included."""
        wg = RunDFPT.build(
            codes=dfpt_codes,
            nscf_remote_folder=nscf_remote,
            occ_retrieved={"b00": occ_retrieved},
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
            kgrid=[2, 2, 2],
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
            kgrid=[2, 2, 2],
        )
        assert wg.tasks["dfpt"].inputs["group_orbitals_tol"].value is None
