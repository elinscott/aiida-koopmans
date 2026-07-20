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
        outputs = prepare_kcw_wannier_files._callable(occ_retrieved)
        names = sorted(outputs["wannier_files"].base.repository.list_object_names())
        assert names == ["aiida_centres.xyz", "aiida_hr.dat", "aiida_u.mat"]

    def test_emp_files_are_renamed(self, aiida_profile, occ_retrieved, emp_retrieved):
        outputs = prepare_kcw_wannier_files._callable(occ_retrieved, emp_retrieved)
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
            prepare_kcw_wannier_files._callable(incomplete)


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
            occ_retrieved=occ_retrieved,
            emp_retrieved=emp_retrieved,
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

    def test_alpha_guess_skips_screening(
        self, dfpt_codes, nscf_remote, occ_retrieved, emp_retrieved
    ):
        wg = RunDFPT.build(
            codes=dfpt_codes,
            nscf_remote_folder=nscf_remote,
            occ_retrieved=occ_retrieved,
            emp_retrieved=emp_retrieved,
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
                    "occ": _block("occ", range(1, 5)),
                    "emp": _block("emp", range(5, 9)),
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
        assert "wannierize_occ" in names
        assert "wannierize_emp" in names
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
        for wannierize in ("wannierize_occ", "wannierize_emp"):
            w90_overrides = wg.tasks[wannierize].inputs["overrides"]
            inputpp = w90_overrides["pw2wannier90"].value
            assert inputpp["spin_component"] == "up"
            w90_params = w90_overrides["wannier90"].value
            assert w90_params["write_u_matrices"] is True
            assert w90_params["write_xyz"] is True

    def test_occ_only(self, dfpt_codes, silicon_structure, kmesh):
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": _block("occ", range(1, 5))}},
            kpoints=kmesh,
            kgrid=[2, 2, 2],
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )
        names = [t.name for t in wg.tasks]
        assert "wannierize_occ" in names
        assert "wannierize_emp" not in names
        assert "dfpt" in names

    def test_user_overrides_cannot_disable_nspin2(self, dfpt_codes, silicon_structure, kmesh):
        """The nspin=2 forcing is physics, so it wins over caller overrides."""
        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": _block("occ", range(1, 5))}},
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
                    "occ": _block("occ_up", range(1, 6)),
                    "emp": _block("emp_up", range(6, 9)),
                },
                "down": {
                    "occ": _block("occ_down", range(1, 4)),
                    "emp": _block("emp_down", range(4, 9)),
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
            "wannierize_occ_up",
            "wannierize_emp_up",
            "dfpt_up",
            "wannierize_occ_down",
            "wannierize_emp_down",
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
            w90_overrides = wg.tasks[f"wannierize_occ{suffix}"].inputs["overrides"]
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
                manifolds={"up": {"occ": _block("occ_up", range(1, 5))}},
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
            manifolds={"none": {"occ": _block("occ", range(1, 9))}},
            kpoints=kmesh,
            kgrid=[2, 2, 2],
            pseudo_family="SSSP/1.3/PBE/efficiency",
            spin=SpinType.SPIN_ORBIT,
        )
        names = [t.name for t in wg.tasks]
        assert "wannierize_occ" in names
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
        w90_overrides = wg.tasks["wannierize_occ"].inputs["overrides"]
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
            manifolds={"none": {"occ": _block("occ", range(1, 9))}},
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
        occ_block, emp_block, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=[occ, emp],
            nelec=16,
            nbnd=12,
        )
        assert occ_block["num_wann"] == 8
        assert occ_block["num_bands"] == 8
        assert occ_block["exclude_bands"] == [9, 10, 11, 12]
        assert occ_block["projections"] == ["Si:l=1", "Si:l=0"]
        assert emp_block is not None
        assert emp_block["num_wann"] == 2
        assert emp_block["num_bands"] == 4
        assert emp_block["exclude_bands"] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert emp_block is not None and emp_block["num_bands"] != emp_block["num_wann"]
        assert n_orbitals == 10

    def test_hybrid_multiplicity_and_no_empty(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        # sp3 hybrids: l=-3 -> 4 orbitals per atom, 2 atoms -> 8.
        occ = [_FakeProjection("Si", -3)]
        occ_block, emp_block, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=[occ],
            nelec=16,
            nbnd=None,
        )
        assert occ_block["num_wann"] == 8
        assert occ_block["exclude_bands"] is None
        assert emp_block is None
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

    def test_multi_occupied_blocks_raise(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        blocks = [[_FakeProjection("Si", 0)], [_FakeProjection("Si", 0)]]  # 2 + 2 occ
        with pytest.raises(NotImplementedError, match="merge machinery"):
            derive_dfpt_manifolds(
                structure=silicon_structure, projection_blocks=blocks, nelec=8, nbnd=4
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
        occ_up, emp_up, n_up = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=up_blocks,
            nelec=14,
            nbnd=8,
            spin_channel=SpinChannel.UP,
            nocc=8,
        )
        occ_dn, emp_dn, n_dn = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=dn_blocks,
            nelec=14,
            nbnd=6,
            spin_channel=SpinChannel.DOWN,
            nocc=6,
        )
        assert occ_up["label"] == "occ_up"
        assert occ_up["spin"] == SpinChannel.UP
        assert (occ_up["num_wann"], n_up) == (8, 8)
        assert occ_dn["label"] == "occ_down"
        assert occ_dn["spin"] == SpinChannel.DOWN
        assert (occ_dn["num_wann"], n_dn) == (6, 6)
        assert emp_up is None and emp_dn is None

    def test_spinor_doubles_num_wann_and_uses_nelec_occupations(self, silicon_structure):
        from aiida_koopmans.types import SpinChannel
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        # KCW example05.1 nspin4: the same sp3 block that gives num_wann=8
        # in a collinear run spans 16 spinor Wannier functions, and all
        # nelec=16 bands are singly occupied.
        occ = [_FakeProjection("Si", -3)]  # 8 orbitals -> 16 spinor WFs
        emp = [_FakeProjection("Si", 0)]  # 2 orbitals -> 4 spinor WFs
        occ_block, emp_block, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=[occ, emp],
            nelec=16,
            nbnd=22,
            spin_channel=SpinChannel.SPINOR,
        )
        assert occ_block["label"] == "occ"
        assert occ_block["spin"] == SpinChannel.SPINOR
        assert occ_block["num_wann"] == 16
        assert occ_block["num_bands"] == 16
        assert occ_block["exclude_bands"] == list(range(17, 23))
        assert emp_block is not None
        assert emp_block["num_wann"] == 4
        assert emp_block["num_bands"] == 6
        assert emp_block is not None and emp_block["num_bands"] != emp_block["num_wann"]
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
