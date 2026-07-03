"""Construction-level tests for the Koopmans DFPT workgraphs.

Build the ``KoopmansDFPTTask`` and ``SinglepointDFPT`` graphs (no daemon, no
real code execution) and introspect their task lists / wiring, mirroring the
style of ``test_block_wannierize.py``. Also unit-tests the
``prepare_kcw_wannier_files`` calcfunction via its raw ``._callable``.
"""

from __future__ import annotations

import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
from aiida_koopmans.workgraphs.dfpt import (
    KoopmansDFPTTask,
    SinglepointDFPT,
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
def silicon_structure(aiida_profile):
    from aiida.orm import StructureData

    cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
    struct = StructureData(cell=cell, pbc=True)
    struct.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
    struct.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si", name="Si")
    return struct


@pytest.fixture
def kmesh(aiida_profile):
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints_mesh([2, 2, 2])
    return kpts


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
        wg = KoopmansDFPTTask.build(
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
        wg = KoopmansDFPTTask.build(
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
        wg = SinglepointDFPT.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            occ_block=_block("occ", range(1, 5)),
            emp_block=_block("emp", range(5, 9)),
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

    def test_occ_only(self, dfpt_codes, silicon_structure, kmesh):
        wg = SinglepointDFPT.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            occ_block=_block("occ", range(1, 5)),
            kpoints=kmesh,
            kgrid=[2, 2, 2],
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )
        names = [t.name for t in wg.tasks]
        assert "wannierize_occ" in names
        assert "wannierize_emp" not in names
        assert "dfpt" in names


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
        occ_block, emp_block, has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=[occ, emp],
            nelec=16,
            nbnd=12,
        )
        assert occ_block["num_wann"] == 8
        assert occ_block["num_bands"] == 8
        assert occ_block["exclude_bands"] == "9-12"
        assert occ_block["projections"] == ["Si:l=1", "Si:l=0"]
        assert emp_block is not None
        assert emp_block["num_wann"] == 2
        assert emp_block["num_bands"] == 4
        assert emp_block["exclude_bands"] == "1-8"
        assert has_disentangle is True
        assert n_orbitals == 10

    def test_hybrid_multiplicity_and_no_empty(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        # sp3 hybrids: l=-3 -> 4 orbitals per atom, 2 atoms -> 8.
        occ = [_FakeProjection("Si", -3)]
        occ_block, emp_block, has_disentangle, n_orbitals = derive_dfpt_manifolds(
            structure=silicon_structure,
            projection_blocks=[occ],
            nelec=16,
            nbnd=None,
        )
        assert occ_block["num_wann"] == 8
        assert occ_block["exclude_bands"] is None
        assert emp_block is None
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

    def test_multi_occupied_blocks_raise(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        blocks = [[_FakeProjection("Si", 0)], [_FakeProjection("Si", 0)]]  # 2 + 2 occ
        with pytest.raises(NotImplementedError, match="merge machinery"):
            derive_dfpt_manifolds(
                structure=silicon_structure, projection_blocks=blocks, nelec=8, nbnd=4
            )

    def test_odd_electron_count_raises(self, silicon_structure):
        from aiida_koopmans.workgraphs.dfpt import derive_dfpt_manifolds

        with pytest.raises(NotImplementedError, match="Odd electron count"):
            derive_dfpt_manifolds(
                structure=silicon_structure,
                projection_blocks=[[_FakeProjection("Si", 0)]],
                nelec=7,
                nbnd=None,
            )


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
