"""Tests for the mesh an SCF step samples.

Three graphs take an SCF mesh — ``RunScfNscf``, ``RunPwBands`` and
``DielectricTask`` — and each is checked on the materialized
``PwBaseWorkChain`` inputs: an explicit mesh must displace the protocol's
``kpoints_distance``, and no mesh must leave the protocol in charge. The
graphs above these (``WannierizeBlocks``, ``MlwfInitialization``,
``SinglepointDFPTWorkflow``) keep nested graphs as single tasks at build
time, so they are checked on the forwarded socket.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.projections import ExplicitProjectionBlock
from tests.fixtures import explicit_block


def _mesh(task, socket="kpoints"):
    """Return the Monkhorst-Pack grid of a task's mesh socket, or ``None``."""
    value = task.inputs[socket].value
    return None if value is None else list(value.get_kpoints_mesh()[0])


def _step_mesh(task, step):
    """Return the Monkhorst-Pack grid of a nested step's mesh socket, or ``None``."""
    value = task.inputs[step]["kpoints"].value
    return None if value is None else list(value.get_kpoints_mesh()[0])


def _silicon_blocks() -> list[ExplicitProjectionBlock]:
    return [
        explicit_block("block_1", range(1, 5)),
        explicit_block("block_2", range(5, 9)),
    ]


class TestRunScfNscf:
    def test_explicit_mesh_displaces_the_protocol_distance(
        self, fake_cutoffs_family, silicon_structure, kmesh, kpath, pw_code
    ):
        """The SCF samples ``scf_kpoints``; the NSCF keeps its own k-points."""
        from aiida_koopmans.workgraphs.pw import RunScfNscf

        wg = RunScfNscf.build(
            code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            scf_kpoints=kmesh,
            nscf_kpoints=kpath,
        )
        scf, nscf = wg.tasks["scf"], wg.tasks["nscf"]

        assert scf.inputs["kpoints"].value.uuid == kmesh.uuid
        # The workchain accepts exactly one of the two, so the protocol's
        # distance has to be gone rather than merely overruled.
        assert scf.inputs["kpoints_distance"].value is None
        assert scf.inputs["kpoints_force_parity"].value is None

        assert nscf.inputs["kpoints"].value.uuid == kpath.uuid
        assert nscf.inputs["kpoints_distance"].value is None

    def test_no_mesh_leaves_the_protocol_in_charge(
        self, fake_cutoffs_family, silicon_structure, kpath, pw_code
    ):
        """Without ``scf_kpoints`` the SCF still gets its mesh from the protocol.

        Callers that prescribe no mesh must keep working, so setting one
        cannot become mandatory by accident.
        """
        from aiida_koopmans.workgraphs.pw import RunScfNscf

        wg = RunScfNscf.build(
            code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kpath,
        )
        scf = wg.tasks["scf"]
        assert scf.inputs["kpoints"].value is None
        assert scf.inputs["kpoints_distance"].value.value > 0


class TestRunPwBands:
    def test_explicit_mesh_displaces_the_protocol_distance(
        self, fake_cutoffs_family, silicon_structure, kmesh, kpath, pw_code
    ):
        """The SCF samples ``scf_kpoints``; the bands step keeps its path."""
        from aiida_koopmans.workgraphs.pw import RunPwBands

        wg = RunPwBands.build(
            codes={"pw": pw_code},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            scf_kpoints=kmesh,
            bands_kpoints=kpath,
        )
        task = wg.tasks["PwBandsWorkChain"]

        assert task.inputs["scf"]["kpoints"].value.uuid == kmesh.uuid
        # The workchain accepts exactly one of the two, so the protocol's
        # distance has to be gone rather than merely overruled.
        assert task.inputs["scf"]["kpoints_distance"].value is None
        assert task.inputs["scf"]["kpoints_force_parity"].value is None

        # The bands step samples a path, not a mesh; it must be unaffected.
        assert task.inputs["bands_kpoints"].value.uuid == kpath.uuid

    def test_no_mesh_leaves_the_protocol_in_charge(
        self, fake_cutoffs_family, silicon_structure, kpath, pw_code
    ):
        """Without ``scf_kpoints`` the SCF still gets its mesh from the protocol."""
        from aiida_koopmans.workgraphs.pw import RunPwBands

        wg = RunPwBands.build(
            codes={"pw": pw_code},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kpath,
        )
        scf = wg.tasks["PwBandsWorkChain"].inputs["scf"]
        assert scf["kpoints"].value is None
        assert scf["kpoints_distance"].value.value > 0


class TestDielectricTask:
    def test_explicit_mesh_displaces_the_protocol_distance(
        self, ph_codes, silicon_structure, fake_cutoffs_family, kmesh
    ):
        """The ground state the response is taken about samples ``scf_kpoints``."""
        from aiida_koopmans.workgraphs.ph import DielectricTask

        wg = DielectricTask.build(
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            scf_kpoints=kmesh,
        )
        scf = wg.tasks["scf"]
        assert scf.inputs["kpoints"].value.uuid == kmesh.uuid
        assert scf.inputs["kpoints_distance"].value is None
        assert scf.inputs["kpoints_force_parity"].value is None

        # The q-mesh is a separate sampling and stays at Gamma.
        assert wg.tasks["ph"].inputs["qpoints"].value.get_kpoints_mesh()[0] == [1, 1, 1]

    def test_no_mesh_leaves_the_protocol_in_charge(
        self, ph_codes, silicon_structure, fake_cutoffs_family
    ):
        from aiida_koopmans.workgraphs.ph import DielectricTask

        wg = DielectricTask.build(
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
        )
        scf = wg.tasks["scf"]
        assert scf.inputs["kpoints"].value is None
        assert scf.inputs["kpoints_distance"].value.value > 0


class TestWannierizeBlocks:
    def test_mesh_reaches_the_shared_scf(self, wannier_codes, silicon_structure, kmesh, kpath):
        from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks

        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kpath,
            scf_kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )
        scf_nscf = wg.tasks["scf_nscf"]
        assert _mesh(scf_nscf, "scf_kpoints") == [2, 2, 2]
        # The Wannierization k-list is untouched: the blocks read the full
        # unreduced grid whatever the SCF samples.
        assert scf_nscf.inputs["nscf_kpoints"].value.uuid == kpath.uuid

    def test_external_scratch_rejects_a_mesh(
        self, wannier_codes, silicon_structure, kmesh, kpath, nscf_remote
    ):
        """No SCF runs here, so a mesh for it names the mistake rather than vanish."""
        from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks

        with pytest.raises(ValueError, match=r"scf_kpoints.*nscf_remote_folder"):
            WannierizeBlocks.build(
                codes=wannier_codes,
                structure=silicon_structure,
                blocks=_silicon_blocks(),
                kpoints=kpath,
                scf_kpoints=kmesh,
                nscf_remote_folder=nscf_remote,
                pseudo_family="SSSP/1.3/PBE/efficiency",
            )


class TestSinglepointDFPT:
    def test_the_scf_samples_the_kpoints_mesh(self, dfpt_codes, silicon_structure, kmesh, kpath):
        """The user mesh feeds the SCF; the NSCF gets its unreduced expansion."""
        from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow

        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [explicit_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            bands_kpoints=kpath,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf=11.7,
        )
        scf_nscf = wg.tasks["scf_nscf"]
        assert _mesh(scf_nscf, "scf_kpoints") == [2, 2, 2]
        assert len(scf_nscf.inputs["nscf_kpoints"].value.get_kpoints()) == 8
        # The mesh kcw.x counts in comes from ``kpoints`` alone.
        assert list(wg.tasks["dfpt"].inputs["kgrid"].value) == [2, 2, 2]

    def test_a_kpath_names_the_mistake(self, dfpt_codes, silicon_structure, kpath):
        """An explicit list has no mesh dimensions, so it cannot reach kcw.x unnoticed."""
        from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow

        with pytest.raises(ValueError, match=r"kpoints.*Monkhorst-Pack mesh"):
            SinglepointDFPTWorkflow.build(
                codes=dfpt_codes,
                structure=silicon_structure,
                manifolds={"none": {"occ": [explicit_block("occ", range(1, 5))]}},
                kpoints=kpath,
                pseudo_family="SSSP/1.3/PBE/efficiency",
                eps_inf=11.7,
            )

    def test_scf_kpoints_moves_the_scf_alone(
        self, dfpt_codes, silicon_structure, kmesh, kpath, denser_kmesh
    ):
        """A denser ground state must not move the mesh kcw.x counts in."""
        from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow

        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [explicit_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            scf_kpoints=denser_kmesh,
            bands_kpoints=kpath,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf=11.7,
        )
        scf_nscf = wg.tasks["scf_nscf"]
        assert _mesh(scf_nscf, "scf_kpoints") == [4, 4, 4]
        assert len(scf_nscf.inputs["nscf_kpoints"].value.get_kpoints()) == 8
        # kcw.x counts in the nscf mesh, so the denser SCF must leave
        # ``CONTROL.mp1-3`` on the 2x2x2 the Wannier functions were built on.
        assert list(wg.tasks["dfpt"].inputs["kgrid"].value) == [2, 2, 2]

    def test_an_scf_kpoints_distance_leaves_the_scf_meshless(
        self, dfpt_codes, silicon_structure, kmesh, kpath
    ):
        """The two inputs exclude each other, so the fallback mesh has to stand down."""
        from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow

        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": [explicit_block("occ", range(1, 5))]}},
            kpoints=kmesh,
            overrides={"scf": {"kpoints_distance": 0.11}},
            bands_kpoints=kpath,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf=11.7,
        )
        scf_nscf = wg.tasks["scf_nscf"]
        assert scf_nscf.inputs["scf_kpoints"].value is None
        assert scf_nscf.inputs["overrides"].value["scf"]["kpoints_distance"] == 0.11


class TestMlwfInitialization:
    def test_the_wannierization_scf_samples_the_kpoints_mesh(
        self, mlwf_codes, periodic_ozone_structure, ozone_real_pseudos
    ):
        from aiida.orm import KpointsData, List

        from aiida_koopmans.workgraphs.mlwf_init import MlwfInitialization
        from aiida_koopmans.workgraphs.supercell import primitive_to_supercell

        kpoints = KpointsData()
        kpoints.set_kpoints_mesh([2, 1, 1])
        supercell = primitive_to_supercell._callable(periodic_ozone_structure, List(list=[2, 1, 1]))
        wg = MlwfInitialization.build(
            codes={**mlwf_codes, "kcp": mlwf_codes["pw"]},
            structure=periodic_ozone_structure,
            supercell=supercell,
            pseudos=ozone_real_pseudos,
            blocks=[explicit_block("block_1", range(1, 10), filled=True)],
            kpoints=kpoints,
            kgrid=[2, 1, 1],
            nelec=36,
            nelup=18,
            neldw=18,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=20,
            pseudo_family="unused-here",
        )
        assert _mesh(wg.tasks["wannierize"], "scf_kpoints") == [2, 1, 1]
