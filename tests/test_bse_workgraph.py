"""Construction-level tests for :func:`RunBse` and :func:`SinglepointBSEWorkflow`.

Build the graphs (no daemon, no real yambo/QE execution) and introspect the
task list / wiring, mirroring the style of ``test_block_wannierize.py``.
``k2y`` and ``aiida_yambo`` are regular ``aiida-koopmans`` dependencies, but
the plain canonical venv the rest of the suite runs against has not synced
them in (see ``test_bse_calcfunction.py``): ``pytest.importorskip`` keeps
this module a clean skip there, not a collection error. ``RunBse`` needs
``aiida_yambo`` itself (``WorkflowFactory('yambo.yambo.yambowf')``), on top
of the ``k2y`` import ``workgraphs/bse.py`` already makes at module scope.

``fake_cutoffs_family`` (not a bare ``"SSSP/..."`` string) is required
here: unlike the block-wannierize / DFPT graphs, whose PW steps go through
``Wannier90WorkChain.get_scf_nscf_builders_from_protocol``,
:func:`~aiida_koopmans.workgraphs.bse.RunBse` builds its scf/nscf steps
through ``YamboWorkflow.get_builder_from_protocol``, which calls
``PwBaseWorkChain.get_builder_from_protocol`` directly -- the strict path
that looks the family up in the database regardless of whether the
overrides already carry both cutoffs.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("k2y")
pytest.importorskip("aiida_yambo")

from aiida_koopmans.workgraphs.bse import RunBse, SinglepointBSEWorkflow
from tests.fixtures import assert_graph_roundtrips, explicit_block

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def bse_codes(aiida_localhost):
    """Return a codes dict of stand-in nodes for :func:`RunBse` (pw, p2y, yambo)."""
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
        "pw": _code("bse-pw", "quantumespresso.pw"),
        "p2y": _code("bse-p2y", "yambo.yambo"),
        "yambo": _code("bse-yambo", "yambo.yambo"),
    }


@pytest.fixture
def bse_full_codes(dfpt_codes, bse_codes):
    """Return the combined codes namespace :func:`SinglepointBSEWorkflow` needs."""
    return {**dfpt_codes, "p2y": bse_codes["p2y"], "yambo": bse_codes["yambo"]}


@pytest.fixture
def nscf_output_parameters() -> dict:
    """Return a koopmans nscf's parsed ``output_parameters`` (eV cutoffs).

    653.06 / 2612.24 eV are 48 / 192 Ry (``qe_tools.CONSTANTS.hartree_to_ev
    / 2`` converts back), a 1:4 wfc:rho ratio typical of a norm-conserving
    pseudopotential.
    """
    return {"wfc_cutoff": 653.0665, "rho_cutoff": 2612.266}


@pytest.fixture
def nscf_output_band(kmesh):
    """Return an ``output_band`` on the same 2x2x2 mesh as ``kmesh``."""
    from aiida.orm import BandsData

    n_kpoints = int(np.prod(kmesh.get_kpoints_mesh()[0]))
    bands = BandsData()
    bands.set_kpoints(np.zeros((n_kpoints, 3)))
    bands.set_bands(np.zeros((n_kpoints, 4)))
    return bands.store()


@pytest.fixture
def ham_output_parameters(nscf_output_band) -> dict:
    """Return a synthetic kcw.x ``ham`` ``output_parameters`` (KI/KS eigenvalues)."""
    n_kpoints = nscf_output_band.get_array("kpoints").shape[0]
    n_bands = 4
    ks = np.tile(np.arange(n_bands, dtype=float), (n_kpoints, 1))
    return {
        "ki_eigenvalues_on_grid": (ks - 0.3).tolist(),
        "ks_eigenvalues_on_grid": ks.tolist(),
    }


@pytest.fixture
def bse_parameters() -> dict:
    """Return a minimal yambo BSE ``variables``/``arguments`` block."""
    return {
        "arguments": ["rim_cut"],
        "variables": {
            "BndsRnXs": [[1, 20], ""],
            "NGsBlkXs": [2, "Ry"],
            "BSENGBlk": [2, "Ry"],
            "BSEBands": [[3, 6], ""],
            "BEnRange": [[0, 10], "eV"],
            "BEnSteps": [1000, ""],
            "BDmRange": [[0.1, 0.1], "eV"],
        },
    }


def _manifolds():
    return {
        "none": {
            "occ": [explicit_block("block_1", range(1, 5), filled=True)],
            "emp": [explicit_block("block_2", range(5, 9), filled=False)],
        }
    }


# ----------------------------------------------------------------------
# RunBse
# ----------------------------------------------------------------------


class TestRunBseGraphBuild:
    def _build(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
        **extra,
    ):
        return RunBse.build(
            codes=bse_codes,
            structure=silicon_structure,
            kpoints=kmesh,
            nscf_output_band=nscf_output_band,
            nscf_output_parameters=nscf_output_parameters,
            ham_output_parameters=ham_output_parameters,
            bse_parameters=bse_parameters,
            pseudo_family=fake_cutoffs_family.label,
            **extra,
        )

    def test_graph_builds_init_qp_and_bse_tasks(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            nscf_output_parameters,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        names = [t.name for t in wg.tasks]
        assert "yambo_init" in names
        assert "bse" in names
        assert "generate_qp_database" in names
        assert_graph_roundtrips(wg)

    def test_init_and_bse_carry_both_cutoffs(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        """Landmine C: an ecutwfc-only override is replaced wholesale by the family's own.

        Both ``ecutwfc`` and ``ecutrho`` must reach every scf/nscf SYSTEM
        namelist ``PwBaseWorkChain.get_builder_from_protocol`` builds here
        (48 / 192 Ry, from ``nscf_output_parameters``'s eV cutoffs), rather
        than ``fake_cutoffs_family``'s own recommendation (30 / 240 Ry) --
        confirming the override wins, not the family default.
        """
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            nscf_output_parameters,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        for task_name in ("yambo_init", "bse"):
            for step in ("scf", "nscf"):
                system = (
                    wg.tasks[task_name].inputs[step]["pw"]["parameters"].value.get_dict()["SYSTEM"]
                )
                assert system["ecutwfc"] == pytest.approx(48.0, rel=1e-3)
                assert system["ecutrho"] == pytest.approx(192.0, rel=1e-3)

    def test_kmesh_matches_the_koopmans_nscf_mesh(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            nscf_output_parameters,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        for task_name in ("yambo_init", "bse"):
            yambo_kpoints = wg.tasks[task_name].inputs["nscf"]["kpoints"].value
            assert list(yambo_kpoints.get_kpoints_mesh()[0]) == [2, 2, 2]

    def test_parallelization_sets_ranks_on_both_yambo_steps_and_bs_roles_on_bse(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        """``parallelization['yambo']['ntasks']`` reaches both calc's resources.

        The BSE step alone also gets a k-point-only ``BS_CPU``/``BS_ROLEs``
        split (see :func:`~aiida_koopmans.workgraphs.bse._bse_mpi_roles`) --
        yambo_init runs with ``INITIALISE=True`` and never reads those keys.
        """
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            nscf_output_parameters,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
            parallelization={"yambo": {"ntasks": 4}},
        )
        for task_name in ("yambo_init", "bse"):
            resources = (
                wg.tasks[task_name]
                .inputs["yres"]["yambo"]["metadata"]["options"]["resources"]
                .value
            )
            assert resources["num_mpiprocs_per_machine"] == 4
        bse_variables = (
            wg.tasks["bse"].inputs["yres"]["yambo"]["parameters"].value.get_dict()["variables"]
        )
        assert bse_variables["BS_CPU"] == "4 1 1"
        assert bse_variables["BS_ROLEs"] == "k eh t"

    def test_single_rank_omits_bs_roles(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            nscf_output_parameters,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        bse_variables = (
            wg.tasks["bse"].inputs["yres"]["yambo"]["parameters"].value.get_dict()["variables"]
        )
        assert "BS_CPU" not in bse_variables
        assert "BS_ROLEs" not in bse_variables

    def test_bse_qp_corrections_wired_from_the_qp_database(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            nscf_output_parameters,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        qp_socket = wg.tasks["bse"].inputs["yres"]["yambo"]["QP_corrections"]
        assert qp_socket._links
        parent_folder_socket = wg.tasks["bse"].inputs["parent_folder"]
        assert parent_folder_socket._links

    def test_kfnqpdb_override_is_refused(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        bse_parameters = {**bse_parameters}
        bse_parameters["variables"] = {**bse_parameters["variables"], "KfnQPdb": "nonsense"}
        with pytest.raises(ValueError, match="KfnQPdb"):
            self._build(
                bse_codes,
                silicon_structure,
                kmesh,
                nscf_output_band,
                nscf_output_parameters,
                ham_output_parameters,
                bse_parameters,
                fake_cutoffs_family,
            )

    def test_missing_code_surfaces_as_missing_inputs(
        self,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        bse_codes,
        fake_cutoffs_family,
    ):
        incomplete_codes = {"pw": bse_codes["pw"], "p2y": bse_codes["p2y"]}
        with pytest.raises(Exception, match="yambo"):
            self._build(
                incomplete_codes,
                silicon_structure,
                kmesh,
                nscf_output_band,
                nscf_output_parameters,
                ham_output_parameters,
                bse_parameters,
                fake_cutoffs_family,
            )

    def test_roundtrips_from_dict(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        nscf_output_parameters,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            nscf_output_parameters,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        from aiida_workgraph import WorkGraph

        rebuilt = WorkGraph.from_dict(wg.to_dict())
        assert {t.name for t in rebuilt.tasks} == {t.name for t in wg.tasks}


# ----------------------------------------------------------------------
# SinglepointBSEWorkflow
# ----------------------------------------------------------------------


class TestSinglepointBSEWorkflow:
    def test_protocol_and_protocol_qe_reach_run_bse_separately(
        self, bse_full_codes, silicon_structure, kmesh, bse_parameters, fake_cutoffs_family
    ):
        """``protocol`` must reach the nested RunBse call, not get replaced by ``protocol_qe``.

        Landmine: an earlier wiring called ``RunBse(protocol=protocol_qe,
        protocol_qe=protocol_qe, ...)``, silently discarding
        ``SinglepointBSEWorkflow``'s own ``protocol`` argument for the BSE
        step. ``RunBse`` is a nested ``@task.graph`` call here, not a
        ``.build()`` -- it shows up as a single ``"bse"`` task node whose
        own ``protocol``/``protocol_qe`` input sockets carry exactly the
        values this graph passed it, without RunBse's own body running.
        """
        wg = SinglepointBSEWorkflow.build(
            codes=bse_full_codes,
            structure=silicon_structure,
            manifolds=_manifolds(),
            kpoints=kmesh,
            bse_parameters=bse_parameters,
            pseudo_family=fake_cutoffs_family.label,
            protocol="fast",
            protocol_qe="precise",
        )
        bse_task = wg.tasks["bse"]
        assert bse_task.inputs["protocol"].value == "fast"
        assert bse_task.inputs["protocol_qe"].value == "precise"

    def test_graph_composes_dfpt_and_bse(
        self, bse_full_codes, silicon_structure, kmesh, bse_parameters, fake_cutoffs_family
    ):
        wg = SinglepointBSEWorkflow.build(
            codes=bse_full_codes,
            structure=silicon_structure,
            manifolds=_manifolds(),
            kpoints=kmesh,
            bse_parameters=bse_parameters,
            pseudo_family=fake_cutoffs_family.label,
        )
        names = [t.name for t in wg.tasks]
        assert "dfpt" in names
        assert "select_channel" in names
        assert "bse" in names
        assert_graph_roundtrips(wg)

    def test_non_none_manifold_keys_are_refused(
        self, bse_full_codes, silicon_structure, kmesh, bse_parameters, fake_cutoffs_family
    ):
        with pytest.raises(NotImplementedError, match="spin='none'"):
            SinglepointBSEWorkflow.build(
                codes=bse_full_codes,
                structure=silicon_structure,
                manifolds={"up": _manifolds()["none"], "down": _manifolds()["none"]},
                kpoints=kmesh,
                bse_parameters=bse_parameters,
                pseudo_family=fake_cutoffs_family.label,
            )

    def test_molecular_structure_is_refused(
        self, bse_full_codes, kmesh, bse_parameters, fake_cutoffs_family, aiida_profile
    ):
        from aiida.orm import StructureData

        molecule = StructureData(pbc=(False, False, False))
        molecule.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
        with pytest.raises(NotImplementedError, match="periodic"):
            SinglepointBSEWorkflow.build(
                codes=bse_full_codes,
                structure=molecule,
                manifolds=_manifolds(),
                kpoints=kmesh,
                bse_parameters=bse_parameters,
                pseudo_family=fake_cutoffs_family.label,
            )

    def test_roundtrips_from_dict(
        self, bse_full_codes, silicon_structure, kmesh, bse_parameters, fake_cutoffs_family
    ):
        wg = SinglepointBSEWorkflow.build(
            codes=bse_full_codes,
            structure=silicon_structure,
            manifolds=_manifolds(),
            kpoints=kmesh,
            bse_parameters=bse_parameters,
            pseudo_family=fake_cutoffs_family.label,
        )
        from aiida_workgraph import WorkGraph

        rebuilt = WorkGraph.from_dict(wg.to_dict())
        assert {t.name for t in rebuilt.tasks} == {t.name for t in wg.tasks}
