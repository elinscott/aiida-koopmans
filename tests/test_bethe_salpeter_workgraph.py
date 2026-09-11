"""Construction-level tests for RunBetheSalpeter and SinglepointBetheSalpeterWorkflow.

Build the graphs (no daemon, no real yambo/QE execution) and introspect the
task list / wiring, mirroring the style of ``test_block_wannierize.py``.
``k2y`` and ``aiida_yambo`` are regular ``aiida-koopmans`` dependencies, but
the plain canonical venv the rest of the suite runs against has not synced
them in (see ``test_bethe_salpeter_calcfunction.py``): ``pytest.importorskip``
keeps this module a clean skip there, not a collection error.
``RunBetheSalpeter`` needs ``aiida_yambo`` itself
(``WorkflowFactory('yambo.yambo.yambowf')``), on top of the ``k2y`` import
``workgraphs/bethe_salpeter.py`` already makes at module scope.

``fake_cutoffs_family`` (not a bare ``"SSSP/..."`` string) is required
here: unlike the block-wannierize / DFPT graphs, whose PW steps go through
``Wannier90WorkChain.get_scf_nscf_builders_from_protocol``,
:func:`~aiida_koopmans.workgraphs.bethe_salpeter.RunBetheSalpeter` builds its scf/nscf steps
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

from aiida_koopmans.workgraphs.bethe_salpeter import (
    RunBetheSalpeter,
    SinglepointBetheSalpeterWorkflow,
)
from tests.fixtures import assert_graph_roundtrips, explicit_block

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def bse_codes(aiida_localhost):
    """Return a codes dict of stand-in nodes for :func:`RunBetheSalpeter` (pw, p2y, yambo)."""
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
    """Return the combined codes namespace :func:`SinglepointBetheSalpeterWorkflow` needs."""
    return {**dfpt_codes, "p2y": bse_codes["p2y"], "yambo": bse_codes["yambo"]}


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


def _cutoff_overrides(ecutwfc=48.0, ecutrho=192.0):
    """Return the ``overrides`` shape carrying both pw cutoffs, as ``_pw_cutoffs_from`` reads it."""
    return {"scf": {"pw": {"parameters": {"SYSTEM": {"ecutwfc": ecutwfc, "ecutrho": ecutrho}}}}}


# ----------------------------------------------------------------------
# RunBetheSalpeter
# ----------------------------------------------------------------------


class TestRunBetheSalpeterGraphBuild:
    def _build(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        ham_output_parameters,
        bse_parameters,
        cutoffs_family,
        ecutwfc=48.0,
        ecutrho=192.0,
        **extra,
    ):
        return RunBetheSalpeter.build(
            codes=bse_codes,
            structure=silicon_structure,
            kpoints=kmesh,
            nscf_output_band=nscf_output_band,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            ham_output_parameters=ham_output_parameters,
            bse_parameters=bse_parameters,
            pseudo_family=cutoffs_family.label,
            **extra,
        )

    def test_graph_builds_init_qp_and_bse_tasks(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
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
        ham_output_parameters,
        bse_parameters,
        fake_bethe_salpeter_cutoffs_family,
    ):
        """An ecutwfc-only override would leave ecutrho at the family's own recommendation.

        ``PwBaseWorkChain.get_builder_from_protocol`` applies overrides on
        top of the pseudo family's recommendation, and only skips that
        recommendation when both ``ecutwfc`` and ``ecutrho`` are present in
        the override together -- so both must reach every scf/nscf SYSTEM
        namelist ``RunBetheSalpeter`` builds here (``_build``'s own 48 / 192 Ry
        defaults), not just ``ecutwfc``.

        ``fake_bethe_salpeter_cutoffs_family`` (50 / 300 Ry) rather than the plain
        ``fake_cutoffs_family`` (30 / 240 eV, ~2.2 / 17.6 Ry) is required to
        catch a dropped ``ecutrho``: ``YamboWorkflow.get_builder_from_protocol``
        itself floors ``ecutrho`` at ``4 * ecutwfc`` (192 here), so a family
        recommendation below that floor would land on the same 192 whether
        or not the override's own ``ecutrho`` reached the builder -- only a
        family recommendation above the floor (this fixture's 300) makes a
        dropped override visible.
        """
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            ham_output_parameters,
            bse_parameters,
            fake_bethe_salpeter_cutoffs_family,
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
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
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
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        """``parallelization['yambo']['ntasks']`` reaches both calc's resources.

        The BSE step alone also gets a k-point-only ``BS_CPU``/``BS_ROLEs``
        split (see :func:`~aiida_koopmans.workgraphs.bethe_salpeter._bse_mpi_roles`) --
        yambo_init runs with ``INITIALISE=True`` and never reads those keys.
        """
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
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

    def test_parallelization_pw_reaches_every_scf_nscf_pw_namespace(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        """``parallelization['pw']`` must reach both steps' own scf/nscf pw.x runs.

        The init and BSE steps each run their own fresh scf/nscf, pw.x
        calculations like any other -- a caller's ``pw`` entry must not be
        silently dropped just because only the yambo calc's own resources
        were ever wired.
        """
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
            parallelization={"pw": {"ntasks": 3, "npool": 2}},
        )
        for task_name in ("yambo_init", "bse"):
            for step in ("scf", "nscf"):
                pw_inputs = wg.tasks[task_name].inputs[step]["pw"]
                resources = pw_inputs["metadata"]["options"]["resources"].value
                assert resources["num_mpiprocs_per_machine"] == 3
                cmdline = pw_inputs["settings"].value.get_dict()["cmdline"]
                assert cmdline == ["-npool", "2"]

    def test_init_and_bse_nscf_nbnd_agree(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        ham_output_parameters,
        fake_cutoffs_family,
    ):
        """The init step's own nscf must be sized for the same bands the BSE step needs.

        ``YamboWorkflow.get_builder_from_protocol`` sizes each step's own
        nscf ``nbnd`` off its own runcard's requested band range: absent a
        caller override, that range is a protocol-computed default (300
        bands for this fixture's silicon/pseudo-family/moderate-protocol
        combination, via ``GbndRnge``, a GW-only leftover key
        ``YamboRestart.get_builder_from_protocol`` always sets and never
        drops for a ``'bse_*'`` protocol -- see
        :data:`~aiida_koopmans.workgraphs.bethe_salpeter._GW_ONLY_RUNCARD_KEYS`). A
        ``BndsRnXs`` request *above* that default only reaches the BSE
        step's own override, not the init step's -- exercise that gap with
        a ``BndsRnXs`` upper bound past 300, matching the default's own
        magnitude: a mismatch between the init and BSE steps' nscf ``nbnd``
        makes ``YamboWorkflow`` redo the BSE step's nscf and p2y at run
        time, against a fresh SAVE the already-built quasiparticle
        database was never made from.
        """
        bse_parameters = {
            "arguments": ["rim_cut"],
            "variables": {"BndsRnXs": [[1, 500], ""]},
        }
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        init_nbnd = (
            wg.tasks["yambo_init"]
            .inputs["nscf"]["pw"]["parameters"]
            .value.get_dict()["SYSTEM"]["nbnd"]
        )
        bse_nbnd = (
            wg.tasks["bse"].inputs["nscf"]["pw"]["parameters"].value.get_dict()["SYSTEM"]["nbnd"]
        )
        assert init_nbnd == bse_nbnd == 500

    def test_bse_variables_carry_no_gw_only_keys(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        """The BSE runcard must not carry protocol-derived GW-only leftover keys.

        ``YamboRestart.get_builder_from_protocol`` computes ``GbndRnge``
        from the pre-rename ``BndsRnXp`` before its own BSE branch pops
        that key, and inherits ``GTermKind``/``FFTGvecs`` from its
        protocol's ``default_inputs`` regardless of calc type -- none of
        the three mean anything to a BSE run.
        """
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        bse_variables = (
            wg.tasks["bse"].inputs["yres"]["yambo"]["parameters"].value.get_dict()["variables"]
        )
        for gw_only_key in ("GbndRnge", "FFTGvecs", "GTermKind"):
            assert gw_only_key not in bse_variables

    def test_pki_eigenvalues_are_refused(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        """``eigenvalues='pki'`` has no producing parser yet (aiida-koopmans#134)."""
        with pytest.raises(NotImplementedError, match="pki"):
            self._build(
                bse_codes,
                silicon_structure,
                kmesh,
                nscf_output_band,
                ham_output_parameters,
                bse_parameters,
                fake_cutoffs_family,
                eigenvalues="pki",
            )

    @pytest.mark.parametrize(
        "parallelization",
        [None, {"yambo": {"ntasks": 1}}],
        ids=["no-parallelization", "explicit-single-rank"],
    )
    def test_single_rank_omits_bs_roles(
        self,
        bse_codes,
        silicon_structure,
        kmesh,
        nscf_output_band,
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
        parallelization,
    ):
        """A ``yambo`` entry present but at one rank must omit the MPI-role split too.

        Not just an absent ``parallelization`` altogether -- ``ntasks: 1``
        stated explicitly takes a different path through
        :func:`~aiida_koopmans.workgraphs.bethe_salpeter._bse_mpi_roles` (a present but
        falsy-after-int-cast rank count, not a missing entry) and must reach
        the same result.
        """
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
            parallelization=parallelization,
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
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
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
                ham_output_parameters,
                bse_parameters,
                fake_cutoffs_family,
            )

    def test_missing_code_surfaces_as_missing_inputs(
        self,
        silicon_structure,
        kmesh,
        nscf_output_band,
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
        ham_output_parameters,
        bse_parameters,
        fake_cutoffs_family,
    ):
        wg = self._build(
            bse_codes,
            silicon_structure,
            kmesh,
            nscf_output_band,
            ham_output_parameters,
            bse_parameters,
            fake_cutoffs_family,
        )
        from aiida_workgraph import WorkGraph

        rebuilt = WorkGraph.from_dict(wg.to_dict())
        assert {t.name for t in rebuilt.tasks} == {t.name for t in wg.tasks}


# ----------------------------------------------------------------------
# SinglepointBetheSalpeterWorkflow
# ----------------------------------------------------------------------


class TestSinglepointBetheSalpeterWorkflow:
    def test_protocol_reaches_dfpt_and_bse_at_one_value(
        self, bse_full_codes, silicon_structure, kmesh, bse_parameters, fake_cutoffs_family
    ):
        """A single ``protocol`` must reach the nested DFPT and BSE calls unchanged.

        There is no reason for the QE steps :func:`RunBetheSalpeter` runs to
        differ in precision from the DFPT chain's own ground state, so
        ``SinglepointBetheSalpeterWorkflow`` takes one ``protocol`` and drives
        both the nested ``dfpt`` call and the nested ``bse`` call with it.
        Both are nested ``@task.graph`` calls here, not ``.build()`` calls --
        each shows up as a single task node whose own ``protocol`` input
        socket carries exactly the value this graph passed it, without
        either nested graph's own body running.
        """
        wg = SinglepointBetheSalpeterWorkflow.build(
            codes=bse_full_codes,
            structure=silicon_structure,
            manifolds=_manifolds(),
            kpoints=kmesh,
            bse_parameters=bse_parameters,
            pseudo_family=fake_cutoffs_family.label,
            overrides=_cutoff_overrides(),
            protocol="fast",
        )
        assert wg.tasks["dfpt"].inputs["protocol"].value == "fast"
        bse_task = wg.tasks["bse"]
        assert bse_task.inputs["protocol"].value == "fast"
        assert "protocol_qe" not in bse_task.inputs

    def test_graph_composes_dfpt_and_bse(
        self, bse_full_codes, silicon_structure, kmesh, bse_parameters, fake_cutoffs_family
    ):
        wg = SinglepointBetheSalpeterWorkflow.build(
            codes=bse_full_codes,
            structure=silicon_structure,
            manifolds=_manifolds(),
            kpoints=kmesh,
            bse_parameters=bse_parameters,
            pseudo_family=fake_cutoffs_family.label,
            overrides=_cutoff_overrides(),
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
            SinglepointBetheSalpeterWorkflow.build(
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
            SinglepointBetheSalpeterWorkflow.build(
                codes=bse_full_codes,
                structure=molecule,
                manifolds=_manifolds(),
                kpoints=kmesh,
                bse_parameters=bse_parameters,
                pseudo_family=fake_cutoffs_family.label,
            )

    def test_missing_cutoffs_in_overrides_is_refused(
        self, bse_full_codes, silicon_structure, kmesh, bse_parameters, fake_cutoffs_family
    ):
        """With neither cutoff in ``overrides``, the BSE step's own cutoffs cannot be built."""
        with pytest.raises(ValueError, match=r"ecutwfc.*ecutrho|ecutrho.*ecutwfc"):
            SinglepointBetheSalpeterWorkflow.build(
                codes=bse_full_codes,
                structure=silicon_structure,
                manifolds=_manifolds(),
                kpoints=kmesh,
                bse_parameters=bse_parameters,
                pseudo_family=fake_cutoffs_family.label,
            )

    def test_roundtrips_from_dict(
        self, bse_full_codes, silicon_structure, kmesh, bse_parameters, fake_cutoffs_family
    ):
        wg = SinglepointBetheSalpeterWorkflow.build(
            codes=bse_full_codes,
            structure=silicon_structure,
            manifolds=_manifolds(),
            kpoints=kmesh,
            bse_parameters=bse_parameters,
            pseudo_family=fake_cutoffs_family.label,
            overrides=_cutoff_overrides(),
        )
        from aiida_workgraph import WorkGraph

        rebuilt = WorkGraph.from_dict(wg.to_dict())
        assert {t.name for t in rebuilt.tasks} == {t.name for t in wg.tasks}
