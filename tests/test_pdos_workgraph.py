"""Construction-level test for the RunPdos parallelization threading.

Builds the ``RunPdos`` graph (no daemon, no real execution) and checks that a
projwfc parallelization entry reaches the projwfc.x step. This guards against
the aiida-quantumespresso ``PdosWorkChain.get_builder_from_protocol`` dropping
``projwfc.settings`` (it seeds only code / parameters / metadata).
"""

from __future__ import annotations

from aiida_quantumespresso.common.types import ElectronicType

from aiida_koopmans.workgraphs.pdos import RunPdos


def test_projwfc_npool_and_pd_reach_the_projwfc_step(
    run_pdos_codes, silicon_structure, fake_cutoffs_family
):
    """The projwfc entry lands on projwfc.settings despite the workchain dropping it."""
    wg = RunPdos.build(
        codes=run_pdos_codes,
        structure=silicon_structure,
        pseudo_family=fake_cutoffs_family.label,
        parallelization={"projwfc": {"ntasks": 4, "npool": 2, "pd": True}},
    )
    tasks = [t for t in wg.tasks if "projwfc" in t.inputs]
    assert tasks, f"no task with a projwfc namespace among {[t.name for t in wg.tasks]}"
    projwfc = tasks[0].inputs["projwfc"]
    assert projwfc["settings"].value["cmdline"] == ["-npool", "2", "-pd", "true"]
    assert projwfc["metadata"]["options"]["resources"].value["num_mpiprocs_per_machine"] == 4


def _system(task, *namespace):
    """Return the SYSTEM namelist of a built task's ``<namespace...>.parameters``."""
    node = task.inputs
    for key in namespace:
        node = node[key]
    return node["parameters"].value.get_dict()["SYSTEM"]


class TestRunPdosOccupations:
    """``RunPdos`` honours ``electronic_type`` on its scf and nscf steps.

    Like ``RunPwBands``, ``PdosWorkChain.get_builder_from_protocol`` only
    forwards ``electronic_type`` to its sub-builders via ``**kwargs``.
    """

    def test_default_insulator_fixes_both_steps(
        self, run_pdos_codes, silicon_structure, fake_cutoffs_family
    ):
        """No ``electronic_type`` given: the declared ``INSULATOR`` default fires."""
        wg = RunPdos.build(
            codes=run_pdos_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
        )
        task = wg.tasks["PdosWorkChain"]
        system = _system(task, "scf", "pw")
        assert system["occupations"] == "fixed"
        assert "smearing" not in system
        assert "degauss" not in system

    def test_metal_still_smears(self, run_pdos_codes, silicon_structure, fake_cutoffs_family):
        """A metallic run keeps the protocol's smearing — the fix is not a blanket override."""
        wg = RunPdos.build(
            codes=run_pdos_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            electronic_type=ElectronicType.METAL,
        )
        task = wg.tasks["PdosWorkChain"]
        system = _system(task, "scf", "pw")
        assert system["occupations"] == "smearing"
        assert system["degauss"] > 0
