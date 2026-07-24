"""Construction-level test for the RunPdos parallelization threading.

Builds the ``RunPdos`` graph (no daemon, no real execution) and checks that a
projwfc parallelization entry reaches the projwfc.x step. This guards against
the aiida-quantumespresso ``PdosWorkChain.get_builder_from_protocol`` dropping
``projwfc.settings`` (it seeds only code / parameters / metadata).
"""

from __future__ import annotations

import pytest

from aiida_koopmans.workgraphs.pdos import RunPdos


@pytest.fixture
def pdos_codes(aiida_localhost):
    """Stand-in pw / dos / projwfc codes (construction-only; never executed)."""
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
        "pw": _code("pdos-pw", "quantumespresso.pw"),
        "dos": _code("pdos-dos", "quantumespresso.dos"),
        "projwfc": _code("pdos-pjw", "quantumespresso.projwfc"),
    }


def test_projwfc_npool_and_pd_reach_the_projwfc_step(
    pdos_codes, silicon_structure, fake_cutoffs_family
):
    """The projwfc entry lands on projwfc.settings despite the workchain dropping it."""
    wg = RunPdos.build(
        codes=pdos_codes,
        structure=silicon_structure,
        pseudo_family=fake_cutoffs_family.label,
        parallelization={"projwfc": {"ntasks": 4, "npool": 2, "pd": True}},
    )
    tasks = [t for t in wg.tasks if "projwfc" in t.inputs]
    assert tasks, f"no task with a projwfc namespace among {[t.name for t in wg.tasks]}"
    projwfc = tasks[0].inputs["projwfc"]
    assert projwfc["settings"].value["cmdline"] == ["-npool", "2", "-pd", "true"]
    assert projwfc["metadata"]["options"]["resources"].value["num_mpiprocs_per_machine"] == 4
