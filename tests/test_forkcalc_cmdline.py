"""Render-level check that the koopmans-kcp fork calcs never emit -npool / -pd.

kcp.x and wann2kcp.x come from the koopmans-kcp fork, which predates QE's
global ``Modules/command_line_options.f90`` parser and reads no CLI flags at
all; the koopmans2 schema also forbids ``npool`` / ``pd`` for them. This runs
each calc's ``prepare_for_submission`` and confirms the rendered command line
stays clean while the requested (ntasks-driven) MPI resources are applied.
"""

from __future__ import annotations

import pytest


def _wann2kcp_inputs(code, resources):
    from aiida import orm

    return {
        "code": code,
        "parameters": orm.Dict(dict={"wan_mode": "ks2kcp"}),
        "metadata": {"options": {"resources": resources}},
    }


def _kcp_inputs(code, resources, structure, pseudos):
    from aiida import orm

    from aiida_koopmans.workgraphs.kcp import KcpBaseInputs, _build_dft_parameters

    base = KcpBaseInputs(
        ecutwfc=20.0,
        ecutrho=80.0,
        nspin=2,
        nelec=18,
        ntyp=len(structure.kinds),
        mt_correction=not any(structure.pbc),
        nelup=9,
        neldw=9,
        tot_magnetization=None,
    )
    return {
        "code": code,
        "structure": structure,
        "parameters": orm.Dict(dict=_build_dft_parameters(base, nbnd=10)),
        "pseudos": pseudos,
        "metadata": {"options": {"resources": resources}},
    }


@pytest.mark.parametrize(
    "entry_point, input_filename, mpiprocs",
    [
        # wann2kcp.x is serial-only (a buffer-scratch race rejects >1 rank).
        ("koopmans.wann2kcp", "aiida.wki", 1),
        # kcp.x exercises the >1-rank (ntasks) path.
        ("koopmans.kcp", "aiida.cpi", 2),
    ],
)
def test_fork_calc_cmdline_free_of_pool_and_pd(
    entry_point,
    input_filename,
    mpiprocs,
    aiida_profile,
    fixture_sandbox,
    fixture_localhost,
    aiida_local_code_factory,
    ozone_structure,
    ozone_real_pseudos,
):
    """The rendered cmdline carries no -npool / -pd; the MPI resources still land."""
    from aiida.engine.utils import instantiate_process
    from aiida.manage.manager import get_manager
    from aiida.plugins import CalculationFactory

    code = aiida_local_code_factory(executable="true", entry_point=entry_point)
    resources = {"num_machines": 1, "num_mpiprocs_per_machine": mpiprocs}
    if entry_point == "koopmans.wann2kcp":
        inputs = _wann2kcp_inputs(code, resources)
    else:
        inputs = _kcp_inputs(code, resources, ozone_structure, ozone_real_pseudos)

    process = instantiate_process(
        get_manager().get_runner(), CalculationFactory(entry_point), **inputs
    )
    calc_info = process.prepare_for_submission(fixture_sandbox)

    cmdline = calc_info.codes_info[0].cmdline_params
    assert cmdline == ["-in", input_filename]
    assert "-npool" not in cmdline
    assert "-pd" not in cmdline
    # ntasks rides the scheduler resources, never the command line.
    assert process.node.get_option("resources") == resources
