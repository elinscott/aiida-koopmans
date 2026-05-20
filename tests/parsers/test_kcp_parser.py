"""Parser regression tests for ``KcpParser``.

Mirrors the pattern from ``aiida-quantumespresso.tests.parsers.test_cp``:
a test-name subdirectory under ``tests/parsers/fixtures/kcp/`` holds a
frozen copy of retrieved-folder contents from a known-good kcp.x run; the
parser is invoked against that folder and its ``output_parameters`` dict
is snapshotted with ``data_regression``.

The fixtures for ``tutorial_1_ozone_ki`` come from a completed legacy
``koopmans`` run at
``/home/linsco_e/code/koopmans/tutorials/tutorial_1/01-koopmans-dscf-ki/``.
File names have been adjusted to match the aiida-koopmans CalcJob's
``prefix=aiida`` (legacy used ``prefix=kc``).
"""

from __future__ import annotations

import numpy as np
from aiida import orm

from aiida_koopmans.types import Correction
from aiida_koopmans.workgraphs.kcp import KcpBaseInputs, _build_orbdep_parameters


def test_kcp_parser_tutorial_1_ozone_ki(
    aiida_profile,
    fixture_localhost,
    generate_calc_job_node,
    generate_parser,
    ozone_structure,
    ozone_real_pseudos,
    data_regression,
):
    """Pin the parsed output of the KI-correction step of tutorial_1 (ozone).

    Attaches the frozen legacy-run retrieved folder to a mock ``CalcJobNode``
    and runs ``KcpParser`` against it. Snapshot captures the
    ``output_parameters`` dict plus the shapes of the eigenvalue and lambda
    array outputs.
    """
    base = KcpBaseInputs(
        ecutwfc=65.0,
        ecutrho=260.0,
        nspin=2,
        nelec=18,
        ntyp=1,
        mt_correction=False,
        nelup=9,
        neldw=9,
        tot_magnetization=None,
    )
    ki_params = _build_orbdep_parameters(base, nbnd=10, correction=Correction.KI)
    parameters = orm.Dict(dict=ki_params)

    from aiida_koopmans.types import SpinChannel

    alphas = orm.Dict(
        dict={
            "filled": {SpinChannel.UP: [0.6] * 9, SpinChannel.DOWN: [0.6] * 9},
            "empty": {SpinChannel.UP: [0.6], SpinChannel.DOWN: [0.6]},
        }
    )

    node = generate_calc_job_node(
        entry_point_name="koopmans.kcp",
        computer=fixture_localhost,
        test_name="tutorial_1_ozone_ki",
        fixture_subdir="kcp",
        inputs={
            "structure": ozone_structure,
            "parameters": parameters,
            "alphas": alphas,
            "pseudos": ozone_real_pseudos,
        },
    )

    # The CalcJob streams Hamiltonian XMLs into ``retrieve_temporary_list``;
    # the same fixture directory stands in for that scratch folder here.
    from pathlib import Path

    fixture_dir = Path(__file__).parent / "fixtures" / "kcp" / "tutorial_1_ozone_ki"

    parser = generate_parser("koopmans.kcp")
    results, calcfunction = parser.parse_from_node(
        node,
        store_provenance=False,
        retrieved_temporary_folder=str(fixture_dir),
    )

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert "output_parameters" in results
    assert "output_eigenvalues" in results
    assert "output_lambdas" in results
    assert "output_bare_lambdas" in results

    eig = results["output_eigenvalues"].get_array("eigenvalues")
    lam = results["output_lambdas"].get_array("lambdas")
    bare = results["output_bare_lambdas"].get_array("lambdas")

    # Strip floats whose exact value is stdout-format-dependent from the
    # snapshotted output_parameters, then record array shapes instead of full
    # array contents so the snapshot stays small and diff-friendly.
    params = results["output_parameters"].get_dict()
    # ``walltime`` depends on the machine the legacy run was executed on.
    params.pop("walltime", None)

    snapshot = {
        "output_parameters": _sanitize(params),
        "eigenvalues_shape": list(eig.shape),
        "eigenvalues_has_nan": bool(np.isnan(eig).any()),
        "lambdas_shape": list(lam.shape),
        "bare_lambdas_shape": list(bare.shape),
    }
    data_regression.check(snapshot)


def _sanitize(value):
    """Recursively convert numbers to ``float``/``int`` so YAML output is stable."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        # Round to 8 sig figs — legacy .cpo stdout floats have only ~6-10
        # significant digits depending on the printf format, and a tighter
        # comparison would flake on trivial last-bit differences.
        if np.isnan(value):
            return float("nan")
        return float(f"{value:.8g}")
    return value
