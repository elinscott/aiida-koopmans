"""Parser tests for ``Wann2kcpParser``.

Focus on the file-output contract: a completed ``wannier2kcp`` run re-emits
every retrieved ``evcw*.dat`` as a ``SinglefileData`` output so the folded
wavefunctions are explicit nodes in the provenance graph.
"""

from __future__ import annotations

from aiida import orm


def _parse(generate_calc_job_node, generate_parser, *, test_name, parameters):
    """Run ``Wann2kcpParser`` against a frozen retrieved folder."""
    node = generate_calc_job_node(
        entry_point_name="koopmans.wann2kcp",
        test_name=test_name,
        fixture_subdir="wann2kcp",
        inputs={"parameters": orm.Dict(dict=parameters)},
        input_filename="aiida.wki",
        output_filename="aiida.wko",
    )
    parser = generate_parser("koopmans.wann2kcp")
    return parser.parse_from_node(node, store_provenance=False)


def test_spinless_run_emits_both_evcw_files(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    """A spinless wannier2kcp run yields ``evcw1`` + ``evcw2`` SinglefileData."""
    results, calcfunction = _parse(
        generate_calc_job_node,
        generate_parser,
        test_name="wannier2kcp_spinless",
        parameters={"wan_mode": "wannier2kcp"},
    )
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert results["output_parameters"]["job_done"] is True
    assert isinstance(results["evcw1"], orm.SinglefileData)
    assert isinstance(results["evcw2"], orm.SinglefileData)
    assert "evcw" not in results
    assert results["evcw1"].get_content(mode="rb") == b"wf1"
    assert results["evcw1"].filename == "evcw1.dat"


def test_spin_resolved_run_emits_single_evcw_file(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    """A spin-resolved run (``spin_component`` set) yields the single ``evcw``."""
    results, calcfunction = _parse(
        generate_calc_job_node,
        generate_parser,
        test_name="wannier2kcp_spin_up",
        parameters={"wan_mode": "wannier2kcp", "spin_component": "up"},
    )
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert isinstance(results["evcw"], orm.SinglefileData)
    assert "evcw1" not in results
    assert "evcw2" not in results


def test_missing_evcw_files_is_an_error(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser, tmp_path
):
    """A completed wannier2kcp run with no wavefunctions exits with 320."""
    import shutil
    from pathlib import Path

    from aiida.common import LinkType

    # Build a fixture folder holding only the stdout.
    src = Path(__file__).parent / "fixtures" / "wann2kcp" / "wannier2kcp_spinless" / "aiida.wko"
    fixture = tmp_path / "wann2kcp" / "no_evcw"
    fixture.mkdir(parents=True)
    shutil.copy(src, fixture / "aiida.wko")

    node = generate_calc_job_node(
        entry_point_name="koopmans.wann2kcp",
        inputs={"parameters": orm.Dict(dict={"wan_mode": "wannier2kcp"})},
        input_filename="aiida.wki",
        output_filename="aiida.wko",
    )
    retrieved = orm.FolderData()
    retrieved.base.repository.put_object_from_tree(str(fixture))
    retrieved.base.links.add_incoming(node, link_type=LinkType.CREATE, link_label="retrieved")
    retrieved.store()

    parser = generate_parser("koopmans.wann2kcp")
    _, calcfunction = parser.parse_from_node(node, store_provenance=False)
    assert calcfunction.exit_status == 320


def test_ks2kcp_mode_emits_no_wavefunctions(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    """``ks2kcp`` mode is exempt from the evcw contract."""
    results, calcfunction = _parse(
        generate_calc_job_node,
        generate_parser,
        test_name="wannier2kcp_spinless",
        parameters={"wan_mode": "ks2kcp"},
    )
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert "evcw1" not in results
    assert "evcw2" not in results
