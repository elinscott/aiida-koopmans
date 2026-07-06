"""Parser tests for ``MergeEvcParser``.

Focus on the file-output contract: the merged wavefunction is re-emitted as
the ``merged_file`` ``SinglefileData`` output so the downstream kcp.x staging
consumes an explicit node rather than a filename inside a remote folder.
"""

from __future__ import annotations

from aiida import orm


def test_merged_file_is_emitted(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    """A successful merge yields the ``merged_file`` SinglefileData."""
    node = generate_calc_job_node(
        entry_point_name="koopmans.merge_evc",
        test_name="merged_occupied",
        fixture_subdir="merge_evc",
        inputs={"dest_filename": orm.Str("evc_occupied1.dat")},
        output_filename="aiida.out",
    )
    parser = generate_parser("koopmans.merge_evc")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert results["output_parameters"]["merged"] is True
    merged = results["merged_file"]
    assert isinstance(merged, orm.SinglefileData)
    assert merged.filename == "evc_occupied1.dat"
    assert merged.get_content(mode="rb") == b"merged"


def test_missing_merged_file_is_an_error(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    """A merge whose output file never landed exits with ``ERROR_OUTPUT_FILE_MISSING``."""
    node = generate_calc_job_node(
        entry_point_name="koopmans.merge_evc",
        test_name="merged_occupied",
        fixture_subdir="merge_evc",
        inputs={"dest_filename": orm.Str("evc_occupied2.dat")},
        output_filename="aiida.out",
    )
    parser = generate_parser("koopmans.merge_evc")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.exit_status == 302
    assert "merged_file" not in results
