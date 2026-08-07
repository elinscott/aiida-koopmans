"""Parser regression tests for the kcw.x parsers.

The fixture stdout files are frozen reference outputs of a 2x2x2 silicon-like
KI run (4 occupied + 4 empty Wannier functions), named with the CalcJob's
``aiida.*`` prefix.
"""

from __future__ import annotations

import numpy as np
import pytest
from aiida import orm


@pytest.fixture
def screen_parameters():
    return orm.Dict(
        {
            "CONTROL": {"kcw_at_ks": False, "read_unitary_matrix": True},
            "WANNIER": {"seedname": "aiida", "num_wann_occ": 4, "num_wann_emp": 4},
            "SCREEN": {"tr2": 1e-18, "nmix": 4, "niter": 33},
        }
    )


def test_wann2kc_parser(aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser):
    node = generate_calc_job_node(
        entry_point_name="koopmans.kcw_wann2kc",
        computer=fixture_localhost,
        test_name="default",
        fixture_subdir="kcw_wann2kc",
        input_filename="aiida.w2ki",
        output_filename="aiida.w2ko",
        inputs={"parameters": orm.Dict({"CONTROL": {}, "WANNIER": {}})},
    )
    parser = generate_parser("koopmans.kcw_wann2kc")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    params = results["output_parameters"].get_dict()
    assert params["job_done"] is True
    assert params["walltime"] == pytest.approx(0.21)


def test_screen_parser_extracts_alphas(
    aiida_profile,
    fixture_localhost,
    generate_calc_job_node,
    generate_parser,
    screen_parameters,
):
    node = generate_calc_job_node(
        entry_point_name="koopmans.kcw_screen",
        computer=fixture_localhost,
        test_name="default",
        fixture_subdir="kcw_screen",
        input_filename="aiida.ksi",
        output_filename="aiida.kso",
        inputs={"parameters": screen_parameters},
    )
    parser = generate_parser("koopmans.kcw_screen")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message

    alphas = results["alphas"].get_list()
    # 8 orbitals: 4 occupied (spread group alpha 0.14357807) + 4 empty
    # (alpha 0.09079424) -- the ``iwann*`` group-copied lines included.
    assert len(alphas) == 8
    assert alphas[0] == pytest.approx(0.14357807)
    assert alphas[4] == pytest.approx(0.09079424)

    params = results["output_parameters"].get_dict()
    assert params["job_done"] is True
    assert len(params["self_hartree"]) == 8
    # 0.35538007 Ry -> eV
    assert params["self_hartree"][0] == pytest.approx(0.35538007 * 13.605693122994, rel=1e-6)
    assert params["relaxed"][0] == pytest.approx(0.09354923)
    assert params["unrelaxed"][0] == pytest.approx(0.65155656)


def test_screen_parser_no_alphas_is_an_error(
    aiida_profile,
    fixture_localhost,
    generate_calc_job_node,
    generate_parser,
    screen_parameters,
):
    # The wann2kc stdout contains no ``relaxed`` lines: reuse it as a
    # screen output to trigger the missing-alphas exit code.
    node = generate_calc_job_node(
        entry_point_name="koopmans.kcw_screen",
        computer=fixture_localhost,
        test_name="default",
        fixture_subdir="kcw_wann2kc",
        input_filename="aiida.ksi",
        output_filename="aiida.w2ko",
        inputs={"parameters": screen_parameters},
    )
    parser = generate_parser("koopmans.kcw_screen")
    _results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.exit_status == 320


def test_ham_parser_bands_and_grid_eigenvalues(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    node = generate_calc_job_node(
        entry_point_name="koopmans.kcw_ham",
        computer=fixture_localhost,
        test_name="default",
        fixture_subdir="kcw_ham",
        input_filename="aiida.khi",
        output_filename="aiida.kho",
        inputs={
            "parameters": orm.Dict(
                {
                    "CONTROL": {},
                    "WANNIER": {},
                    "HAM": {"do_bands": True, "write_hr": True},
                }
            ),
            "alphas": orm.List(list=[0.14] * 8),
        },
    )
    parser = generate_parser("koopmans.kcw_ham")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message

    bands = results["bands"].get_bands()
    # 23 interpolated k-points along the path, 8 Wannier bands each.
    assert bands.shape == (23, 8)
    # First interpolated point (Gamma): lowest KI eigenvalue -6.1254 eV.
    assert bands[0][0] == pytest.approx(-6.1254)

    params = results["output_parameters"].get_dict()
    assert params["job_done"] is True
    # 2x2x2 grid -> 8 k-points, 8 bands each, for both KS and KI tables.
    assert len(params["ks_eigenvalues_on_grid"]) == 8
    assert len(params["ki_eigenvalues_on_grid"]) == 8
    assert all(len(row) == 8 for row in params["ki_eigenvalues_on_grid"])
    # The KS/KI "highest occupied, lowest unoccupied" summary lines must not
    # be swept into the grid tables (they would inject NaNs) ...
    assert all(len(row) == 8 for row in params["ks_eigenvalues_on_grid"])
    assert all(x == x for row in params["ks_eigenvalues_on_grid"] for x in row)
    # ... and land as scalars instead.
    assert params["ks_homo_energy"] == pytest.approx(6.5277)
    assert params["ks_lumo_energy"] == pytest.approx(7.1112)
    assert params["ki_homo_energy"] == pytest.approx(5.8913)
    assert params["ki_lumo_energy"] == pytest.approx(7.4306)
    assert params["ks_eigenvalues_on_grid"][0][0] == pytest.approx(-5.4890)
    assert params["ki_eigenvalues_on_grid"][0][0] == pytest.approx(-6.1254)


#: The path the ``kcw_ham`` fixture's stdout interpolates along: Gamma to X and
#: back, 23 points, exactly as its "KC interpolated eigenvalues at k=" lines
#: report them. The ``kpoints`` input must list the same points in the same
#: order, which is what a real run passes.
HAM_FIXTURE_KPATH = [
    [t / 11.0 * 0.5, 0.0, t / 11.0 * 0.5] for t in list(range(12)) + list(range(10, -1, -1))
]

#: The fcc silicon cell the fixture run used, in Angstrom.
SI_CELL = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]


def test_ham_parser_seeds_the_bands_from_the_k_path(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    """The ``bands`` output inherits the cell and labels of the ``kpoints`` input.

    ``set_kpoints`` sets neither, so the cell and the labels are what tell the
    two apart: rebuilding the output from the stdout alone leaves a band
    structure that plots on a crystal-coordinate axis with unnamed ticks, and
    that nothing downstream can distinguish from a mesh.
    """
    kpoints = orm.KpointsData()
    kpoints.set_cell(SI_CELL)
    kpoints.set_kpoints(HAM_FIXTURE_KPATH)
    kpoints.labels = [(0, "GAMMA"), (11, "X"), (22, "GAMMA")]

    node = generate_calc_job_node(
        entry_point_name="koopmans.kcw_ham",
        computer=fixture_localhost,
        test_name="default",
        fixture_subdir="kcw_ham",
        input_filename="aiida.khi",
        output_filename="aiida.kho",
        inputs={
            "parameters": orm.Dict({"CONTROL": {}, "WANNIER": {}, "HAM": {"do_bands": True}}),
            "alphas": orm.List(list=[0.14] * 8),
            "kpoints": kpoints,
        },
    )
    parser = generate_parser("koopmans.kcw_ham")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message

    bands = results["bands"]
    assert bands.labels == [(0, "GAMMA"), (11, "X"), (22, "GAMMA")]
    assert np.allclose(bands.cell, SI_CELL)
    # The k-points the input listed are the ones the stdout reports, to the
    # four decimals it prints them at.
    assert np.allclose(bands.get_kpoints(), HAM_FIXTURE_KPATH, atol=1e-4)
    # The eigenvalues are still the ones the stdout reported.
    assert bands.get_bands().shape == (23, 8)
    assert bands.get_bands()[0][0] == pytest.approx(-6.1254)


def test_ham_parser_without_a_k_path_falls_back_to_the_stdout(
    aiida_profile, fixture_localhost, generate_calc_job_node, generate_parser
):
    """A run given no ``kpoints`` keeps the k-points its stdout reported."""
    node = generate_calc_job_node(
        entry_point_name="koopmans.kcw_ham",
        computer=fixture_localhost,
        test_name="default",
        fixture_subdir="kcw_ham",
        input_filename="aiida.khi",
        output_filename="aiida.kho",
        inputs={
            "parameters": orm.Dict({"CONTROL": {}, "WANNIER": {}, "HAM": {"do_bands": True}}),
            "alphas": orm.List(list=[0.14] * 8),
        },
    )
    parser = generate_parser("koopmans.kcw_ham")
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished_ok, calcfunction.exit_message
    bands = results["bands"]
    assert bands.labels is None
    assert np.allclose(bands.get_kpoints(), HAM_FIXTURE_KPATH, atol=1e-4)
