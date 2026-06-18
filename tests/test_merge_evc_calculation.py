"""Unit tests for ``MergeEvcCalculation`` command-line assembly.

merge_evc.x takes no namelist, so the contract under test is the exact
command-line shape ``-nr <prod(kgrid)> -i input_0.dat ... -o <dest>`` and the
``input_{i}.dat`` symlink naming. The ``_build_cmdline`` helper is a
staticmethod, so the core cases run without an AiiDA daemon.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.calculations.merge_evc import MergeEvcCalculation

# ----------------------------------------------------------------------
# _build_cmdline (pure function)
# ----------------------------------------------------------------------


class TestBuildCmdline:
    @pytest.mark.parametrize(
        ("kgrid", "n_files", "dest", "expected_nr"),
        [
            ([2, 2, 2], 2, "evcw.dat", 8),
            ([4, 4, 1], 3, "evcw1.dat", 16),
            ([1, 1, 1], 1, "evcw2.dat", 1),
            ([3, 2, 2], 4, "evcw.dat", 12),
        ],
    )
    def test_exact_command_line(self, kgrid, n_files, dest, expected_nr):
        input_names = [f"input_{i}.dat" for i in range(n_files)]
        params = MergeEvcCalculation._build_cmdline(kgrid, input_names, dest)

        expected = ["-nr", str(expected_nr)]
        for name in input_names:
            expected += ["-i", name]
        expected += ["-o", dest]
        assert params == expected

    def test_nr_is_grid_product(self):
        params = MergeEvcCalculation._build_cmdline([3, 4, 5], ["input_0.dat"], "evcw.dat")
        assert params[0] == "-nr"
        assert params[1] == "60"

    def test_input_order_is_index_order(self):
        names = ["input_0.dat", "input_1.dat", "input_2.dat"]
        params = MergeEvcCalculation._build_cmdline([1, 1, 1], names, "evcw.dat")
        # Extract the -i operands in order.
        operands = [params[i + 1] for i, tok in enumerate(params) if tok == "-i"]
        assert operands == names


# ----------------------------------------------------------------------
# End-to-end: command line + symlink staging
# ----------------------------------------------------------------------


def test_merge_evc_full_calc_info(
    aiida_profile,
    fixture_sandbox,
    generate_calc_job,
    fixture_localhost,
    aiida_local_code_factory,
    tmp_path_factory,
):
    """Assemble a full ``MergeEvcCalculation`` and check the command + symlinks.

    Two stand-in source RemoteData folders are merged into ``evcw.dat``; the
    test asserts the ``-nr 8 -i input_0.dat -i input_1.dat -o evcw.dat`` command
    and that each source is symlinked as ``input_{i}.dat`` in sorted-key order.
    """
    from aiida import orm
    from aiida.common import LinkType, datastructures

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.merge_evc")

    def _make_remote(label):
        calc = orm.CalcJobNode(computer=fixture_localhost, process_type="")
        calc.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        calc.store()
        root = tmp_path_factory.mktemp(label)
        remote = orm.RemoteData(computer=fixture_localhost, remote_path=root.as_posix())
        remote.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label="remote_folder")
        remote.store()
        return remote

    inputs = {
        "code": code,
        "kgrid": orm.List(list=[2, 2, 2]),
        "dest_filename": orm.Str("evcw.dat"),
        "source_files": {
            "block_0": _make_remote("merge-src-0"),
            "block_1": _make_remote("merge-src-1"),
        },
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    calc_info = generate_calc_job(fixture_sandbox, "koopmans.merge_evc", inputs)

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert calc_info.codes_info[0].cmdline_params == [
        "-nr",
        "8",
        "-i",
        "input_0.dat",
        "-i",
        "input_1.dat",
        "-o",
        "evcw.dat",
    ]

    # Each source symlinked as input_{i}.dat in sorted-key order.
    dests = [item[2] for item in calc_info.remote_symlink_list]
    assert dests == ["input_0.dat", "input_1.dat"]
    # Default source filename equals dest filename.
    sources = [item[1] for item in calc_info.remote_symlink_list]
    assert all(s.endswith("/evcw.dat") for s in sources)

    # Merged output + stdout retrieved.
    assert "evcw.dat" in calc_info.retrieve_list
    assert "aiida.out" in calc_info.retrieve_list


def test_merge_evc_source_filenames_override(
    aiida_profile,
    fixture_sandbox,
    generate_calc_job,
    fixture_localhost,
    aiida_local_code_factory,
    tmp_path_factory,
):
    """``settings['source_filenames']`` selects a per-source file name."""
    from aiida import orm
    from aiida.common import LinkType

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.merge_evc")

    calc = orm.CalcJobNode(computer=fixture_localhost, process_type="")
    calc.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
    calc.store()
    root = tmp_path_factory.mktemp("merge-src")
    remote = orm.RemoteData(computer=fixture_localhost, remote_path=root.as_posix())
    remote.base.links.add_incoming(calc, link_type=LinkType.CREATE, link_label="remote_folder")
    remote.store()

    inputs = {
        "code": code,
        "kgrid": orm.List(list=[1, 1, 1]),
        "dest_filename": orm.Str("evcw1.dat"),
        "source_files": {"block_0": remote},
        "settings": orm.Dict(dict={"source_filenames": {"block_0": "evcw.dat"}}),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    calc_info = generate_calc_job(fixture_sandbox, "koopmans.merge_evc", inputs)
    sources = [item[1] for item in calc_info.remote_symlink_list]
    assert sources[0].endswith("/evcw.dat")
    assert calc_info.codes_info[0].cmdline_params[-1] == "evcw1.dat"
