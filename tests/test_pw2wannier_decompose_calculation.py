"""Tests for the ``Pw2wannierDecomposeCalculation`` input preparation."""

from __future__ import annotations

import io

import pytest

from aiida_koopmans.calculations.pw2wannier_decompose import Pw2wannierDecomposeCalculation


@pytest.fixture
def register_decompose_entry_points(entry_points):
    """Register the (not-yet-installed) decompose calc/parser entry points.

    The worktree adds new entry points that the editable install in the shared
    venv does not yet expose, so the AiiDA plugin factories cannot find them.
    aiida-core's ``entry_points`` fixture registers them for the test session.
    """
    from aiida_koopmans.parsers.pw2wannier_decompose import Pw2wannierDecomposeParser

    entry_points.add(
        Pw2wannierDecomposeCalculation,
        "aiida.calculations:koopmans.pw2wannier_decompose",
    )
    entry_points.add(
        Pw2wannierDecomposeParser,
        "aiida.parsers:koopmans.pw2wannier_decompose",
    )


class TestNormalizeParameters:
    """``_normalize_parameters`` key handling."""

    def test_lowercases_keys(self):
        out = Pw2wannierDecomposeCalculation._normalize_parameters({"Decompose_N_Max": 4})
        assert out == {"decompose_n_max": 4}

    def test_rejects_blocked_keys(self):
        for blocked in ("outdir", "prefix", "seedname", "wan_mode", "decompose_centres_file"):
            with pytest.raises(ValueError, match="set by the CalcJob"):
                Pw2wannierDecomposeCalculation._normalize_parameters({blocked: "x"})

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown pw2wannier90 decompose parameter"):
            Pw2wannierDecomposeCalculation._normalize_parameters({"decompose_r_cut": 4.0})

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            Pw2wannierDecomposeCalculation._normalize_parameters(["decompose_n_max"])

    def test_accepts_all_valid_decompose_keys(self):
        params = {
            "decompose_n_max": 4,
            "decompose_l_max": 4,
            "decompose_r_min": 0.5,
            "decompose_r_max": 4.0,
        }
        assert Pw2wannierDecomposeCalculation._normalize_parameters(params) == params


class TestInjectOwnedKeys:
    """``_inject_owned_keys`` defaults and owned-key injection."""

    @staticmethod
    def _stub(has_centres_file: bool):
        """Build a duck-typed stand-in with the class constants + an ``inputs`` map.

        ``CalcJob.inputs`` is a data-descriptor property, so a real instance
        cannot have its ``inputs`` shadowed; calling the method unbound against
        a ``SimpleNamespace`` avoids constructing a full process.
        """
        from types import SimpleNamespace

        return SimpleNamespace(
            inputs={"centres_file": object()} if has_centres_file else {},
            _DEFAULT_OUTDIR=Pw2wannierDecomposeCalculation._DEFAULT_OUTDIR,
            _DEFAULT_PREFIX=Pw2wannierDecomposeCalculation._DEFAULT_PREFIX,
            _DEFAULT_SEEDNAME=Pw2wannierDecomposeCalculation._DEFAULT_SEEDNAME,
            _CENTRES_FILE=Pw2wannierDecomposeCalculation._CENTRES_FILE,
            _DEFAULTS=Pw2wannierDecomposeCalculation._DEFAULTS,
        )

    def test_injects_owned_keys_and_defaults(self):
        params: dict = {}
        Pw2wannierDecomposeCalculation._inject_owned_keys(self._stub(False), params)
        assert params["outdir"] == "./TMP/"
        assert params["prefix"] == "aiida"
        assert params["seedname"] == "aiida"
        assert params["wan_mode"] == "decompose"
        assert params["decompose_n_max"] == 4
        assert params["decompose_l_max"] == 4
        assert params["decompose_r_min"] == 0.5
        assert params["decompose_r_max"] == 4.0
        # No centres file input -> no group-density channel requested.
        assert "decompose_centres_file" not in params

    def test_sets_centres_file_when_input_present(self):
        params: dict = {}
        Pw2wannierDecomposeCalculation._inject_owned_keys(self._stub(True), params)
        assert params["decompose_centres_file"] == "gc_centres.dat"

    def test_does_not_override_user_supplied_defaults(self):
        params = {"decompose_n_max": 8}
        Pw2wannierDecomposeCalculation._inject_owned_keys(self._stub(False), params)
        assert params["decompose_n_max"] == 8


class TestRenderNamelist:
    """``_render_namelist`` produces a well-formed ``&INPUTPP`` block."""

    def test_namelist_is_inputpp(self):
        rendered = Pw2wannierDecomposeCalculation._render_namelist({"wan_mode": "decompose"})
        assert rendered.startswith("&INPUTPP\n")
        assert rendered.rstrip().endswith("/")

    def test_quoted_strings_and_numbers(self):
        rendered = Pw2wannierDecomposeCalculation._render_namelist(
            {"seedname": "aiida", "decompose_n_max": 4, "decompose_r_min": 0.5}
        )
        assert "seedname = 'aiida'" in rendered
        assert "decompose_n_max = 4" in rendered
        # Reals render in Fortran double form via convert_input_to_namelist_entry.
        assert "decompose_r_min =" in rendered
        assert "5.0000000000d-01" in rendered


@pytest.fixture
def _decompose_inputs(fixture_localhost, aiida_local_code_factory, tmp_path_factory):
    """Build the common inputs for a full decompose render (code + parent + files)."""
    from aiida import orm
    from aiida.common import LinkType

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.pw2wannier_decompose")

    parent_calc = orm.CalcJobNode(computer=fixture_localhost, process_type="")
    parent_calc.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
    parent_calc.store()
    parent_root = tmp_path_factory.mktemp("dec-parent")
    remote = orm.RemoteData(computer=fixture_localhost, remote_path=parent_root.as_posix())
    remote.base.links.add_incoming(
        parent_calc, link_type=LinkType.CREATE, link_label="remote_folder"
    )
    remote.store()

    u_mat = orm.SinglefileData(io.BytesIO(b"u matrix"), filename="anything_u.mat")
    centres = orm.SinglefileData(io.BytesIO(b"2\n\nX 0 0 0\nX 1 1 1\n"), filename="c.xyz")
    return code, remote, parent_root, u_mat, centres


def test_full_render_stages_products_and_retrieves_globs(
    aiida_profile,
    register_decompose_entry_points,
    fixture_sandbox,
    generate_calc_job,
    _decompose_inputs,
):
    """Render a decompose input and verify staging, symlink, and retrieval globs."""
    from aiida import orm
    from aiida.common import datastructures

    code, remote, parent_root, u_mat, centres = _decompose_inputs
    gc = orm.SinglefileData(io.BytesIO(b"0 0 0\n1 1 1\n"), filename="gc.dat")

    inputs = {
        "code": code,
        "parameters": orm.Dict(dict={"decompose_n_max": 4, "decompose_l_max": 4}),
        "parent_folder": remote,
        "u_mat": u_mat,
        "centres_xyz": centres,
        "centres_file": gc,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    calc_info = generate_calc_job(fixture_sandbox, "koopmans.pw2wannier_decompose", inputs)

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert calc_info.codes_info[0].cmdline_params == ["-in", "aiida.decompose.in"]

    # Only the parent ``.save`` is symlinked into a real per-calc ``TMP``.
    assert [(item[1], item[2]) for item in calc_info.remote_symlink_list] == [
        (f"{parent_root.as_posix()}/out/aiida.save", "TMP/aiida.save")
    ]

    # Wannier products staged under seedname-derived destinations.
    destinations = {item[2] for item in calc_info.local_copy_list}
    assert {"aiida_u.mat", "aiida_centres.xyz", "gc_centres.dat"} <= destinations
    # No u_dis input -> not staged.
    assert "aiida_u_dis.mat" not in destinations

    # Coefficient/power files retrieved by glob; stdout retrieved by name.
    assert "aiida.decompose.out" in calc_info.retrieve_list
    assert ["aiida_*.coeff", ".", 0] in calc_info.retrieve_list
    assert ["aiida_*.power", ".", 0] in calc_info.retrieve_list

    with fixture_sandbox.open("aiida.decompose.in") as handle:
        rendered = handle.read()
    assert "wan_mode = 'decompose'" in rendered
    assert "outdir = './TMP/'" in rendered
    assert "prefix = 'aiida'" in rendered
    assert "seedname = 'aiida'" in rendered
    assert "decompose_centres_file = 'gc_centres.dat'" in rendered


def test_full_render_without_centres_file_omits_group_channel(
    aiida_profile,
    register_decompose_entry_points,
    fixture_sandbox,
    generate_calc_job,
    _decompose_inputs,
):
    """Without a ``centres_file`` input, no group-density channel is requested."""
    from aiida import orm

    code, remote, _parent_root, u_mat, centres = _decompose_inputs
    u_dis = orm.SinglefileData(io.BytesIO(b"u dis"), filename="d.mat")

    inputs = {
        "code": code,
        "parent_folder": remote,
        "u_mat": u_mat,
        "u_dis_mat": u_dis,
        "centres_xyz": centres,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    calc_info = generate_calc_job(fixture_sandbox, "koopmans.pw2wannier_decompose", inputs)

    destinations = {item[2] for item in calc_info.local_copy_list}
    # Disentanglement matrix staged when supplied.
    assert "aiida_u_dis.mat" in destinations

    with fixture_sandbox.open("aiida.decompose.in") as handle:
        rendered = handle.read()
    assert "decompose_centres_file" not in rendered
