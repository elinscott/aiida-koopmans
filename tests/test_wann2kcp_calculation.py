"""Unit tests for ``Wann2kcpCalculation`` input rendering.

Exercise the pure-function helpers on the class (``_normalize_parameters``,
``_render_namelist``, ``_inject_owned_keys``) without booting an AiiDA daemon,
mirroring the style of ``test_kcp_calculation.py``.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.calculations.wann2kcp import Wann2kcpCalculation

# ----------------------------------------------------------------------
# _normalize_parameters
# ----------------------------------------------------------------------


class TestNormalizeParameters:
    def test_lowercases_keys(self):
        raw = {"Seedname": "wannier90", "WAN_MODE": "wannier2kcp"}
        got = Wann2kcpCalculation._normalize_parameters(raw)
        assert got == {"seedname": "wannier90", "wan_mode": "wannier2kcp"}

    def test_rejects_blocked_keys(self):
        for blocked in ("outdir", "prefix"):
            with pytest.raises(ValueError, match="set by the CalcJob"):
                Wann2kcpCalculation._normalize_parameters({blocked: "nope"})

    def test_rejects_unknown_keys(self):
        with pytest.raises(ValueError, match="Unknown wann2kcp parameter"):
            Wann2kcpCalculation._normalize_parameters({"not_a_key": 1})

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            Wann2kcpCalculation._normalize_parameters("seedname")

    def test_accepts_all_valid_keys(self):
        raw = {
            "seedname": "wannier90",
            "wan_mode": "wannier2kcp",
            "spin_component": "up",
            "gamma_trick": False,
            "print_rho": False,
            "wannier_plot": False,
            "wannier_plot_list": "1-4",
        }
        got = Wann2kcpCalculation._normalize_parameters(raw)
        assert got == raw


# ----------------------------------------------------------------------
# _inject_owned_keys (defaults + owned outdir)
# ----------------------------------------------------------------------


class TestInjectOwnedKeys:
    def test_injects_outdir_and_defaults(self):
        calc = Wann2kcpCalculation.__new__(Wann2kcpCalculation)
        params: dict = {}
        calc._inject_owned_keys(params)
        assert params["outdir"] == "./TMP/"
        # Matches aiida-quantumespresso's fixed pw.x prefix (the nscf scratch
        # symlinked in as TMP is always an ``aiida.save`` tree).
        assert params["prefix"] == "aiida"
        assert params["seedname"] == "wannier90"
        assert params["wan_mode"] == "wannier2kcp"

    def test_does_not_override_user_supplied_defaults(self):
        calc = Wann2kcpCalculation.__new__(Wann2kcpCalculation)
        params = {"seedname": "my_seed", "wan_mode": "ks2kcp"}
        calc._inject_owned_keys(params)
        assert params["seedname"] == "my_seed"
        assert params["wan_mode"] == "ks2kcp"
        # outdir is always owned, never user-supplied
        assert params["outdir"] == "./TMP/"


# ----------------------------------------------------------------------
# _render_namelist (Fortran value formatting + namelist name)
# ----------------------------------------------------------------------


class TestRenderNamelist:
    def test_namelist_is_inputpp(self):
        text = Wann2kcpCalculation._render_namelist({"seedname": "wannier90"})
        assert text.startswith("&INPUTPP\n")
        assert text.rstrip().endswith("/")

    def test_quoted_string_and_path(self):
        text = Wann2kcpCalculation._render_namelist(
            {"seedname": "wannier90", "outdir": "./TMP/", "wan_mode": "wannier2kcp"}
        )
        assert "seedname = 'wannier90'" in text
        assert "outdir = './TMP/'" in text
        assert "wan_mode = 'wannier2kcp'" in text

    def test_fortran_booleans(self):
        text = Wann2kcpCalculation._render_namelist({"gamma_trick": True, "print_rho": False})
        assert "gamma_trick = .true." in text
        assert "print_rho = .false." in text


# ----------------------------------------------------------------------
# End-to-end: wannier2kcp default input + linked-file set
# ----------------------------------------------------------------------


def test_wannier2kcp_full_render(
    aiida_profile,
    fixture_sandbox,
    generate_calc_job,
    fixture_localhost,
    aiida_local_code_factory,
    tmp_path_factory,
):
    """Render a wannier2kcp ``.wki`` and verify the namelist + parent symlink.

    Builds a ``Wann2kcpCalculation`` with a stand-in nscf ``parent_folder`` and
    checks that (a) the ``&inputpp`` namelist carries the owned ``outdir`` plus
    the spin component, (b) the parent ``outdir`` is symlinked into ``TMP``,
    and (c) the ``evcw*`` files are scheduled for retrieval in wannier2kcp mode.
    """
    from aiida import orm
    from aiida.common import LinkType, datastructures

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.wann2kcp")

    parent_calc = orm.CalcJobNode(computer=fixture_localhost, process_type="")
    parent_calc.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
    parent_calc.store()
    parent_root = tmp_path_factory.mktemp("w2k-parent")
    remote = orm.RemoteData(computer=fixture_localhost, remote_path=parent_root.as_posix())
    remote.base.links.add_incoming(
        parent_calc, link_type=LinkType.CREATE, link_label="remote_folder"
    )
    remote.store()

    inputs = {
        "code": code,
        "parameters": orm.Dict(dict={"spin_component": "up", "wan_mode": "wannier2kcp"}),
        "parent_folder": remote,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    calc_info = generate_calc_job(fixture_sandbox, "koopmans.wann2kcp", inputs)

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert calc_info.codes_info[0].cmdline_params == ["-in", "aiida.wki"]

    # The parent's pw.x scratch (its ``out`` subfolder) symlinked into ./TMP/,
    # so ``outdir + prefix`` resolves to ``TMP/aiida.save``.
    assert [(item[1], item[2]) for item in calc_info.remote_symlink_list] == [
        (f"{parent_root.as_posix()}/out", "TMP")
    ]

    # evcw files retrieved in wannier2kcp mode.
    retrieved = set(calc_info.retrieve_list)
    assert {"evcw.dat", "evcw1.dat", "evcw2.dat"} <= retrieved
    assert "aiida.wko" in retrieved

    with fixture_sandbox.open("aiida.wki") as handle:
        rendered = handle.read()
    assert rendered.startswith("&INPUTPP\n")
    assert "outdir = './TMP/'" in rendered
    # ``prefix`` must match aiida-quantumespresso's fixed pw.x prefix.
    assert "prefix = 'aiida'" in rendered
    assert "seedname = 'wannier90'" in rendered
    assert "spin_component = 'up'" in rendered
    assert "wan_mode = 'wannier2kcp'" in rendered


def test_wannier2kcp_stages_nnkp_chk_and_hr(
    aiida_profile,
    fixture_sandbox,
    generate_calc_job,
    aiida_local_code_factory,
):
    """Stage ``.nnkp`` / ``.chk`` / ``_hr.dat`` via the dedicated inputs.

    The ``nnkp_file`` SinglefileData lands as ``<seedname>.nnkp``; the two
    wannier90 artefacts inside ``wannier_folder`` (upstream seedname
    ``aiida``) land as ``<seedname>.chk`` / ``<seedname>_hr.dat``.
    """
    import io

    from aiida import orm

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.wann2kcp")

    nnkp = orm.SinglefileData(io.BytesIO(b"nnkp"), filename="aiida.nnkp")
    wannier_folder = orm.FolderData()
    wannier_folder.put_object_from_bytes(b"chk", "aiida.chk")
    wannier_folder.put_object_from_bytes(b"hr", "aiida_hr.dat")

    inputs = {
        "code": code,
        "parameters": orm.Dict(dict={"wan_mode": "wannier2kcp", "seedname": "aiida"}),
        "nnkp_file": nnkp,
        "wannier_folder": wannier_folder,
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    calc_info = generate_calc_job(fixture_sandbox, "koopmans.wann2kcp", inputs)

    copies = {(src, dest) for _, src, dest in calc_info.local_copy_list}
    assert copies == {
        ("aiida.nnkp", "aiida.nnkp"),
        ("aiida.chk", "aiida.chk"),
        ("aiida_hr.dat", "aiida_hr.dat"),
    }


def test_wannier2kcp_staging_respects_seednames(
    aiida_profile,
    fixture_sandbox,
    generate_calc_job,
    aiida_local_code_factory,
):
    """Destination names follow the namelist seedname; sources follow settings."""
    import io

    from aiida import orm

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.wann2kcp")

    nnkp = orm.SinglefileData(io.BytesIO(b"nnkp"), filename="pp.nnkp")
    wannier_folder = orm.FolderData()
    wannier_folder.put_object_from_bytes(b"chk", "w90.chk")
    wannier_folder.put_object_from_bytes(b"hr", "w90_hr.dat")

    inputs = {
        "code": code,
        "parameters": orm.Dict(dict={"wan_mode": "wannier2kcp"}),
        "nnkp_file": nnkp,
        "wannier_folder": wannier_folder,
        "settings": orm.Dict(dict={"wannier_source_seedname": "w90"}),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }

    calc_info = generate_calc_job(fixture_sandbox, "koopmans.wann2kcp", inputs)

    # Default namelist seedname is ``wannier90`` — destinations follow it.
    copies = {(src, dest) for _, src, dest in calc_info.local_copy_list}
    assert copies == {
        ("pp.nnkp", "wannier90.nnkp"),
        ("w90.chk", "wannier90.chk"),
        ("w90_hr.dat", "wannier90_hr.dat"),
    }


def test_ks2kcp_does_not_retrieve_evcw(
    aiida_profile,
    fixture_sandbox,
    generate_calc_job,
    fixture_localhost,
    aiida_local_code_factory,
):
    """In ``ks2kcp`` mode the ``evcw*`` files are not part of the retrieve list."""
    from aiida import orm

    code = aiida_local_code_factory(executable="true", entry_point="koopmans.wann2kcp")
    inputs = {
        "code": code,
        "parameters": orm.Dict(dict={"wan_mode": "ks2kcp"}),
        "metadata": {"options": {"resources": {"num_machines": 1}}},
    }
    calc_info = generate_calc_job(fixture_sandbox, "koopmans.wann2kcp", inputs)
    retrieved = set(calc_info.retrieve_list)
    assert "evcw.dat" not in retrieved
    assert "aiida.wko" in retrieved
    # No parent_folder => no symlinks.
    assert calc_info.remote_symlink_list == []
