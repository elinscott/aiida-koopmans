"""Unit tests for the kcw.x CalcJobs (``Wann2kcCalculation`` & friends).

Class-helper tests (no AiiDA profile) mirror ``test_kcp_calculation.py``;
the ``TestPrepareForSubmission`` cases run the full
``prepare_for_submission`` against a fake parent scratch on localhost to
pin the rendered input file, the alpharef side files, the per-file parent
symlinks, and the retrieve list.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.calculations.kcw import (
    KcwCalculation,
    KcwHamCalculation,
    KcwScreenCalculation,
    Wann2kcCalculation,
)

# ----------------------------------------------------------------------
# _normalize_parameters / _inject_owned_keys / _validate_parameters
# ----------------------------------------------------------------------


class TestNormalizeParameters:
    def test_uppercases_namelists_and_lowercases_keys(self):
        raw = {"control": {"Kcw_At_Ks": False}, "Wannier": {"SEEDNAME": "aiida"}}
        got = Wann2kcCalculation._normalize_parameters(raw)
        assert got == {"CONTROL": {"kcw_at_ks": False}, "WANNIER": {"seedname": "aiida"}}

    def test_rejects_blocked_control_keys(self):
        for blocked in ("prefix", "outdir", "calculation"):
            with pytest.raises(ValueError, match="set by the CalcJob"):
                Wann2kcCalculation._normalize_parameters({"CONTROL": {blocked: "nope"}})

    def test_rejects_mode_namelist_mismatch(self):
        # A wann2kcw run has no SCREEN or HAM namelist ...
        with pytest.raises(ValueError, match="not valid for a"):
            Wann2kcCalculation._normalize_parameters({"SCREEN": {"nmix": 4}})
        # ... a screen run accepts SCREEN but not HAM ...
        with pytest.raises(ValueError, match="not valid for a"):
            KcwScreenCalculation._normalize_parameters({"HAM": {"do_bands": True}})
        # ... and vice versa.
        with pytest.raises(ValueError, match="not valid for a"):
            KcwHamCalculation._normalize_parameters({"SCREEN": {"nmix": 4}})

    def test_rejects_non_dict_namelist(self):
        with pytest.raises(ValueError, match="must map to a dict"):
            Wann2kcCalculation._normalize_parameters({"CONTROL": "screen"})


class TestInjectOwnedKeys:
    @pytest.mark.parametrize(
        "cls,calculation",
        [
            (Wann2kcCalculation, "wann2kcw"),
            (KcwScreenCalculation, "screen"),
            (KcwHamCalculation, "ham"),
        ],
    )
    def test_injects_prefix_outdir_calculation(self, cls, calculation):
        params: dict = {}
        cls._inject_owned_keys(params)
        assert params["CONTROL"] == {
            "prefix": "aiida",
            "outdir": "./out/",
            "calculation": calculation,
        }


class TestValidateParameters:
    def _valid_params(self, cls) -> dict:
        params = {"CONTROL": {"kcw_at_ks": False, "read_unitary_matrix": True}}
        cls._inject_owned_keys(params)
        return params

    def test_accepts_valid_namelists(self):
        params = self._valid_params(KcwScreenCalculation)
        params["WANNIER"] = {"seedname": "aiida", "num_wann_occ": 4, "have_empty": True}
        params["SCREEN"] = {"tr2": 1e-18, "nmix": 4, "niter": 33, "check_spread": True}
        KcwScreenCalculation._validate_parameters(params)

    def test_rejects_unknown_key(self):
        params = self._valid_params(Wann2kcCalculation)
        params["WANNIER"] = {"not_a_kcw_keyword": 1}
        with pytest.raises(ValueError, match="Invalid ``WANNIER`` namelist"):
            Wann2kcCalculation._validate_parameters(params)

    def test_rejects_off_spec_value(self):
        params = self._valid_params(Wann2kcCalculation)
        # spin_component only admits the values 1 and 2
        params["CONTROL"]["spin_component"] = 3
        with pytest.raises(ValueError, match="Invalid ``CONTROL`` namelist"):
            Wann2kcCalculation._validate_parameters(params)


class TestRenderNamelists:
    def test_canonical_order_and_fortran_formatting(self):
        params = {
            "SCREEN": {"tr2": 1e-18, "check_spread": True},
            "WANNIER": {"seedname": "aiida", "num_wann_occ": 4},
            "CONTROL": {"calculation": "screen", "kcw_at_ks": False},
        }
        text = KcwScreenCalculation._render_namelists(params)
        order = [text.index(f"&{name}") for name in ("CONTROL", "WANNIER", "SCREEN")]
        assert order == sorted(order), f"namelists emitted out of order:\n{text}"
        assert "calculation = 'screen'" in text
        assert "kcw_at_ks = .false." in text
        assert "check_spread = .true." in text
        assert "num_wann_occ = 4" in text
        assert text.count("/\n") == 3


# ----------------------------------------------------------------------
# Full prepare_for_submission
# ----------------------------------------------------------------------


@pytest.fixture
def kcw_code(fixture_localhost):
    """Return a throwaway ``InstalledCode`` standing in for kcw.x."""
    from aiida import orm

    return orm.InstalledCode(
        label="kcw-test",
        computer=fixture_localhost,
        filepath_executable="/bin/true",
        default_calc_job_plugin="koopmans.kcw_wann2kc",
    ).store()


@pytest.fixture
def parent_scratch(fixture_localhost, tmp_path):
    """Return a fake parent ``RemoteData`` with a minimal ``out/`` tree."""
    from aiida import orm

    save = tmp_path / "out" / "aiida.save"
    save.mkdir(parents=True)
    (save / "data-file-schema.xml").write_text("<xml/>")
    (save / "charge-density.dat").write_text("rho")
    (tmp_path / "out" / "aiida.xml").write_text("<xml/>")
    (tmp_path / "out" / "kcw").mkdir()
    (tmp_path / "out" / "kcw" / "conversion.dat").write_text("kcw")
    return orm.RemoteData(computer=fixture_localhost, remote_path=str(tmp_path)).store()


@pytest.fixture
def wannier_files():
    """Return a ``wannier_files`` FolderData with occupied + empty manifold products."""
    from aiida import orm

    folder = orm.FolderData()
    for name in (
        "aiida_u.mat",
        "aiida_hr.dat",
        "aiida_centres.xyz",
        "aiida_emp_u.mat",
        "aiida_emp_u_dis.mat",
        "aiida_emp_hr.dat",
        "aiida_emp_centres.xyz",
    ):
        folder.base.repository.put_object_from_bytes(name.encode(), name)
    return folder.store()


def _control() -> dict:
    return {
        "kcw_iverbosity": 1,
        "kcw_at_ks": False,
        "read_unitary_matrix": True,
        "lrpa": False,
        "l_vcut": True,
        "spin_component": 1,
        "mp1": 2,
        "mp2": 2,
        "mp3": 2,
    }


def _wannier() -> dict:
    return {
        "seedname": "aiida",
        "check_ks": True,
        "num_wann_occ": 4,
        "num_wann_emp": 4,
        "have_empty": True,
        "has_disentangle": True,
    }


class TestPrepareForSubmission:
    def test_wann2kc(
        self,
        aiida_profile,
        fixture_sandbox,
        generate_calc_job,
        kcw_code,
        parent_scratch,
        wannier_files,
    ):
        from aiida import orm

        calc_info = generate_calc_job(
            fixture_sandbox,
            "koopmans.kcw_wann2kc",
            {
                "code": kcw_code,
                "parameters": orm.Dict({"CONTROL": _control(), "WANNIER": _wannier()}),
                "parent_folder": parent_scratch,
                "wannier_files": wannier_files,
            },
        )

        with fixture_sandbox.open("aiida.w2ki") as handle:
            content = handle.read()
        assert "calculation = 'wann2kcw'" in content
        assert "prefix = 'aiida'" in content
        assert "outdir = './out/'" in content
        assert "&WANNIER" in content
        # The DFPT chain runs on the up channel of an nspin=2 parent scratch.
        assert "spin_component = 1" in content

        # Every parent out/ file is symlinked at its own relative path.
        dests = sorted(dest for _, _, dest in calc_info.remote_symlink_list)
        assert dests == [
            "out/aiida.save/charge-density.dat",
            "out/aiida.save/data-file-schema.xml",
            "out/aiida.xml",
            "out/kcw/conversion.dat",
        ]
        # Wannier files are copied into the workdir root under their names.
        copied = sorted(name for _, _, name in calc_info.local_copy_list)
        assert "aiida_u.mat" in copied
        assert "aiida_emp_u_dis.mat" in copied
        assert calc_info.retrieve_list == ["aiida.w2ko"]

    def test_ham_card_alpharef_and_retrieve(
        self,
        aiida_profile,
        fixture_sandbox,
        generate_calc_job,
        kcw_code,
        parent_scratch,
        wannier_files,
    ):
        from aiida import orm

        kpts = orm.KpointsData()
        kpts.set_kpoints([[0.0, 0.0, 0.0], [0.25, 0.0, 0.25], [0.5, 0.0, 0.5]])

        calc_info = generate_calc_job(
            fixture_sandbox,
            "koopmans.kcw_ham",
            {
                "code": kcw_code,
                "parameters": orm.Dict(
                    {
                        "CONTROL": _control(),
                        "WANNIER": _wannier(),
                        "HAM": {"do_bands": True, "use_ws_distance": True, "write_hr": True},
                    }
                ),
                "parent_folder": parent_scratch,
                "wannier_files": wannier_files,
                "alphas": orm.List(list=[0.14, 0.14, 0.09, 0.09]),
                "kpoints": kpts,
            },
        )

        with fixture_sandbox.open("aiida.khi") as handle:
            content = handle.read()
        assert "calculation = 'ham'" in content
        assert "&HAM" in content
        assert "K_POINTS crystal_b" in content
        assert "\n3\n" in content
        assert content.count(" 0\n") >= 3  # three path points, zero weights

        with fixture_sandbox.open("file_alpharef.txt") as handle:
            alpharef = handle.read()
        assert alpharef.splitlines()[0] == "4"
        assert alpharef.splitlines()[1] == "1 0.14 1.0"
        with fixture_sandbox.open("file_alpharef_empty.txt") as handle:
            assert handle.read() == "0\n"

        assert "aiida.kcw_hr_occ.dat" in calc_info.retrieve_list
        assert "aiida.kcw_hr_emp.dat" in calc_info.retrieve_list

    def test_ham_do_bands_without_kpoints_raises(
        self,
        aiida_profile,
        fixture_sandbox,
        generate_calc_job,
        kcw_code,
        parent_scratch,
    ):
        from aiida import orm

        with pytest.raises(ValueError, match="no ``kpoints`` input"):
            generate_calc_job(
                fixture_sandbox,
                "koopmans.kcw_ham",
                {
                    "code": kcw_code,
                    "parameters": orm.Dict(
                        {"CONTROL": _control(), "WANNIER": _wannier(), "HAM": {"do_bands": True}}
                    ),
                    "parent_folder": parent_scratch,
                    "alphas": orm.List(list=[0.5]),
                },
            )

    def test_screen_namelist_rendered(
        self,
        aiida_profile,
        fixture_sandbox,
        generate_calc_job,
        kcw_code,
        parent_scratch,
        wannier_files,
    ):
        from aiida import orm

        generate_calc_job(
            fixture_sandbox,
            "koopmans.kcw_screen",
            {
                "code": kcw_code,
                "parameters": orm.Dict(
                    {
                        "CONTROL": _control(),
                        "WANNIER": _wannier(),
                        "SCREEN": {"tr2": 1e-18, "nmix": 4, "niter": 33, "eps_inf": 5.3},
                    }
                ),
                "parent_folder": parent_scratch,
                "wannier_files": wannier_files,
            },
        )

        with fixture_sandbox.open("aiida.ksi") as handle:
            content = handle.read()
        assert "calculation = 'screen'" in content
        assert "&SCREEN" in content
        assert "eps_inf" in content


def test_base_class_is_not_registered():
    """The shared base has no calculation mode and must stay unregistered."""
    assert KcwCalculation._CALCULATION == ""
