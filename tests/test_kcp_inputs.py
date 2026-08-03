"""Unit tests for the kcp.x input assembler (``calculations/kcp_inputs.py``).

Covers the FFT-dimension arithmetic and the ``SYSTEM.nr{1,2,3}b`` box-grid
autogeneration in isolation from any CalcJob; ``build_kcp_inputs`` itself is
exercised by the workgraph tests that build real ``KcpStep`` inputs.
"""

from __future__ import annotations

from aiida_koopmans.calculations.kcp_inputs import (
    _autogenerate_nrb,
    _fft_dimension_allowed,
    _good_fft,
    build_kcp_inputs,
)


class TestFftDimensionAllowed:
    def test_rejects_dimensions_below_one(self):
        assert _fft_dimension_allowed(0) is False

    def test_accepts_products_of_two_three_five(self):
        # 24 = 2^3 * 3
        assert _fft_dimension_allowed(24) is True

    def test_rejects_a_prime_factor_of_seven_or_eleven(self):
        assert _fft_dimension_allowed(7) is False
        assert _fft_dimension_allowed(11) is False


class TestGoodFft:
    def test_bumps_a_disallowed_dimension_up(self):
        # 7 has a factor of 7; 8 = 2^3 is the next FFT-friendly dimension.
        assert _good_fft(7) == 8

    def test_leaves_an_already_allowed_dimension(self):
        assert _good_fft(24) == 24


class TestAutogenerateNrb:
    def test_user_supplied_nrb_is_left_untouched(self, ozone_structure, ozone_real_pseudos):
        parameters = {"SYSTEM": {"nr1b": 10, "nr2b": 10, "nr3b": 10}}
        _autogenerate_nrb(ozone_structure, ozone_real_pseudos, parameters)
        assert parameters["SYSTEM"] == {"nr1b": 10, "nr2b": 10, "nr3b": 10}

    def test_no_core_corrected_pseudo_leaves_nrb_unset(
        self, ozone_structure, generate_full_upf_data
    ):
        pseudos = {"O": generate_full_upf_data("O", core_correction=False)}
        parameters = {"SYSTEM": {"ecutwfc": 30.0}}
        _autogenerate_nrb(ozone_structure, pseudos, parameters)
        assert "nr1b" not in parameters["SYSTEM"]

    def test_core_corrected_pseudo_fills_in_the_box_grid(self, generate_full_upf_data):
        from aiida.orm import StructureData

        # A 10 Angstrom cubic cell with ecutwfc=30 gives nr1b=nr2b=nr3b=24
        # by the formula in the docstring (rc_safe=3 Bohr): computed
        # independently and pinned here as a golden value.
        structure = StructureData(cell=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], pbc=True)
        structure.append_atom(position=[0.0, 0.0, 0.0], symbols="O", name="O")
        pseudos = {"O": generate_full_upf_data("O", core_correction=True)}
        parameters = {"SYSTEM": {"ecutwfc": 30.0}}
        _autogenerate_nrb(structure, pseudos, parameters)
        assert parameters["SYSTEM"]["nr1b"] == 24
        assert parameters["SYSTEM"]["nr2b"] == 24
        assert parameters["SYSTEM"]["nr3b"] == 24


class TestBuildKcpInputs:
    def test_wires_the_evcfixed_parent_folder(self, ozone_structure, ozone_real_pseudos, kcp_code):
        from aiida import orm

        dummy_remote = orm.RemoteData(remote_path="/nonexistent/fake")
        inputs = build_kcp_inputs(
            code=kcp_code,
            structure=ozone_structure,
            parameters={"SYSTEM": {"ecutwfc": 30.0}},
            pseudos=ozone_real_pseudos,
            parent_folder_evcfixed=dummy_remote,
        )
        assert inputs["parent_folder_evcfixed"] is dummy_remote
