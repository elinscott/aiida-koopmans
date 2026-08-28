"""Unit tests for the kcp.x input assembler (``calculations/kcp_inputs.py``).

Covers the FFT-dimension arithmetic and the ``SYSTEM.nr{1,2,3}b`` box-grid
derivation in isolation from any CalcJob; ``build_kcp_inputs`` itself is
exercised by the workgraph tests that build real ``KcpStep`` inputs.
"""

from __future__ import annotations

from aiida_koopmans.calculations.kcp_inputs import (
    _fft_dimension_allowed,
    _good_fft,
    autogenerate_nrb,
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
    def test_no_core_corrected_pseudo_returns_none(self, ozone_structure, generate_full_upf_data):
        pseudos = {"O": generate_full_upf_data("O", core_correction=False)}
        assert autogenerate_nrb(ozone_structure, pseudos, ecutwfc=30.0, ecutrho=120.0) is None

    def test_core_corrected_pseudo_gives_the_box_grid(self, generate_full_upf_data):
        from aiida.orm import StructureData

        # A 10 Angstrom cubic cell with ecutrho=120 gives nr1b=nr2b=nr3b=24
        # by the formula in the docstring (rc_safe=3 Bohr): computed
        # independently and pinned here as a golden value.
        structure = StructureData(cell=[[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]], pbc=True)
        structure.append_atom(position=[0.0, 0.0, 0.0], symbols="O", name="O")
        pseudos = {"O": generate_full_upf_data("O", core_correction=True)}
        assert autogenerate_nrb(structure, pseudos, ecutwfc=30.0, ecutrho=120.0) == (24, 24, 24)

    def test_an_unreadable_pseudo_is_not_silently_no_nlcc(self, ozone_structure):
        """Anything but an incomplete UPF header propagates.

        A pseudo we cannot inspect must not pass as "no core correction":
        that is how a run reaches kcp.x with the box grid unset.
        """
        import pytest

        class Unreadable:
            filename = "O.upf"

            def get_content(self):
                raise OSError("no repository file")

        with pytest.raises(OSError):
            autogenerate_nrb(ozone_structure, {"O": Unreadable()}, ecutwfc=30.0, ecutrho=120.0)


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

    def test_leaves_the_callers_parameters_dict_alone(
        self, ozone_structure, ozone_real_pseudos, kcp_code
    ):
        """``build_kcp_inputs`` must not mutate the dict it is handed.

        Inside a ``@task.graph`` body a socket-fed dict is passed on by
        reference to its upstream node, so an in-place edit never reaches
        the CalcJob's stored inputs. Nothing here may rely on one.
        """
        parameters = {"SYSTEM": {"ecutwfc": 30.0}}
        build_kcp_inputs(
            code=kcp_code,
            structure=ozone_structure,
            parameters=parameters,
            pseudos=ozone_real_pseudos,
        )
        assert parameters == {"SYSTEM": {"ecutwfc": 30.0}}
