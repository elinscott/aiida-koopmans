"""Tests that a graph body's enum arguments reach the protocol builders intact.

A graph input is a ``TaggedValue`` proxy, for which ``is`` against an enum
member is false while ``==`` is true. ``aiida-quantumespresso`` branches on
``electronic_type is ElectronicType.INSULATOR`` and
``spin_type is SpinType.COLLINEAR``, so an uncoerced proxy silently takes the
metallic / half-magnetic branch. The graph-build tests therefore read the
*materialized* pw parameters, and each pairs the insulating assertion with a
metallic one so a blanket override could not pass them both.
"""

from __future__ import annotations

from aiida_quantumespresso.common.types import ElectronicType, SpinType
from node_graph.socket import TaggedValue

from aiida_koopmans.workgraphs import unwrap_enum
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks
from aiida_koopmans.workgraphs.pw import RunPwBands, RunScfNscf
from aiida_koopmans.workgraphs.wannier90 import Wannierize
from tests.fixtures import explicit_block


def _system(task, *namespace):
    """Return the SYSTEM namelist of a built task's ``<namespace...>.parameters``."""
    node = task.inputs
    for key in namespace:
        node = node[key]
    return node["parameters"].value.get_dict()["SYSTEM"]


def _assert_fixed(system):
    assert system["occupations"] == "fixed"
    assert "smearing" not in system
    assert "degauss" not in system


def _assert_smeared(system):
    assert system["occupations"] == "smearing"
    assert system["degauss"] > 0


class TestUnwrapEnum:
    def test_member_passes_through_identically(self):
        assert unwrap_enum(ElectronicType.METAL, ElectronicType) is ElectronicType.METAL

    def test_proxy_becomes_the_genuine_member(self):
        proxied = TaggedValue(ElectronicType.INSULATOR)
        assert proxied is not ElectronicType.INSULATOR
        assert unwrap_enum(proxied, ElectronicType) is ElectronicType.INSULATOR

    def test_proxy_wrapping_the_value_string_becomes_the_member(self):
        assert unwrap_enum(TaggedValue("collinear"), SpinType) is SpinType.COLLINEAR

    def test_bare_value_string_becomes_the_member(self):
        assert unwrap_enum("insulator", ElectronicType) is ElectronicType.INSULATOR

    def test_none_stays_none(self):
        assert unwrap_enum(None, ElectronicType) is None


class TestWannierizeOccupations:
    """The scf and nscf feeding a Wannierization honour ``electronic_type``."""

    @staticmethod
    def _build(fake_cutoffs_family, silicon_structure, wannier_codes, **kwargs):
        return Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            **kwargs,
        )

    def test_default_insulator_fixes_both_steps(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        """The declared ``INSULATOR`` default reaches both pw steps."""
        wg = self._build(fake_cutoffs_family, silicon_structure, wannier_codes)
        task = wg.tasks["Wannier90WorkChain"]
        _assert_fixed(_system(task, "scf", "pw"))
        _assert_fixed(_system(task, "nscf", "pw"))

    def test_metal_still_smears(self, fake_cutoffs_family, silicon_structure, wannier_codes):
        """A metallic run keeps the protocol's smearing — the fix is not a blanket override."""
        wg = self._build(
            fake_cutoffs_family,
            silicon_structure,
            wannier_codes,
            electronic_type=ElectronicType.METAL,
        )
        task = wg.tasks["Wannier90WorkChain"]
        _assert_smeared(_system(task, "scf", "pw"))
        _assert_smeared(_system(task, "nscf", "pw"))

    def test_proxied_insulator_fixes_both_steps(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        """A proxy — the form a graph input actually takes — fixes occupations too."""
        wg = self._build(
            fake_cutoffs_family,
            silicon_structure,
            wannier_codes,
            electronic_type=TaggedValue(ElectronicType.INSULATOR),
        )
        task = wg.tasks["Wannier90WorkChain"]
        _assert_fixed(_system(task, "scf", "pw"))
        _assert_fixed(_system(task, "nscf", "pw"))

    def test_proxied_collinear_sets_nspin_with_the_magnetization(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        """A collinear run gets ``nspin=2``, not ``starting_magnetization`` alone."""
        wg = self._build(
            fake_cutoffs_family,
            silicon_structure,
            wannier_codes,
            spin_type=TaggedValue(SpinType.COLLINEAR),
        )
        system = _system(wg.tasks["Wannier90WorkChain"], "scf", "pw")
        assert system["starting_magnetization"]
        assert system["nspin"] == 2


class TestSplitRouteBandsStepOccupations:
    """The split route's bands step honours ``electronic_type`` as well.

    Its scf and nscf sit in a nested graph, whose body has not run at build
    time, but the bands step is assembled by a plain helper called from
    ``WannierizeBlocks``' own body — so in production its ``electronic_type``
    is a proxy, exactly as for a top-level graph.
    """

    @staticmethod
    def _bands_step(auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family, **kwargs):
        wg = WannierizeBlocks.build(
            codes=auto_codes,
            structure=silicon_structure,
            blocks=[
                explicit_block("block_1", range(1, 5), ["Si: sp3"]),
                explicit_block("block_2", range(5, 9), ["Si: sp3"]),
            ],
            kpoints=kmesh,
            mp_grid=[2, 2, 2],
            bands_kpoints=kpath,
            num_occ_bands=4,
            split_threshold=1.5,
            pseudo_family=fake_cutoffs_family.label,
            **kwargs,
        )
        return wg.tasks["bands"]

    def test_default_insulator_fixes_the_bands_step(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        task = self._bands_step(auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family)
        _assert_fixed(_system(task, "pw"))

    def test_metal_still_smears_the_bands_step(
        self, auto_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        task = self._bands_step(
            auto_codes,
            silicon_structure,
            kmesh,
            kpath,
            fake_cutoffs_family,
            electronic_type=ElectronicType.METAL,
        )
        _assert_smeared(_system(task, "pw"))


class TestRunScfNscfOccupations:
    """``RunScfNscf`` honours ``electronic_type`` on both of its steps."""

    @staticmethod
    def _build(fake_cutoffs_family, silicon_structure, kmesh, pw_code, **kwargs):
        return RunScfNscf.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kmesh,
            **kwargs,
        )

    def test_proxied_insulator_fixes_both_steps(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        wg = self._build(
            fake_cutoffs_family,
            silicon_structure,
            kmesh,
            pw_code,
            electronic_type=TaggedValue(ElectronicType.INSULATOR),
        )
        _assert_fixed(_system(wg.tasks["scf"], "pw"))
        _assert_fixed(_system(wg.tasks["nscf"], "pw"))

    def test_proxied_metal_still_smears(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        wg = self._build(
            fake_cutoffs_family,
            silicon_structure,
            kmesh,
            pw_code,
            electronic_type=TaggedValue(ElectronicType.METAL),
        )
        _assert_smeared(_system(wg.tasks["scf"], "pw"))
        _assert_smeared(_system(wg.tasks["nscf"], "pw"))


class TestRunScfNscfSpin:
    """``RunScfNscf`` honours ``spin_type`` on both of its steps.

    This is the ground state every Wannierization is built on, so a regime
    that stops at the graph boundary leaves the blocks Wannierized off an
    unpolarized density whatever the run asked for. Every regime is
    proxied, the form a graph input takes in production.
    """

    @staticmethod
    def _system_pair(fake_cutoffs_family, silicon_structure, kmesh, pw_code, spin_type):
        wg = RunScfNscf.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kmesh,
            spin_type=TaggedValue(spin_type),
            # Fixed occupations under nspin = 2 need a magnetization.
            overrides={
                step: {"pw": {"parameters": {"SYSTEM": {"tot_magnetization": 2}}}}
                for step in ("scf", "nscf")
            },
        )
        return _system(wg.tasks["scf"], "pw"), _system(wg.tasks["nscf"], "pw")

    def test_default_none_leaves_the_namelist_unpolarized(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        """Nothing given: no spin keyword appears — the negative control."""
        wg = RunScfNscf.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kmesh,
        )
        for system in (_system(wg.tasks["scf"], "pw"), _system(wg.tasks["nscf"], "pw")):
            assert "nspin" not in system
            assert "noncolin" not in system
            assert "starting_magnetization" not in system

    def test_collinear_sets_nspin_beside_the_magnetization(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        """Both keywords together: pw.x refuses a moment with no ``nspin``."""
        for system in self._system_pair(
            fake_cutoffs_family, silicon_structure, kmesh, pw_code, SpinType.COLLINEAR
        ):
            assert system["nspin"] == 2
            assert system["tot_magnetization"] == 2
            assert "noncolin" not in system

    def test_non_collinear_sets_noncolin_without_spinorb(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        for system in self._system_pair(
            fake_cutoffs_family, silicon_structure, kmesh, pw_code, SpinType.NON_COLLINEAR
        ):
            assert system["noncolin"] is True
            assert system["nspin"] == 4
            assert "lspinorb" not in system

    def test_spin_orbit_adds_lspinorb(self, fake_cutoffs_family, silicon_structure, kmesh, pw_code):
        for system in self._system_pair(
            fake_cutoffs_family, silicon_structure, kmesh, pw_code, SpinType.SPIN_ORBIT
        ):
            assert system["noncolin"] is True
            assert system["lspinorb"] is True


class TestWannierizeBlocksSpinReachesTheGroundState:
    """``WannierizeBlocks`` hands its ``spin_type`` to every pw.x step it runs.

    The shared scf + nscf sits in a nested graph whose body has not run at
    build time, so it is checked on the forwarded socket; the quality-check
    bands step is assembled by a plain helper in this graph's own body and
    is checked on its materialized namelist. The bands step reads the scf
    density back channel by channel, so a regime that reached the scf but
    not the bands run would abort mid-run.
    """

    @staticmethod
    def _build(pdos_codes, silicon_structure, kmesh, labelled_kpath, fake_cutoffs_family, **kwargs):
        return WannierizeBlocks.build(
            codes=pdos_codes,
            structure=silicon_structure,
            blocks=[
                explicit_block("block_1", range(1, 5), ["Si: sp3"]),
                explicit_block("block_2", range(5, 9), ["Si: sp3"]),
            ],
            kpoints=kmesh,
            pseudo_family=fake_cutoffs_family.label,
            interpolation_kpoints=labelled_kpath,
            **kwargs,
        )

    def test_default_none_leaves_both_unpolarized(
        self, pdos_codes, silicon_structure, kmesh, labelled_kpath, fake_cutoffs_family
    ):
        """The negative control for the collinear case below."""
        wg = self._build(pdos_codes, silicon_structure, kmesh, labelled_kpath, fake_cutoffs_family)
        assert wg.tasks["scf_nscf"].inputs["spin_type"].value == SpinType.NONE
        assert "nspin" not in _system(wg.tasks["bands"], "pw")

    def test_collinear_reaches_the_shared_pair_and_the_bands_step(
        self, pdos_codes, silicon_structure, kmesh, labelled_kpath, fake_cutoffs_family
    ):
        """One regime, stated once, reaches every ground-state step."""
        magnetized = {"pw": {"parameters": {"SYSTEM": {"tot_magnetization": 2}}}}
        wg = self._build(
            pdos_codes,
            silicon_structure,
            kmesh,
            labelled_kpath,
            fake_cutoffs_family,
            spin_type=TaggedValue(SpinType.COLLINEAR),
            overrides={"scf": magnetized, "nscf": magnetized},
        )
        assert wg.tasks["scf_nscf"].inputs["spin_type"].value == SpinType.COLLINEAR
        system = _system(wg.tasks["bands"], "pw")
        assert system["nspin"] == 2
        assert system["tot_magnetization"] == 2


class TestRunPwBandsOccupations:
    """``RunPwBands`` honours ``electronic_type`` on both of its steps.

    ``PwBandsWorkChain.get_builder_from_protocol`` only forwards
    ``electronic_type`` to its scf/bands sub-builders via ``**kwargs``; the
    default here must reach both without ``RunPwBands`` naming the keyword
    explicitly at the call site.
    """

    @staticmethod
    def _build(fake_cutoffs_family, silicon_structure, kmesh, pw_code, **kwargs):
        return RunPwBands.build(
            codes={"pw": pw_code},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kmesh,
            **kwargs,
        )

    def test_default_insulator_fixes_both_steps(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        """No ``electronic_type`` given: the declared ``INSULATOR`` default fires."""
        wg = self._build(fake_cutoffs_family, silicon_structure, kmesh, pw_code)
        task = wg.tasks["PwBandsWorkChain"]
        _assert_fixed(_system(task, "scf", "pw"))
        _assert_fixed(_system(task, "bands", "pw"))

    def test_proxied_insulator_fixes_both_steps(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        wg = self._build(
            fake_cutoffs_family,
            silicon_structure,
            kmesh,
            pw_code,
            electronic_type=TaggedValue(ElectronicType.INSULATOR),
        )
        task = wg.tasks["PwBandsWorkChain"]
        _assert_fixed(_system(task, "scf", "pw"))
        _assert_fixed(_system(task, "bands", "pw"))

    def test_proxied_metal_still_smears(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        """A metallic run keeps the protocol's smearing — the fix is not a blanket override."""
        wg = self._build(
            fake_cutoffs_family,
            silicon_structure,
            kmesh,
            pw_code,
            electronic_type=TaggedValue(ElectronicType.METAL),
        )
        task = wg.tasks["PwBandsWorkChain"]
        _assert_smeared(_system(task, "scf", "pw"))
        _assert_smeared(_system(task, "bands", "pw"))


class TestRunPwBandsSpin:
    """``RunPwBands`` honours ``spin_type`` on both of its steps.

    Every regime is proxied, the form a graph input takes in production.
    ``PwBandsWorkChain.get_builder_from_protocol`` forwards ``spin_type``
    to its scf/bands sub-builders through ``**kwargs`` only, so a keyword
    the caller never names cannot reach either step.
    """

    @staticmethod
    def _system_pair(fake_cutoffs_family, silicon_structure, kmesh, pw_code, spin_type):
        wg = RunPwBands.build(
            codes={"pw": pw_code},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kmesh,
            spin_type=TaggedValue(spin_type),
            # Fixed occupations under nspin = 2 need a magnetization.
            overrides={
                step: {"pw": {"parameters": {"SYSTEM": {"tot_magnetization": 0}}}}
                for step in ("scf", "bands")
            },
        )
        task = wg.tasks["PwBandsWorkChain"]
        return _system(task, "scf", "pw"), _system(task, "bands", "pw")

    def test_default_none_leaves_the_namelist_unpolarized(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        """Nothing given: neither spin keyword appears — the negative control."""
        wg = RunPwBands.build(
            codes={"pw": pw_code},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            bands_kpoints=kmesh,
        )
        for system in (
            _system(wg.tasks["PwBandsWorkChain"], "scf", "pw"),
            _system(wg.tasks["PwBandsWorkChain"], "bands", "pw"),
        ):
            assert "nspin" not in system
            assert "noncolin" not in system
            assert "starting_magnetization" not in system

    def test_collinear_sets_nspin_on_both_steps(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        for system in self._system_pair(
            fake_cutoffs_family, silicon_structure, kmesh, pw_code, SpinType.COLLINEAR
        ):
            assert system["nspin"] == 2
            assert "noncolin" not in system

    def test_non_collinear_sets_noncolin_without_spinorb(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        for system in self._system_pair(
            fake_cutoffs_family, silicon_structure, kmesh, pw_code, SpinType.NON_COLLINEAR
        ):
            assert system["noncolin"] is True
            assert system["nspin"] == 4
            assert "lspinorb" not in system

    def test_spin_orbit_adds_lspinorb(self, fake_cutoffs_family, silicon_structure, kmesh, pw_code):
        for system in self._system_pair(
            fake_cutoffs_family, silicon_structure, kmesh, pw_code, SpinType.SPIN_ORBIT
        ):
            assert system["noncolin"] is True
            assert system["lspinorb"] is True
