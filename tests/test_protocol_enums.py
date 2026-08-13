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
from aiida_koopmans.workgraphs.wannier90 import OptimizeWannierization, Wannierize
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


class TestOptimizeWannierizationOccupations:
    """The optimizing Wannierization honours ``electronic_type`` the same way."""

    @staticmethod
    def _build(fake_cutoffs_family, silicon_structure, wannier_codes, electronic_type):
        wg = OptimizeWannierization.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            electronic_type=electronic_type,
        )
        return wg.tasks["Wannier90OptimizeWorkChain"]

    def test_proxied_insulator_fixes_both_steps(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        task = self._build(
            fake_cutoffs_family,
            silicon_structure,
            wannier_codes,
            TaggedValue(ElectronicType.INSULATOR),
        )
        _assert_fixed(_system(task, "scf", "pw"))
        _assert_fixed(_system(task, "nscf", "pw"))

    def test_proxied_metal_still_smears(
        self, fake_cutoffs_family, silicon_structure, wannier_codes
    ):
        task = self._build(
            fake_cutoffs_family,
            silicon_structure,
            wannier_codes,
            TaggedValue(ElectronicType.METAL),
        )
        _assert_smeared(_system(task, "scf", "pw"))
        _assert_smeared(_system(task, "nscf", "pw"))


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
