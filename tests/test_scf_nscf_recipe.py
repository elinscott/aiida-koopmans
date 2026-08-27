"""Tests for the shared scf + nscf recipe behind ``RunScfNscf``.

The pair is seeded from
``Wannier90WorkChain.get_scf_nscf_builders_from_protocol``, so the nscf
inherits the invariants a Wannierization needs instead of restating them.
These build the graphs (no daemon, no code execution) and read the
materialized ``PwBaseWorkChain`` inputs.
"""

from __future__ import annotations

import pytest
from aiida_quantumespresso.common.types import ElectronicType
from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

from aiida_koopmans.projections import ExplicitProjectionBlock, nbnd_covering_blocks
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks
from aiida_koopmans.workgraphs.pw import RunScfNscf
from tests.fixtures import expand_subgraph, explicit_block


def _silicon_blocks() -> list[ExplicitProjectionBlock]:
    """Si tutorial shape: one 4-band occupied block, one 4-band empty block."""
    return [
        explicit_block("block_1", range(1, 5), filled=True),
        explicit_block("block_2", range(5, 9), filled=False),
    ]


def _parameters(task):
    """Return a built pw step's parameters as a plain dict."""
    return task.inputs["pw"]["parameters"].value.get_dict()


def _pw_step_parameters(wg):
    """Yield the parameters of every pw step a built graph carries."""
    for task in wg.tasks:
        try:
            value = task.inputs["pw"]["parameters"].value
        except (AttributeError, KeyError, TypeError):
            continue
        if value is not None:
            yield value.get_dict()


class TestRecipeIsShared:
    def test_nscf_parameters_are_the_recipe_builder_s(
        self, fake_cutoffs_family, silicon_structure, kmesh, kpath, pw_code
    ):
        """Every keyword the recipe's nscf builder sets survives to the step.

        The pin that makes the recipe *shared*: change it upstream and this
        comparison carries the change through, because the expected values
        are read off the recipe rather than written out here.
        """
        overrides = {"nscf": {"pw": {"parameters": {"SYSTEM": {"ecutwfc": 60.0}}}}}
        _, expected_builder = Wannier90WorkChain.get_scf_nscf_builders_from_protocol(
            pw_code,
            structure=silicon_structure,
            nbnd=8,
            kpoints=kpath,
            overrides=overrides,
            pseudo_family=fake_cutoffs_family.label,
            electronic_type=ElectronicType.INSULATOR,
        )
        expected = expected_builder["pw"]["parameters"].get_dict()

        wg = RunScfNscf.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            scf_kpoints=kmesh,
            nscf_kpoints=kpath,
            nbnd=8,
            overrides=overrides,
        )
        built = _parameters(wg.tasks["nscf"])

        for namelist, keywords in expected.items():
            for keyword, value in keywords.items():
                assert built[namelist][keyword] == value, (namelist, keyword)

    def test_nscf_carries_the_wannier_invariants(
        self, fake_cutoffs_family, silicon_structure, kmesh, kpath, pw_code
    ):
        """The keywords a Wannierization nscf cannot run without.

        ``nosym`` / ``noinv`` keep the full k-grid, ``diago_full_acc``
        converges the empty states wannier90 disentangles over, and
        ``startingpot = 'file'`` reads the scf density rather than an atomic
        guess.
        """
        wg = RunScfNscf.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            scf_kpoints=kmesh,
            nscf_kpoints=kpath,
        )
        parameters = _parameters(wg.tasks["nscf"])
        assert parameters["SYSTEM"]["nosym"] is True
        assert parameters["SYSTEM"]["noinv"] is True
        assert parameters["ELECTRONS"]["diago_full_acc"] is True
        assert parameters["ELECTRONS"]["startingpot"] == "file"
        assert parameters["CONTROL"]["calculation"] == "nscf"

        # The scf is a plain ground state: none of the above belongs to it.
        scf_parameters = _parameters(wg.tasks["scf"])
        assert scf_parameters["CONTROL"]["calculation"] == "scf"
        assert "diago_full_acc" not in scf_parameters["ELECTRONS"]
        assert "startingpot" not in scf_parameters["ELECTRONS"]

    def test_a_mesh_reaches_the_nscf_as_the_wannier90_ordered_list(
        self, fake_cutoffs_family, silicon_structure, kmesh, pw_code
    ):
        """A mesh handed to the nscf is expanded, not passed through.

        wannier90 reads the eigenstates in its own ``kmesh.pl`` order, and
        pw.x may reduce a mesh by symmetry; the recipe expands the mesh
        once so the two cannot disagree.
        """
        wg = RunScfNscf.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            nscf_kpoints=kmesh,
        )
        nscf_kpoints = wg.tasks["nscf"].inputs["kpoints"].value
        mesh = kmesh.get_kpoints_mesh()[0]
        assert len(nscf_kpoints.get_kpoints()) == mesh[0] * mesh[1] * mesh[2]
        # Exactly one of ``kpoints`` and ``kpoints_distance`` may be given,
        # and the parity flag only qualifies the distance.
        assert wg.tasks["nscf"].inputs["kpoints_distance"].value is None
        assert wg.tasks["nscf"].inputs["kpoints_force_parity"].value is None


class TestNscfBandCount:
    def test_nbnd_covers_the_highest_band_any_block_reads(
        self, wannier_codes, silicon_structure, kmesh, fake_cutoffs_family, pw_code
    ):
        """Without a stated ``nbnd`` the blocks fix the nscf band count."""
        blocks = _silicon_blocks()
        assert nbnd_covering_blocks(blocks) == 8

        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=blocks,
            kpoints=kmesh,
            pseudo_family=fake_cutoffs_family.label,
        )
        assert wg.tasks["scf_nscf"].inputs["nbnd"].value == 8

        inner = expand_subgraph(wg.tasks["scf_nscf"], RunScfNscf)
        assert _parameters(inner.tasks["nscf"])["SYSTEM"]["nbnd"] == 8

    def test_a_larger_stated_nbnd_wins(
        self, wannier_codes, silicon_structure, kmesh, fake_cutoffs_family
    ):
        """Asking for more bands than the blocks read keeps them."""
        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            pseudo_family=fake_cutoffs_family.label,
            overrides={"nscf": {"pw": {"parameters": {"SYSTEM": {"nbnd": 12}}}}},
        )
        assert wg.tasks["scf_nscf"].inputs["nbnd"].value == 12

    def test_an_nbnd_below_the_blocks_raises(
        self, wannier_codes, silicon_structure, kmesh, fake_cutoffs_family
    ):
        """An nscf too short for the blocks is refused, not silently run."""
        with pytest.raises(ValueError, match="read up to band 8"):
            WannierizeBlocks.build(
                codes=wannier_codes,
                structure=silicon_structure,
                blocks=_silicon_blocks(),
                kpoints=kmesh,
                pseudo_family=fake_cutoffs_family.label,
                overrides={"nscf": {"pw": {"parameters": {"SYSTEM": {"nbnd": 6}}}}},
            )

    def test_the_quality_check_bands_run_computes_the_same_bands(
        self, wannier_codes, silicon_structure, kmesh, kpath, fake_cutoffs_family
    ):
        """The reference curve must reach every band the interpolation covers.

        pw.x defaults to roughly the occupied bands, so a bands run seeded
        without ``nbnd`` would stop at the valence top and the empty-state
        comparison would have nothing to compare against.
        """
        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=_silicon_blocks(),
            kpoints=kmesh,
            bands_kpoints=kpath,
            pseudo_family=fake_cutoffs_family.label,
        )
        bands = [
            parameters
            for parameters in _pw_step_parameters(wg)
            if parameters["CONTROL"].get("calculation") == "bands"
        ]
        assert len(bands) == 1
        assert bands[0]["SYSTEM"]["nbnd"] == 8


class TestOtherRoutesKeepTheirForcing:
    def test_dfpt_still_forces_symmetry_off(
        self, dfpt_codes, silicon_structure, kmesh, fake_cutoffs_family
    ):
        """A user ``nosym = False`` cannot reach the DFPT nscf.

        The recipe supplies ``nosym`` / ``noinv`` as protocol *defaults*,
        which an override would win over; the DFPT route forces them on top
        of the overrides, so its wannierization keeps the full grid.
        """
        from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow

        wg = SinglepointDFPTWorkflow.build(
            codes=dfpt_codes,
            structure=silicon_structure,
            manifolds={
                "none": {
                    "occ": [explicit_block("occ", range(1, 5), filled=True)],
                    "emp": [explicit_block("emp", range(5, 9), filled=False)],
                }
            },
            kpoints=kmesh,
            pseudo_family=fake_cutoffs_family.label,
            eps_inf=11.7,
            overrides={"nscf": {"pw": {"parameters": {"SYSTEM": {"nosym": False}}}}},
        )
        nscf_system = (
            wg.tasks["scf_nscf"].inputs["overrides"].value["nscf"]["pw"]["parameters"]["SYSTEM"]
        )
        assert nscf_system["nosym"] is True
        assert nscf_system["noinv"] is True
