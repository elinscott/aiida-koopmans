"""Tests for the shared scf + nscf recipe behind ``RunScfNscf``.

The pair is seeded from
``Wannier90WorkChain.get_scf_nscf_builders_from_protocol``, so the nscf
inherits the invariants a Wannierization needs instead of restating them.
These build the graphs (no daemon, no code execution) and read the
materialized ``PwBaseWorkChain`` inputs.
"""

from __future__ import annotations

from aiida_quantumespresso.common.types import ElectronicType
from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

from aiida_koopmans.workgraphs.pw import RunScfNscf
from tests.fixtures import explicit_block


def _parameters(task):
    """Return a built pw step's parameters as a plain dict."""
    return task.inputs["pw"]["parameters"].value.get_dict()


class TestRecipeIsShared:
    def test_nscf_parameters_are_the_recipe_builder_s(
        self, fake_cutoffs_family, silicon_structure, kmesh, kpath, pw_code
    ):
        """Every keyword the recipe's nscf builder sets survives to the step.

        The pin that makes the recipe *shared*: change it upstream and this
        comparison carries the change through, because the expected values
        are read off the recipe rather than written out here.
        """
        overrides = {"nscf": {"pw": {"parameters": {"SYSTEM": {"ecutwfc": 60.0, "nbnd": 8}}}}}
        _, expected_builder = Wannier90WorkChain.get_scf_nscf_builders_from_protocol(
            pw_code,
            structure=silicon_structure,
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


class TestOtherRoutesKeepTheirForcing:
    def test_dfpt_still_forces_symmetry_off(
        self, dfpt_codes, silicon_structure, kmesh, fake_cutoffs_family
    ):
        """A user ``nosym = False`` cannot reach the DFPT nscf.

        The recipe supplies ``nosym`` / ``noinv`` as protocol defaults,
        which a user override replaces; the DFPT route forces them on top
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
