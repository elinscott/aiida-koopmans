"""Tests for the shared scf + nscf recipe behind ``RunWannierGroundState``.

The pair is seeded from
``Wannier90WorkChain.get_scf_nscf_builders_from_protocol``, so the nscf
inherits the invariants a Wannierization needs instead of restating them.
These build the graphs (no daemon, no code execution) and read the
materialized ``PwBaseWorkChain`` inputs.
"""

from __future__ import annotations

from aiida import orm
from aiida_quantumespresso.common.types import ElectronicType
from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

from aiida_koopmans.workgraphs.wannier_ground_state import RunWannierGroundState
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

        wg = RunWannierGroundState.build(
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
        wg = RunWannierGroundState.build(
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
        wg = RunWannierGroundState.build(
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

    def test_a_user_nosym_override_cannot_reach_the_nscf(
        self, fake_cutoffs_family, silicon_structure, kmesh, kpath, pw_code
    ):
        """A user ``nosym = False`` / ``noinv = False`` is overwritten.

        wannier90 needs the full, symmetry-unreduced grid it orders itself;
        the recipe forces ``nosym`` / ``noinv`` on top of whatever a caller
        supplies, so an override cannot switch the nscf's protocol default
        off. This is the recipe's own invariant -- every caller shares it.
        """
        wg = RunWannierGroundState.build(
            pw_code=pw_code,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            scf_kpoints=kmesh,
            nscf_kpoints=kpath,
            overrides={
                "nscf": {"pw": {"parameters": {"SYSTEM": {"nosym": False, "noinv": False}}}}
            },
        )
        parameters = _parameters(wg.tasks["nscf"])
        assert parameters["SYSTEM"]["nosym"] is True
        assert parameters["SYSTEM"]["noinv"] is True


class TestCallersDoNotForceSymmetryThemselves:
    """Neither Wannierizing route forces ``nosym`` / ``noinv`` on its own.

    The enforcement is :func:`RunWannierGroundState`'s own invariant (see
    ``TestRecipeIsShared.test_a_user_nosym_override_cannot_reach_the_nscf``);
    a route that forced it again on top would just duplicate that logic, so
    a caller's raw override reaches the ``scf_nscf`` task's ``overrides``
    input unmodified.
    """

    def test_dfpt_passes_the_override_through_unmodified(
        self, dfpt_codes, silicon_structure, kmesh, fake_cutoffs_family
    ):
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
        assert nscf_system["nosym"] is False

    def test_block_route_passes_the_override_through_unmodified(
        self, wannier_codes, silicon_structure, kmesh
    ):
        from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlocks

        wg = WannierizeBlocks.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=[
                explicit_block("block_1", range(1, 5), filled=True),
                explicit_block("block_2", range(5, 9), filled=False),
            ],
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            overrides={"nscf": {"pw": {"parameters": {"SYSTEM": {"nosym": False}}}}},
        )
        nscf_system = (
            wg.tasks["scf_nscf"].inputs["overrides"].value["nscf"]["pw"]["parameters"]["SYSTEM"]
        )
        assert nscf_system["nosym"] is False


def _plain(value):
    """Reduce a socket value to something comparable across two builds.

    Nodes the builders create fresh each build (``parameters``,
    ``kpoints``, ``max_iterations``) compare by content; nodes loaded from
    the profile (the structure, the code, the pseudos) compare by uuid.
    """
    if isinstance(value, orm.Dict):
        return value.get_dict()
    if isinstance(value, orm.BaseType):
        return value.value
    if isinstance(value, orm.KpointsData):
        try:
            return ("list", value.get_kpoints().tolist())
        except AttributeError:
            return ("mesh", value.get_kpoints_mesh())
    if isinstance(value, orm.Node):
        return value.uuid
    return value


def _inputs(socket):
    """Reduce a task's input namespace to a nested dict of plain values."""
    if hasattr(socket, "_sockets"):
        return {name: _inputs(child) for name, child in socket._sockets.items()}
    return _plain(socket.value)


class TestAnExternalDensitySkipsTheScf:
    """``scf_remote_folder`` runs the recipe's nscf alone off a given density.

    The dense-mesh re-Wannierization the smooth interpolation needs: the
    coarse run's density is already converged, so only the nscf is left to
    run -- and it must be the recipe's nscf, not a second hand-built one.
    """

    @staticmethod
    def _build(pw_code, structure, family, kpath, scf_remote_folder=None):
        return RunWannierGroundState.build(
            pw_code=pw_code,
            structure=structure,
            pseudo_family=family.label,
            nscf_kpoints=kpath,
            scf_remote_folder=scf_remote_folder,
        )

    def test_no_scf_step_is_built(
        self, fake_cutoffs_family, silicon_structure, kpath, pw_code, scf_remote
    ):
        """The given density replaces the scf run, rather than seeding one."""
        wg = self._build(pw_code, silicon_structure, fake_cutoffs_family, kpath, scf_remote)
        names = [t.name for t in wg.tasks]
        assert "scf" not in names, names
        assert "nscf" in names, names

    def test_the_nscf_restarts_from_the_given_density(
        self, fake_cutoffs_family, silicon_structure, kpath, pw_code, scf_remote
    ):
        wg = self._build(pw_code, silicon_structure, fake_cutoffs_family, kpath, scf_remote)
        nscf = wg.tasks["nscf"]
        assert nscf.inputs["pw"]["parent_folder"].value.uuid == scf_remote.uuid
        assert _parameters(nscf)["CONTROL"]["calculation"] == "nscf"

    def test_the_given_density_comes_back_out(
        self, fake_cutoffs_family, silicon_structure, kpath, pw_code, scf_remote
    ):
        """``scf_remote_folder`` is an output in both modes, so consumers read one socket."""
        wg = self._build(pw_code, silicon_structure, fake_cutoffs_family, kpath, scf_remote)
        links = wg.outputs["scf_remote_folder"]._links
        assert [link.from_socket._name for link in links] == ["scf_remote_folder"]
        assert [link.from_task.name for link in links] == ["graph_inputs"]

    def test_the_nscf_is_otherwise_the_internal_route_s_nscf(
        self, fake_cutoffs_family, silicon_structure, kpath, pw_code, scf_remote
    ):
        """The discriminating check: only the parent folder differs.

        A hand-built nscf would drift from the recipe's silently -- a
        missing ``diago_full_acc``, a different ``nbnd``, its own
        ``startingpot``. Comparing every input against the internal route's
        own nscf catches any such divergence without listing the keywords
        here.
        """
        external = self._build(pw_code, silicon_structure, fake_cutoffs_family, kpath, scf_remote)
        internal = self._build(pw_code, silicon_structure, fake_cutoffs_family, kpath)

        external_inputs = _inputs(external.tasks["nscf"].inputs)
        internal_inputs = _inputs(internal.tasks["nscf"].inputs)

        # The internal route's parent folder is a socket on its own scf
        # step, so it has no value to compare; that difference is the point.
        assert external_inputs["pw"].pop("parent_folder") == scf_remote.uuid
        assert internal_inputs["pw"].pop("parent_folder") is None
        assert internal.tasks["nscf"].inputs["pw"]["parent_folder"]._links

        assert external_inputs == internal_inputs
