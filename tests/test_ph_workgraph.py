"""Construction-level tests for the dielectric-constant (ph.x) workgraph.

Build the ``DielectricTask`` graph (no daemon, no real code execution) and
introspect its task list / wiring. Also unit-tests the
``extract_dielectric_constant`` task via its raw ``._callable`` and the
``eps_inf='auto'`` hook of ``SinglepointDFPTWorkflow``.
"""

from __future__ import annotations

import pytest
from aiida_quantumespresso.common.types import SpinType
from node_graph.socket import TaggedValue

from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow
from aiida_koopmans.workgraphs.ph import DielectricTask, extract_dielectric_constant
from tests.fixtures import explicit_block

# ----------------------------------------------------------------------
# extract_dielectric_constant (raw callable, no engine)
# ----------------------------------------------------------------------


class TestExtractDielectricConstant:
    """Unit tests for the tensor → eps_inf reduction."""

    def test_isotropic_average(self, aiida_profile):
        """eps_inf is the mean of the tensor diagonal (tr/3)."""
        tensor = [[2.0, 0.1, 0.0], [0.1, 3.0, 0.0], [0.0, 0.0, 4.0]]
        outputs = extract_dielectric_constant._callable({"dielectric_constant": tensor})
        assert outputs["eps_inf"] == pytest.approx(3.0)
        assert outputs["dielectric_tensor"] == tensor

    def test_missing_tensor_raises(self, aiida_profile):
        """A ph.x run without epsil produces no tensor: fail loudly."""
        with pytest.raises(ValueError, match="dielectric_constant"):
            extract_dielectric_constant._callable({"number_of_qpoints": 1})


# ----------------------------------------------------------------------
# Graph construction
# ----------------------------------------------------------------------


class TestDielectricTaskBuild:
    """DielectricTask builds the scf → ph → extract chain."""

    def test_chain_and_namelist(self, ph_codes, silicon_structure, fake_cutoffs_family):
        """The chain has three tasks and ph.x runs epsil-only at Gamma."""
        wg = DielectricTask.build(
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
        )
        names = [t.name for t in wg.tasks]
        assert "scf" in names
        assert "ph" in names
        assert "extract_dielectric_constant" in names

        inputph = wg.tasks["ph"].inputs["ph"]["parameters"].value.get_dict()["INPUTPH"]
        assert inputph["epsil"] is True
        assert inputph["trans"] is False
        # The dielectric tensor is a q = 0 response: Gamma-only q mesh.
        assert wg.tasks["ph"].inputs["qpoints"].value.get_kpoints_mesh() == (
            [1, 1, 1],
            [0.0, 0.0, 0.0],
        )

    def test_caller_ph_overrides_survive_forced_keys(
        self, ph_codes, silicon_structure, fake_cutoffs_family
    ):
        """tr2_ph from the caller survives; epsil / trans stay forced."""
        wg = DielectricTask.build(
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            overrides={
                "ph": {"ph": {"parameters": {"INPUTPH": {"tr2_ph": 1.0e-14, "epsil": False}}}}
            },
        )
        inputph = wg.tasks["ph"].inputs["ph"]["parameters"].value.get_dict()["INPUTPH"]
        assert inputph["tr2_ph"] == pytest.approx(1.0e-14)
        assert inputph["epsil"] is True


class TestDielectricTaskSpin:
    """``spin_type`` reaches the ground state, or is refused before it is built."""

    @staticmethod
    def _build(ph_codes, silicon_structure, fake_cutoffs_family, **kwargs):
        return DielectricTask.build(
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            **kwargs,
        )

    def test_default_none_leaves_the_scf_unpolarized(
        self, ph_codes, silicon_structure, fake_cutoffs_family
    ):
        """The negative control for the collinear case below."""
        wg = self._build(ph_codes, silicon_structure, fake_cutoffs_family)
        system = wg.tasks["scf"].inputs["pw"]["parameters"].value.get_dict()["SYSTEM"]
        assert "nspin" not in system

    def test_collinear_sets_nspin_on_the_scf(
        self, ph_codes, silicon_structure, fake_cutoffs_family
    ):
        """A proxied member — the form a graph input takes — still reaches the namelist."""
        wg = self._build(
            ph_codes,
            silicon_structure,
            fake_cutoffs_family,
            spin_type=TaggedValue(SpinType.COLLINEAR),
            overrides={"scf": {"pw": {"parameters": {"SYSTEM": {"tot_magnetization": 0}}}}},
        )
        system = wg.tasks["scf"].inputs["pw"]["parameters"].value.get_dict()["SYSTEM"]
        assert system["nspin"] == 2

    @pytest.mark.parametrize("spin_type", [SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT])
    def test_spinor_regimes_are_refused(
        self, ph_codes, silicon_structure, fake_cutoffs_family, spin_type
    ):
        """ph.x has no electric-field perturbation for noncollinear magnetism."""
        with pytest.raises(NotImplementedError, match="noncollinear magnetism"):
            self._build(
                ph_codes,
                silicon_structure,
                fake_cutoffs_family,
                spin_type=TaggedValue(spin_type),
            )


def _si_manifolds():
    """Return the single-occupied-manifold shape the auto-eps builds run on."""
    return {
        "none": {"occ": [explicit_block("occ", range(1, 5), projections=["Si:sp3"], filled=True)]}
    }


def _si_spin_manifolds():
    """Return the two-channel manifolds a collinear chain runs on."""
    return {
        channel: {
            "occ": [
                explicit_block(
                    f"occ_{channel}", range(1, nocc + 1), projections=["Si:sp3"], filled=True
                )
            ]
        }
        for channel, nocc in (("up", 5), ("down", 3))
    }


class TestSinglepointDFPTAutoEps:
    """eps_inf='auto' prepends the dielectric chain inside SinglepointDFPTWorkflow."""

    def test_auto_adds_dielectric_task(self, ph_codes, silicon_structure, kmesh):
        """A 'dielectric' task appears and the kcw chain is still built."""
        wg = SinglepointDFPTWorkflow.build(
            codes=ph_codes,
            structure=silicon_structure,
            manifolds=_si_manifolds(),
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf="auto",
        )
        names = [t.name for t in wg.tasks]
        assert "dielectric" in names
        assert "dfpt" in names

    def test_the_dielectric_scf_follows_the_chain_scf(
        self, ph_codes, silicon_structure, kmesh, denser_kmesh
    ):
        """One graph must not hold two ground states on two meshes.

        The dielectric chain is independent of the kcw chain but not of the
        input that describes it, and nothing downstream would record the
        disagreement.
        """
        wg = SinglepointDFPTWorkflow.build(
            codes=ph_codes,
            structure=silicon_structure,
            manifolds=_si_manifolds(),
            kpoints=kmesh,
            scf_kpoints=denser_kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf="auto",
        )
        assert wg.tasks["dielectric"].inputs["scf_kpoints"].value.uuid == denser_kmesh.uuid

    def test_the_dielectric_scf_keeps_a_kpoints_distance(self, ph_codes, silicon_structure, kmesh):
        """A spacing must not be displaced by the mesh the dielectric would default to.

        Its overrides already carry the distance; handing it a mesh as well
        leaves the distance inert, so the two ground states drift apart on
        exactly the input that asked them not to.
        """
        wg = SinglepointDFPTWorkflow.build(
            codes=ph_codes,
            structure=silicon_structure,
            manifolds=_si_manifolds(),
            kpoints=kmesh,
            overrides={"scf": {"kpoints_distance": 0.11}},
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf="auto",
        )
        dielectric = wg.tasks["dielectric"]
        assert dielectric.inputs["scf_kpoints"].value is None
        assert dielectric.inputs["overrides"].value["scf"]["kpoints_distance"] == 0.11

    @staticmethod
    def _dielectric_scf_system(wg, ph_codes, structure):
        """Return the SYSTEM namelist pw.x reads for the chain's dielectric scf.

        ``dielectric`` is a nested graph, unexpanded at build: rebuild it from
        the sockets the chain wired so the assertion lands on the namelist
        rather than on an override dict, which is where a missing keyword is
        invisible.
        """
        dielectric = wg.tasks["dielectric"]
        inner = DielectricTask.build(
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
            structure=structure,
            pseudo_family=dielectric.inputs["pseudo_family"].value,
            overrides=dielectric.inputs["overrides"].value,
            spin_type=dielectric.inputs["spin_type"].value,
        )
        return inner.tasks["scf"].inputs["pw"]["parameters"].value.get_dict()["SYSTEM"]

    def test_the_dielectric_scf_declares_the_chains_spin_regime(
        self, ph_codes, silicon_structure, kmesh, fake_cutoffs_family
    ):
        """A collinear chain's moment must reach a ground state that declares nspin=2.

        The dielectric scf inherits the chain's scf overrides, magnetization
        included, but none of kcw.x's own nspin=2 forcing. pw.x refuses that
        pairing outright — ``tot_magnetization requires nspin=2``, for any
        value — so the regime has to travel with the moment.
        """
        magnetization = {"pw": {"parameters": {"SYSTEM": {"tot_magnetization": 2}}}}
        wg = SinglepointDFPTWorkflow.build(
            codes=ph_codes,
            structure=silicon_structure,
            manifolds=_si_spin_manifolds(),
            kpoints=kmesh,
            pseudo_family=fake_cutoffs_family.label,
            spin=SpinType.COLLINEAR,
            eps_inf="auto",
            overrides={"scf": magnetization, "nscf": magnetization},
        )
        system = self._dielectric_scf_system(wg, ph_codes, silicon_structure)
        assert system["tot_magnetization"] == 2
        assert system["nspin"] == 2

    def test_an_unpolarized_chain_leaves_the_dielectric_scf_unpolarized(
        self, ph_codes, silicon_structure, kmesh, fake_cutoffs_family
    ):
        """The negative control: no moment travels, so no nspin does either.

        kcw.x needs a two-channel scratch even closed-shell, but the
        dielectric response is an independent ground state and gains nothing
        from a second channel.
        """
        wg = SinglepointDFPTWorkflow.build(
            codes=ph_codes,
            structure=silicon_structure,
            manifolds=_si_manifolds(),
            kpoints=kmesh,
            pseudo_family=fake_cutoffs_family.label,
            eps_inf="auto",
        )
        system = self._dielectric_scf_system(wg, ph_codes, silicon_structure)
        assert "nspin" not in system
        assert "tot_magnetization" not in system

    @pytest.mark.parametrize("spin", [SpinType.NON_COLLINEAR, SpinType.SPIN_ORBIT])
    def test_auto_is_refused_for_a_spinor_chain(
        self, ph_codes, silicon_structure, kmesh, fake_cutoffs_family, spin
    ):
        """ph.x has no electric-field perturbation for a noncollinear magnet.

        The chain forces ``domag`` under both spinor regimes, so nothing the
        caller states can make the dielectric run: refuse it here, where the
        advice can name ``eps_inf``.
        """
        with pytest.raises(NotImplementedError, match="eps_inf='auto'"):
            SinglepointDFPTWorkflow.build(
                codes=ph_codes,
                structure=silicon_structure,
                manifolds=_si_manifolds(),
                kpoints=kmesh,
                pseudo_family=fake_cutoffs_family.label,
                spin=spin,
                eps_inf="auto",
            )

    def test_numeric_eps_skips_dielectric_task(self, ph_codes, silicon_structure, kmesh):
        """A numeric eps_inf builds no dielectric chain."""
        wg = SinglepointDFPTWorkflow.build(
            codes=ph_codes,
            structure=silicon_structure,
            manifolds=_si_manifolds(),
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf=11.7,
        )
        assert "dielectric" not in [t.name for t in wg.tasks]

    def test_auto_without_ph_code_raises(self, ph_codes, silicon_structure, kmesh):
        """eps_inf='auto' without codes['ph'] builds, but fails the input check.

        ``reference()`` wires ``codes['ph']`` into the nested dielectric chain
        whether or not it was provided, so the missing code is no longer a
        build-time ``ValueError`` — it surfaces as a ``MissingRequiredInputsError``
        naming the nested socket, the same check ``run`` performs first.
        """
        from aiida_workgraph.errors import MissingRequiredInputsError

        codes = {key: value for key, value in ph_codes.items() if key != "ph"}
        wg = SinglepointDFPTWorkflow.build(
            codes=codes,
            structure=silicon_structure,
            manifolds=_si_manifolds(),
            kpoints=kmesh,
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf="auto",
        )
        with pytest.raises(MissingRequiredInputsError) as excinfo:
            wg.check_before_run()
        assert "dielectric.codes.ph" in {entry.socket_path for entry in excinfo.value.missing}
