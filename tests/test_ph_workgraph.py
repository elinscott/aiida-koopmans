"""Construction-level tests for the dielectric-constant (ph.x) workgraph.

Build the ``DielectricTask`` graph (no daemon, no real code execution) and
introspect its task list / wiring. Also unit-tests the
``extract_dielectric_constant`` task via its raw ``._callable`` and the
``eps_inf='auto'`` hook of ``SinglepointDFPTWorkflow``.
"""

from __future__ import annotations

import pytest

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


def _si_manifolds():
    """Return the single-occupied-manifold shape the auto-eps builds run on."""
    return {
        "none": {"occ": [explicit_block("occ", range(1, 5), projections=["Si:sp3"], filled=True)]}
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
        """eps_inf='auto' without codes['ph'] fails at build with the structured report."""
        from aiida_workgraph.errors import MissingRequiredInputsError

        codes = {key: value for key, value in ph_codes.items() if key != "ph"}
        with pytest.raises(MissingRequiredInputsError) as excinfo:
            SinglepointDFPTWorkflow.build(
                codes=codes,
                structure=silicon_structure,
                manifolds=_si_manifolds(),
                kpoints=kmesh,
                pseudo_family="SSSP/1.3/PBE/efficiency",
                eps_inf="auto",
            )
        assert [entry.socket_path for entry in excinfo.value.missing] == ["graph_inputs.codes.ph"]
