"""Construction-level tests for the dielectric-constant (ph.x) workgraph.

Build the ``DielectricTask`` graph (no daemon, no real code execution) and
introspect its task list / wiring. Also unit-tests the
``extract_dielectric_constant`` task via its raw ``._callable`` and the
``eps_inf='auto'`` hook of ``SinglepointDFPTWorkflow``.
"""

from __future__ import annotations

import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.types import ExplicitProjectionBlock, SpinChannel
from aiida_koopmans.workgraphs.dfpt import SinglepointDFPTWorkflow
from aiida_koopmans.workgraphs.ph import DielectricTask, extract_dielectric_constant

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def ph_codes(aiida_localhost):
    """Stand-in codes dict for the dielectric chain (construction-only)."""
    from aiida.common.exceptions import NotExistent
    from aiida.orm import InstalledCode

    def _code(label: str, entry_point: str):
        try:
            return InstalledCode.collection.get(label=label)
        except NotExistent:
            return InstalledCode(
                label=label,
                computer=aiida_localhost,
                filepath_executable="/bin/true",
                default_calc_job_plugin=entry_point,
            ).store()

    return {
        "pw": _code("eps-pw", "quantumespresso.pw"),
        "ph": _code("eps-ph", "quantumespresso.ph"),
        "wannier90": _code("eps-w90", "wannier90.wannier90"),
        "pw2wannier90": _code("eps-p2w", "quantumespresso.pw2wannier90"),
        "kcw": _code("eps-kcw", "koopmans.kcw_wann2kc"),
    }


@pytest.fixture
def silicon_structure(aiida_profile):
    """Return a two-atom silicon ``StructureData``."""
    from aiida.orm import StructureData

    cell = [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]]
    struct = StructureData(cell=cell, pbc=True)
    struct.append_atom(position=(0.0, 0.0, 0.0), symbols="Si", name="Si")
    struct.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si", name="Si")
    return struct


@pytest.fixture
def kmesh(aiida_profile):
    """Return a 2x2x2 ``KpointsData`` mesh."""
    from aiida.orm import KpointsData

    kpts = KpointsData()
    kpts.set_kpoints_mesh([2, 2, 2])
    return kpts


def _block(label: str, include: range) -> ExplicitProjectionBlock:
    n = len(include)
    return ExplicitProjectionBlock(
        label=label,
        spin=SpinChannel.NONE,
        num_wann=n,
        num_bands=n,
        include_bands=list(include),
        projection_type=WannierProjectionType.ANALYTIC,
        projections=["Si:sp3"],
    )


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
            pw_code=ph_codes["pw"],
            ph_code=ph_codes["ph"],
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
            pw_code=ph_codes["pw"],
            ph_code=ph_codes["ph"],
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
            overrides={
                "ph": {"ph": {"parameters": {"INPUTPH": {"tr2_ph": 1.0e-14, "epsil": False}}}}
            },
        )
        inputph = wg.tasks["ph"].inputs["ph"]["parameters"].value.get_dict()["INPUTPH"]
        assert inputph["tr2_ph"] == pytest.approx(1.0e-14)
        assert inputph["epsil"] is True


class TestSinglepointDFPTAutoEps:
    """eps_inf='auto' prepends the dielectric chain inside SinglepointDFPTWorkflow."""

    def test_auto_adds_dielectric_task(self, ph_codes, silicon_structure, kmesh):
        """A 'dielectric' task appears and the kcw chain is still built."""
        wg = SinglepointDFPTWorkflow.build(
            codes=ph_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": _block("occ", range(1, 5))}},
            kpoints=kmesh,
            kgrid=[2, 2, 2],
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf="auto",
        )
        names = [t.name for t in wg.tasks]
        assert "dielectric" in names
        assert "dfpt" in names

    def test_numeric_eps_skips_dielectric_task(self, ph_codes, silicon_structure, kmesh):
        """A numeric eps_inf builds no dielectric chain."""
        wg = SinglepointDFPTWorkflow.build(
            codes=ph_codes,
            structure=silicon_structure,
            manifolds={"none": {"occ": _block("occ", range(1, 5))}},
            kpoints=kmesh,
            kgrid=[2, 2, 2],
            pseudo_family="SSSP/1.3/PBE/efficiency",
            eps_inf=11.7,
        )
        assert "dielectric" not in [t.name for t in wg.tasks]

    def test_auto_without_ph_code_raises(self, ph_codes, silicon_structure, kmesh):
        """eps_inf='auto' without codes['ph'] fails at build time."""
        codes = {key: value for key, value in ph_codes.items() if key != "ph"}
        with pytest.raises(ValueError, match=r"codes\['ph'\]"):
            SinglepointDFPTWorkflow.build(
                codes=codes,
                structure=silicon_structure,
                manifolds={"none": {"occ": _block("occ", range(1, 5))}},
                kpoints=kmesh,
                kgrid=[2, 2, 2],
                pseudo_family="SSSP/1.3/PBE/efficiency",
                eps_inf="auto",
            )
