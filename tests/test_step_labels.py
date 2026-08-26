"""Every step a display shows carries its own name, and naming costs nothing.

Two things have to hold for the workgraphs to name their own steps. The
name has to reach the process — it rides ``metadata.label``, which the
engine writes onto the node — and it has to be free of consequence, or a
rename would invalidate every cached calculation underneath it.

Both hold for a calculation, a wrapped workchain, and — since
aiida-workgraph ``5b140d4`` — a sub-graph; that engine guarantee is
tested upstream (aiida-workgraph #811), not here.
"""

from __future__ import annotations

from aiida import orm
from aiida.engine import calcfunction
from aiida.manage.caching import enable_caching
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.projections import block_display_name
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.variational_orbitals import display_name_for
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlock
from aiida_koopmans.workgraphs.ph import DielectricTask
from aiida_koopmans.workgraphs.wannier90 import Wannierize
from tests.fixtures import explicit_block


@calcfunction
def _add(x, y):
    """Add two integers (a stand-in for any cacheable calculation)."""
    return orm.Int(x + y)


class TestLabelsReachTheStep:
    """A built graph carries each step's name on the process that runs it."""

    def test_both_the_workchain_and_its_calculation_are_named(
        self, ph_codes, silicon_structure, fake_cutoffs_family
    ):
        """A restart splits one step into two processes; both answer to the step's name."""
        wg = DielectricTask.build(
            codes={"pw": ph_codes["pw"], "ph": ph_codes["ph"]},
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
        )
        assert wg.tasks["scf"].inputs["metadata"]["label"].value == "SCF"
        assert wg.tasks["scf"].inputs["pw"]["metadata"]["label"].value == "SCF"
        assert wg.tasks["ph"].inputs["metadata"]["label"].value == "Dielectric response"
        assert wg.tasks["ph"].inputs["ph"]["metadata"]["label"].value == "Dielectric response"


class TestWannier90PreprocessingAndMinimizationLabels:
    """The wannier90.x -pp pass and minimization run each carry their own name.

    Both ``Wannierize`` (whole-manifold) and ``WannierizeBlock`` (per-block)
    submit through the same ``Wannier90WorkChain``, which exposes the -pp
    pass's naming as a metadata-only ``wannier90_pp`` namespace alongside the
    minimization run's own ``wannier90`` namespace (aiida-wannier90-workflows
    fork carrying the ``preserve-caller-metadata`` fix) -- previously one
    shared ``wannier90`` namespace, off which neither step could be named on
    its own. ``wannier90_pp`` carries no physics inputs: the -pp pass and
    the minimization run are two phases of one calculation, so both read
    their physics from ``wannier90`` alone.
    """

    def test_the_whole_manifold_route_labels_both_steps(
        self, wannier_codes, silicon_structure, fake_cutoffs_family
    ):
        wg = Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
        )
        step = wg.tasks["Wannier90WorkChain"].inputs
        assert step["wannier90"]["metadata"]["label"].value == "Minimization"
        assert step["wannier90"]["wannier90"]["metadata"]["label"].value == "Minimization"
        assert step["wannier90_pp"]["metadata"]["label"].value == "Preprocessing"
        # The pw2wannier90 step's own label is untouched by this change.
        assert step["pw2wannier90"]["metadata"]["label"].value == "Overlaps"

    def test_the_per_block_route_labels_both_steps(
        self,
        wannier_codes,
        silicon_structure,
        kmesh,
        aiida_localhost,
        tmp_path,
        fake_cutoffs_family,
    ):
        nscf_scratch = orm.RemoteData(computer=aiida_localhost, remote_path=str(tmp_path))
        block = explicit_block("block_1", range(1, 5), projections=["Si: sp3"])
        wg = WannierizeBlock.build(
            codes=wannier_codes,
            structure=silicon_structure,
            block=block,
            projection_type=WannierProjectionType.ANALYTIC,
            nscf_remote_folder=nscf_scratch,
            kpoints=kmesh,
            pseudo_family=fake_cutoffs_family.label,
        )
        step = wg.tasks["wannier90"].inputs
        assert step["wannier90"]["metadata"]["label"].value == "Minimization"
        assert step["wannier90"]["wannier90"]["metadata"]["label"].value == "Minimization"
        assert step["wannier90_pp"]["metadata"]["label"].value == "Preprocessing"
        # The pw2wannier90 step's own label is untouched by this change.
        assert step["pw2wannier90"]["metadata"]["label"].value == "Overlaps"


class TestLabelsAreFreeOfConsequence:
    """Naming a step must not cost a cache hit."""

    def test_a_renamed_calculation_still_hits_the_cache(self, aiida_profile_clean):
        """The second run reuses the first, whose only difference is its name.

        ``metadata.label`` is stored under the hash-ignored
        ``metadata_inputs`` attribute and copied onto the node's ``label``
        column, and neither takes part in ``get_objects_to_hash``. A node
        stored from the cache then takes the source's name along with the
        rest of it, so a step restored from an earlier run reads as that
        run named it.
        """
        first = _add(orm.Int(2), orm.Int(3), metadata={"label": "Sum"})
        with enable_caching():
            second = _add(orm.Int(2), orm.Int(3), metadata={"label": "A different name"})

        first_node = first.base.links.get_incoming().one().node
        second_node = second.base.links.get_incoming().one().node
        assert second_node.base.caching.get_cache_source() == first_node.uuid
        assert first_node.label == "Sum"
        assert second_node.label == "Sum"


class TestReadableNames:
    """The names built from a block's or an orbital's own identity."""

    def test_a_single_block_manifold_is_named_without_an_index(self):
        """One block per manifold carries the bare ``occ`` / ``emp`` label."""
        assert block_display_name({"label": "emp", "spin": SpinChannel.NONE}) == "empty block"

    def test_a_collinear_block_says_which_channel_it_is(self):
        """A collinear run wannierizes each channel separately."""
        assert (
            block_display_name({"label": "occ_up_2", "spin": SpinChannel.UP})
            == "occupied block 2, spin up"
        )

    def test_a_merge_group_supplies_the_channel_its_blocks_omit(self):
        """Inside a merge group the channel is the group's, not the block's."""
        assert (
            block_display_name({"label": "occ_1"}, SpinChannel.DOWN)
            == "occupied block 1, spin down"
        )

    def test_an_orbital_is_named_by_its_band_index(self):
        """The per-spin 1-based band index, with the channel where there is one."""
        assert display_name_for({"spin": SpinChannel.NONE, "index": 5}) == "Orbital 5"
        assert display_name_for({"spin": SpinChannel.UP, "index": 10}) == "Orbital 10 (spin up)"
