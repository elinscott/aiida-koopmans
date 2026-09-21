"""Tests for the shared Koopmans band structure task, ``workgraphs/ui/band_structure.py``.

Three seams: extracting and combining a manifold's printed Koopmans
Hamiltonian, the per-manifold interpolation fan-out and merge that
``KoopmansBandStructureTask`` runs for whatever manifolds its caller
declares, and the validation that turns a malformed ``manifolds`` list into
a named error before any task is built. Which file holds a manifold's
Hamiltonian and how manifolds are partitioned is route-specific and tested
in ``test_ui_dscf.py`` / ``test_dfpt_workgraph.py``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from aiida import orm

from tests.fixtures import (
    _task_names,
    assert_graph_roundtrips,
    block_wannierization,
    occ_emp_manifold_specs,
)


def _kpath():
    kpath = orm.KpointsData()
    kpath.set_kpoints(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]))
    return kpath


def _retrieved_with_hamiltonians(names):
    folder = orm.FolderData()
    for name in names:
        folder.base.repository.put_object_from_filelike(io.BytesIO(b"h"), name)
    return folder.store()


# ----------------------------------------------------------------------
# Seam 1: extracting and combining printed Hamiltonians
# ----------------------------------------------------------------------


class TestExtractKoopmansHamiltonian:
    """Lifting one printed Hamiltonian out of the retrieved folder."""

    def test_a_missing_file_names_the_folder_contents(self, aiida_profile):
        """The run that did not print them is what the reader has to fix."""
        from aiida_koopmans.workgraphs.ui.band_structure import extract_koopmans_hamiltonian

        retrieved = _retrieved_with_hamiltonians(["ham_occ_1.dat"])
        with pytest.raises(ValueError, match=r"ham_emp_1\.dat"):
            extract_koopmans_hamiltonian._callable(
                retrieved=retrieved, filename=orm.Str("ham_emp_1.dat")
            )

    def test_a_present_file_comes_out_under_its_own_name(self, aiida_profile):
        """Negative control: the same folder yields the file it does hold."""
        from aiida_koopmans.workgraphs.ui.band_structure import extract_koopmans_hamiltonian

        retrieved = _retrieved_with_hamiltonians(["ham_occ_1.dat"])
        lifted = extract_koopmans_hamiltonian._callable(
            retrieved=retrieved, filename=orm.Str("ham_occ_1.dat")
        )
        assert lifted.filename == "ham_occ_1.dat"
        assert lifted.get_content() == "h"


class TestMergeManifoldEnergies:
    """The concatenation and the reference energy."""

    @staticmethod
    def _merge(**kwargs):
        from aiida_koopmans.workgraphs.ui.band_structure import merge_manifold_energies

        return merge_manifold_energies._callable(**kwargs)

    def test_an_offset_shifts_both_energies_and_the_reference(self):
        merged = self._merge(occupied=[[1.0, 2.0], [1.1, 2.1]], empty=[[5.0], [5.1]], offset=0.5)
        assert merged["energies"] == [[1.5, 2.5, 5.5], [1.6, 2.6, 5.6]]
        assert merged["reference"] == pytest.approx(2.6)

    def test_the_default_offset_leaves_the_merge_unchanged(self):
        """Negative control: an explicit zero offset is the same as none."""
        assert self._merge(occupied=[[1.0]], empty=[[5.0]], offset=0.0) == self._merge(
            occupied=[[1.0]], empty=[[5.0]]
        )

    def test_occupied_then_empty_within_a_channel(self):
        merged = self._merge(occupied=[[1.0, 2.0], [1.1, 2.1]], empty=[[5.0], [5.1]])
        assert merged["energies"] == [[1.0, 2.0, 5.0], [1.1, 2.1, 5.1]]
        assert merged["reference"] == pytest.approx(2.1)

    def test_spin_channels_stack_on_a_leading_axis(self):
        merged = self._merge(
            occupied=[[1.0], [1.1]],
            empty=[[5.0], [5.1]],
            occupied_down=[[0.9], [3.0]],
            empty_down=[[4.0], [4.1]],
        )
        # (spin, k-point, band): two channels, two k-points, occ + emp.
        assert np.asarray(merged["energies"]).shape == (2, 2, 2)
        # The valence band maximum is the highest occupied energy anywhere.
        assert merged["reference"] == pytest.approx(3.0)

    def test_half_a_spin_polarized_merge_is_refused(self):
        with pytest.raises(ValueError, match="both `occupied_down` and `empty_down`"):
            self._merge(occupied=[[1.0]], empty=[[5.0]], occupied_down=[[1.0]])

    def test_a_down_channel_without_its_empty_manifold_is_refused(self):
        """An occupied-only merge is no way past the check above.

        With no ``empty`` the two ``*_down`` inputs no longer have to
        arrive together for the shapes to work out, so the asymmetry would
        otherwise be dropped rather than raised.
        """
        with pytest.raises(ValueError, match="both `occupied_down` and `empty_down`"):
            self._merge(occupied=[[1.0]], occupied_down=[[0.9]])

    def test_an_occupied_only_merge_returns_the_occupied_bands(self):
        """A run with no empty projections still gets a band structure."""
        merged = self._merge(occupied=[[1.0, 2.0], [1.1, 2.1]])
        assert merged["energies"] == [[1.0, 2.0], [1.1, 2.1]]
        assert merged["reference"] == pytest.approx(2.1)

    def test_channels_with_different_manifolds_are_refused(self):
        """One channel with an empty manifold and one without cannot stack.

        The occupied-only path must not become a way to smuggle a
        half-populated spin-polarized merge past the check above.
        """
        with pytest.raises(ValueError, match="same manifolds"):
            self._merge(occupied=[[1.0]], occupied_down=[[0.9]], empty_down=[[4.0]])

    def test_manifolds_on_different_paths_are_refused(self):
        with pytest.raises(ValueError, match="different k-paths"):
            self._merge(occupied=[[1.0], [1.1]], empty=[[5.0]])

    def test_spin_channels_of_different_widths_are_refused(self):
        """Two channels holding different band counts cannot stack."""
        with pytest.raises(ValueError, match="different shapes"):
            self._merge(
                occupied=[[1.0, 2.0]],
                empty=[[5.0]],
                occupied_down=[[1.0]],
                empty_down=[[5.0]],
            )


# ----------------------------------------------------------------------
# Seam 2: validating the ``manifolds`` partition
# ----------------------------------------------------------------------


class TestChannelManifoldsValidation:
    """``_channel_manifolds`` turns a malformed spec list into a named error."""

    @staticmethod
    def _group(manifolds):
        from aiida_koopmans.workgraphs.ui.band_structure import _channel_manifolds

        return _channel_manifolds(manifolds)

    def test_two_manifolds_cannot_both_be_the_same_filling_and_spin(self):
        with pytest.raises(ValueError, match="Two manifolds both claim"):
            self._group(
                [
                    {"filled": True, "spin": "none", "filename": "a", "blocks": ["x"]},
                    {"filled": True, "spin": "none", "filename": "b", "blocks": ["y"]},
                ]
            )

    def test_an_empty_manifold_needs_its_occupied_one(self):
        with pytest.raises(ValueError, match="no occupied one"):
            self._group([{"filled": False, "spin": "none", "filename": "a", "blocks": ["x"]}])

    def test_spin_none_cannot_mix_with_a_polarized_channel(self):
        with pytest.raises(ValueError, match="mix spin='none'"):
            self._group(
                [
                    {"filled": True, "spin": "none", "filename": "a", "blocks": ["x"]},
                    {"filled": True, "spin": "up", "filename": "b", "blocks": ["y"]},
                ]
            )

    def test_a_down_channel_needs_an_up_channel_to_pair_with(self):
        """Nothing builds a lone down channel.

        Both routes only ever pass one physical channel as ``spin='none'``,
        or pass up and down together.
        """
        with pytest.raises(ValueError, match=r"spin='down'.*no spin='up'"):
            self._group(
                [
                    {"filled": True, "spin": "down", "filename": "a", "blocks": ["x"]},
                    {"filled": False, "spin": "down", "filename": "b", "blocks": ["y"]},
                ]
            )

    def test_an_up_channel_on_its_own_is_not_refused(self):
        """Negative control: it is 'down without up' that is refused, not a lone channel.

        The DFPT route's per-channel call is exactly one physical channel on
        its own (labelled ``spin='none'`` from this task's point of view);
        a caller stating that channel as ``spin='up'`` must not be refused
        for the same reason a lone ``spin='down'`` is.
        """
        grouped = self._group(
            [
                {"filled": True, "spin": "up", "filename": "a", "blocks": ["x"]},
                {"filled": False, "spin": "up", "filename": "b", "blocks": ["y"]},
            ]
        )
        assert set(grouped) == {"up"}

    def test_a_single_occupied_manifold_is_accepted(self):
        """Negative control: an occupied-only run is a valid, one-entry partition."""
        grouped = self._group([{"filled": True, "spin": "none", "filename": "a", "blocks": ["x"]}])
        assert set(grouped) == {"none"}
        assert set(grouped["none"]) == {True}


class TestMisleadingBlockKeysStayStructural:
    """Merge order, the reference channel and centre wiring follow ``filled``/``spin`` alone."""

    def test_a_filled_manifold_named_like_an_empty_one_is_not_swapped(self, silicon_structure):
        """One manifold is ``filled=True`` but its blocks are named ``emp_1``, and vice versa.

        A label-sniffing implementation — inferring the filling from
        whether a block name starts with ``occ``/``emp`` — would read this
        pair backwards: it would extract the empty Hamiltonian for the
        "occupied" stage and vice versa, and would pull Wannier centres
        from the wrong block. Only ``spec["filled"]`` and ``spec["blocks"]``
        may decide either; nothing here may be inferred from how the block
        happens to be spelled.
        """
        from aiida_koopmans.workgraphs.ui.band_structure import KoopmansBandStructureTask

        blocks_by_label = {
            "emp_1": block_wannierization("emp_1", num_wann=2),
            "occ_1": block_wannierization("occ_1", num_wann=2),
        }
        manifolds = [
            # filled=True, but its one block is named as though it were empty.
            {"filled": True, "spin": "none", "filename": "ham_occ_1.dat", "blocks": ["emp_1"]},
            # filled=False, but its one block is named as though it were occupied.
            {"filled": False, "spin": "none", "filename": "ham_emp_1.dat", "blocks": ["occ_1"]},
        ]
        wg = KoopmansBandStructureTask.build(
            structure=silicon_structure,
            koopmans_ham_retrieved=_retrieved_with_hamiltonians(["ham_occ_1.dat", "ham_emp_1.dat"]),
            manifolds=manifolds,
            block_wannierizations=blocks_by_label,
            kgrid=[2, 2, 2],
            kpath=_kpath(),
        )
        by_name = {task.name: task for task in wg.tasks}

        # The task names ("occ"/"emp") and the Hamiltonian each extracts
        # follow ``filled``, never the block spelling.
        assert by_name["extract_occ_hamiltonian"].inputs["filename"].value == "ham_occ_1.dat"
        assert by_name["extract_emp_hamiltonian"].inputs["filename"].value == "ham_emp_1.dat"

        # The "occ" stage's centres come from the block *its spec lists*
        # ("emp_1"), not the block whose name would suggest it.
        occ_centres = by_name["collect_occ_centres"].inputs["output_parameters"]
        assert occ_centres["b00"].value.uuid == blocks_by_label["emp_1"]["output_parameters"].uuid
        emp_centres = by_name["collect_emp_centres"].inputs["output_parameters"]
        assert emp_centres["b00"].value.uuid == blocks_by_label["occ_1"]["output_parameters"].uuid

    def test_the_reference_channel_is_whichever_spin_is_none_or_up_never_by_position(
        self, silicon_structure
    ):
        """The merge's ``occupied``/``occupied_down`` split follows spin, not list order.

        The down channel's specs are listed *before* the up channel's; a
        merge that picked "first" and "second" by list position rather
        than by ``spin`` would stack the channels backwards.
        """
        from aiida_koopmans.workgraphs.ui.band_structure import KoopmansBandStructureTask

        manifolds = [
            {"filled": True, "spin": "down", "filename": "ham_occ_2.dat", "blocks": ["occ_down"]},
            {"filled": False, "spin": "down", "filename": "ham_emp_2.dat", "blocks": ["emp_down"]},
            {"filled": True, "spin": "up", "filename": "ham_occ_1.dat", "blocks": ["occ_up"]},
            {"filled": False, "spin": "up", "filename": "ham_emp_1.dat", "blocks": ["emp_up"]},
        ]
        wg = KoopmansBandStructureTask.build(
            structure=silicon_structure,
            koopmans_ham_retrieved=_retrieved_with_hamiltonians(
                ["ham_occ_1.dat", "ham_emp_1.dat", "ham_occ_2.dat", "ham_emp_2.dat"]
            ),
            manifolds=manifolds,
            block_wannierizations={
                label: block_wannierization(label, num_wann=2)
                for label in ("occ_up", "emp_up", "occ_down", "emp_down")
            },
            kgrid=[2, 2, 2],
            kpath=_kpath(),
        )
        merge = {task.name: task for task in wg.tasks}["merge_manifold_energies"]
        # ``occupied``/``empty`` (undashed) must be the up channel, whatever
        # order the specs arrived in.
        up_link = merge.inputs["occupied"]._links[0]
        assert up_link.from_task.name == "interpolate_occ_up"
        down_link = merge.inputs["occupied_down"]._links[0]
        assert down_link.from_task.name == "interpolate_occ_down"


class TestManifoldsRoundTrip:
    """A ``list[ManifoldSpec]`` graph input must survive a WorkGraph dict round trip."""

    @staticmethod
    def _build(manifolds, *, silicon_structure, blocks):
        from aiida_koopmans.workgraphs.ui.band_structure import KoopmansBandStructureTask

        filenames = sorted({spec["filename"] for spec in manifolds})
        return KoopmansBandStructureTask.build(
            structure=silicon_structure,
            koopmans_ham_retrieved=_retrieved_with_hamiltonians(filenames),
            manifolds=manifolds,
            block_wannierizations={
                label: block_wannierization(label, num_wann=2) for label in blocks
            },
            kgrid=[2, 2, 2],
            kpath=_kpath(),
        )

    def test_one_manifold(self, silicon_structure):
        manifolds = [
            {"filled": True, "spin": "none", "filename": "ham_occ_1.dat", "blocks": ["occ"]}
        ]
        wg = self._build(manifolds, silicon_structure=silicon_structure, blocks=["occ"])
        assert_graph_roundtrips(wg)

    def test_two_manifolds(self, silicon_structure):
        wg = self._build(
            occ_emp_manifold_specs(), silicon_structure=silicon_structure, blocks=["occ", "emp"]
        )
        assert_graph_roundtrips(wg)

    def test_four_manifolds(self, silicon_structure):
        manifolds = occ_emp_manifold_specs(
            spin="up", filenames=("ham_occ_1.dat", "ham_emp_1.dat"), blocks=("occ_up", "emp_up")
        ) + occ_emp_manifold_specs(
            spin="down",
            filenames=("ham_occ_2.dat", "ham_emp_2.dat"),
            blocks=("occ_down", "emp_down"),
        )
        wg = self._build(
            manifolds,
            silicon_structure=silicon_structure,
            blocks=["occ_up", "emp_up", "occ_down", "emp_down"],
        )
        assert_graph_roundtrips(wg)


# ----------------------------------------------------------------------
# Seam 3: the per-manifold fan-out and the merge
# ----------------------------------------------------------------------


class TestManifoldFanOut:
    """One interpolation per manifold, merged into one band structure."""

    def _build(self, silicon_structure, **overrides):
        from aiida_koopmans.workgraphs.ui.band_structure import KoopmansBandStructureTask

        inputs = {
            "structure": silicon_structure,
            "manifolds": occ_emp_manifold_specs(),
            "block_wannierizations": {
                label: block_wannierization(label, num_wann=2) for label in ("occ", "emp")
            },
            "koopmans_ham_retrieved": _retrieved_with_hamiltonians(
                ["ham_occ_1.dat", "ham_emp_1.dat"]
            ),
            "kgrid": [2, 2, 2],
            "kpath": _kpath(),
        }
        inputs.update(overrides)
        return KoopmansBandStructureTask.build(**inputs)

    def test_one_interpolation_per_manifold(self, silicon_structure):
        wg = self._build(silicon_structure)
        names = _task_names(wg)
        assert "interpolate_occ" in names
        assert "interpolate_emp" in names
        assert "merge_manifold_energies" in names
        assert "build_band_structure" in names

    def test_each_manifold_reads_its_own_printed_hamiltonian(self, silicon_structure):
        """The occupied and empty stages must not read the same file."""
        wg = self._build(silicon_structure)
        by_name = {task.name: task for task in wg.tasks}
        assert by_name["extract_occ_hamiltonian"].inputs["filename"].value == "ham_occ_1.dat"
        assert by_name["extract_emp_hamiltonian"].inputs["filename"].value == "ham_emp_1.dat"

    def test_centres_come_from_the_parsed_outputs_not_a_wout(self, silicon_structure):
        """Threading the parser's table, not re-reading a retrieved folder."""
        wg = self._build(silicon_structure)
        by_name = {task.name: task for task in wg.tasks}
        collected = by_name["collect_occ_centres"]
        namespace = collected.inputs["output_parameters"]
        # The centres arrive as the parsed wannier90 Dict of the manifold's
        # one block, keyed in band order.
        assert sorted(namespace._sockets) == ["b00"]
        assert by_name["interpolate_occ"].inputs["centres"]._links

    def test_spin_polarized_fans_out_over_both_channels(self, silicon_structure):
        manifolds = occ_emp_manifold_specs(
            spin="up", filenames=("ham_occ_1.dat", "ham_emp_1.dat"), blocks=("occ_up", "emp_up")
        ) + occ_emp_manifold_specs(
            spin="down",
            filenames=("ham_occ_2.dat", "ham_emp_2.dat"),
            blocks=("occ_down", "emp_down"),
        )
        wg = self._build(
            silicon_structure,
            manifolds=manifolds,
            block_wannierizations={
                label: block_wannierization(label, num_wann=2)
                for label in ("occ_up", "emp_up", "occ_down", "emp_down")
            },
            koopmans_ham_retrieved=_retrieved_with_hamiltonians(
                ["ham_occ_1.dat", "ham_emp_1.dat", "ham_occ_2.dat", "ham_emp_2.dat"]
            ),
        )
        names = _task_names(wg)
        assert {"interpolate_occ_up", "interpolate_emp_up"} <= set(names)
        assert {"interpolate_occ_down", "interpolate_emp_down"} <= set(names)
        by_name = {task.name: task for task in wg.tasks}
        # kcp.x indexes the down channel 2.
        assert by_name["extract_occ_down_hamiltonian"].inputs["filename"].value == "ham_occ_2.dat"
        assert by_name["merge_manifold_energies"].inputs["occupied_down"]._links

    def test_do_dos_gates_the_dos_task(self, silicon_structure):
        assert "interpolated_dos" in _task_names(self._build(silicon_structure, do_dos=True))
        assert "interpolated_dos" not in _task_names(self._build(silicon_structure, do_dos=False))

    def test_do_dos_defaults_to_off(self, silicon_structure):
        """Negative control: the DFPT route relies on this default; pin it directly."""
        assert "interpolated_dos" not in _task_names(self._build(silicon_structure))

    def test_use_ws_distance_reaches_every_manifolds_interpolation(self, silicon_structure):
        """A caller turning the Wigner-Seitz phase off turns it off for every manifold.

        Both routes thread this from the calculation that printed the
        Koopmans Hamiltonian (kcp.x/kcw.x's own ``use_ws_distance``), so the
        two interpolations of the same Hamiltonian must agree on it.
        """
        wg = self._build(silicon_structure, use_ws_distance=False)
        by_name = {task.name: task for task in wg.tasks}
        assert by_name["interpolate_occ"].inputs["use_ws_distance"].value is False
        assert by_name["interpolate_emp"].inputs["use_ws_distance"].value is False

    def test_use_ws_distance_defaults_to_on(self, silicon_structure):
        """Negative control: it is the caller's ``False`` that changes the wiring."""
        wg = self._build(silicon_structure)
        by_name = {task.name: task for task in wg.tasks}
        assert by_name["interpolate_occ"].inputs["use_ws_distance"].value is True


class TestOffsetWiring:
    """The ``offset`` graph input reaches the merge task, defaulting to 0.0."""

    @staticmethod
    def _build(silicon_structure, **overrides):
        from aiida_koopmans.workgraphs.ui.band_structure import KoopmansBandStructureTask

        inputs = {
            "structure": silicon_structure,
            "manifolds": occ_emp_manifold_specs(),
            "block_wannierizations": {
                label: block_wannierization(label, num_wann=2) for label in ("occ", "emp")
            },
            "koopmans_ham_retrieved": _retrieved_with_hamiltonians(
                ["ham_occ_1.dat", "ham_emp_1.dat"]
            ),
            "kgrid": [2, 2, 2],
            "kpath": _kpath(),
        }
        inputs.update(overrides)
        return KoopmansBandStructureTask.build(**inputs)

    def test_an_offset_reaches_the_merge_task(self, silicon_structure):
        wg = self._build(silicon_structure, offset=2.5315)
        by_name = {task.name: task for task in wg.tasks}
        assert by_name["merge_manifold_energies"].inputs["offset"]._links

    def test_without_an_offset_the_merge_task_gets_zero(self, silicon_structure):
        """Negative control: no offset means the producing route's own scale, i.e. 0.0."""
        wg = self._build(silicon_structure)
        assert wg.inputs["offset"].value == 0.0
        by_name = {task.name: task for task in wg.tasks}
        links = by_name["merge_manifold_energies"].inputs["offset"]._links
        assert len(links) == 1
        assert links[0].from_socket._name == "offset"

    def test_the_graph_survives_a_dict_round_trip(self, silicon_structure):
        wg = self._build(silicon_structure, offset=2.5315)
        assert_graph_roundtrips(wg)


class TestRunAgainstTheSiliconReference:
    """Execute the whole task end to end on the stored silicon fixtures.

    Every task here is pure python, so the graph runs end to end. The same
    Hamiltonian and centres stand in for both manifolds, so the merged band
    structure must be one manifold's eigenvalues concatenated with
    themselves — which no partial wiring produces. The eigenvalues
    themselves come from the interpolation helper the ``test_ui_helpers``
    suite pins against the stored reference; what is under test here is
    the extraction, the centre threading and the merge around it.
    """

    @staticmethod
    def _build(si_reference, **overrides):
        from pathlib import Path

        from aiida_koopmans.workgraphs.ui import helpers as ui_helpers
        from aiida_koopmans.workgraphs.ui.band_structure import KoopmansBandStructureTask

        data_dir = Path(__file__).parent / "data" / "ui"
        centres = ui_helpers.parse_wout_centers((data_dir / "wann.wout").read_text())
        ham = (data_dir / "kc_ham.dat").read_text()

        structure = orm.StructureData(cell=si_reference["cell"])
        structure.append_atom(position=(0.0, 0.0, 0.0), symbols="Si")
        structure.append_atom(position=(1.3575, 1.3575, 1.3575), symbols="Si")

        kpath = orm.KpointsData()
        kpath.set_kpoints(np.array(si_reference["kpath_kpts"]))

        retrieved = orm.FolderData()
        for name in ("ham_occ_1.dat", "ham_emp_1.dat"):
            retrieved.base.repository.put_object_from_filelike(io.BytesIO(ham.encode()), name)
        retrieved.store()

        parsed = orm.Dict(
            {
                "number_wfs": len(centres),
                "wannier_functions_output": [
                    {"wf_ids": i + 1, "wf_centres": list(centre), "wf_spreads": 1.0}
                    for i, centre in enumerate(centres.tolist())
                ],
            }
        ).store()

        inputs = {
            "structure": structure,
            "manifolds": occ_emp_manifold_specs(),
            "block_wannierizations": {
                label: {**block_wannierization(label), "output_parameters": parsed}
                for label in ("occ", "emp")
            },
            "koopmans_ham_retrieved": retrieved,
            "kgrid": list(si_reference["kgrid"]),
            "kpath": kpath,
            "do_dos": False,
        }
        inputs.update(overrides)
        wg = KoopmansBandStructureTask.build(**inputs)
        wg.run()

        expected = ui_helpers.unfold_and_interpolate(
            hr_content=ham,
            centers=centres,
            cell=np.asarray(si_reference["cell"], dtype=float),
            kgrid=tuple(int(n) for n in si_reference["kgrid"]),
            kpath_kpts=np.asarray(si_reference["kpath_kpts"], dtype=float),
        )
        return wg, expected

    def test_the_merged_bands_are_the_manifolds_concatenated(self, aiida_profile, si_reference):
        wg, expected = self._build(si_reference)

        bands = wg.tasks.build_band_structure.outputs.result.value
        assert np.allclose(
            bands.get_bands(), np.concatenate([expected, expected], axis=1), atol=1e-10
        )
        assert bands.units == "eV"
        # The valence-band maximum is the top of the occupied manifold.
        assert wg.tasks.merge_manifold_energies.outputs.reference.value == pytest.approx(
            float(expected.max())
        )

    def test_the_pw_scale_offset_shifts_bands_and_reference(self, aiida_profile, si_reference):
        """Executed end to end: the offset must reach both merge outputs.

        ``compute_dos_from_bands`` reads the same ``energies`` output the
        assertion below checks, so a shift proven here reaches the DOS
        input too without a separate execution.
        """
        offset = 2.5317
        wg, expected = self._build(si_reference, offset=offset)

        bands = wg.tasks.build_band_structure.outputs.result.value
        assert np.allclose(
            bands.get_bands(),
            np.concatenate([expected, expected], axis=1) + offset,
            atol=1e-10,
        )
        assert wg.tasks.merge_manifold_energies.outputs.reference.value == pytest.approx(
            float(expected.max()) + offset
        )


class TestSmoothInterpolationWiring:
    """A second, denser-mesh wannierization swaps in the smooth correction."""

    @staticmethod
    def _build(silicon_structure, *, smooth: bool, blocks_per_manifold: int = 1):
        from aiida_koopmans.workgraphs.ui.band_structure import KoopmansBandStructureTask

        labels = {
            manifold: [
                manifold if blocks_per_manifold == 1 else f"{manifold}_{index}"
                for index in range(blocks_per_manifold)
            ]
            for manifold in ("occ", "emp")
        }
        manifolds = [
            {
                "filled": True,
                "spin": "none",
                "filename": "ham_occ_1.dat",
                "blocks": labels["occ"],
            },
            {
                "filled": False,
                "spin": "none",
                "filename": "ham_emp_1.dat",
                "blocks": labels["emp"],
            },
        ]
        every_label = labels["occ"] + labels["emp"]
        inputs = {
            "structure": silicon_structure,
            "manifolds": manifolds,
            "block_wannierizations": {
                label: block_wannierization(label, num_wann=2) for label in every_label
            },
            "koopmans_ham_retrieved": _retrieved_with_hamiltonians(
                ["ham_occ_1.dat", "ham_emp_1.dat"]
            ),
            "kgrid": [2, 2, 2],
            "kpath": _kpath(),
        }
        if smooth:
            inputs["smooth_block_wannierizations"] = {
                label: block_wannierization(f"{label}_smooth", num_wann=2) for label in every_label
            }
        return KoopmansBandStructureTask.build(**inputs)

    def test_both_dft_hamiltonians_reach_the_interpolation(self, silicon_structure):
        wg = self._build(silicon_structure, smooth=True)
        interpolate = {task.name: task for task in wg.tasks}["interpolate_occ"]
        assert interpolate.inputs["dft_ham_file"]._links
        assert interpolate.inputs["dft_smooth_ham_file"]._links

    def test_without_a_smooth_run_neither_reaches_it(self, silicon_structure):
        """Pin that no smooth wannierization leaves neither DFT socket linked.

        ``helpers.unfold_and_interpolate`` (the wired entry point) ignores a
        coarse Hamiltonian given without its dense counterpart: it only
        switches on the smooth-interpolation subtraction when the dense one
        is present. A coarse socket linked alone would therefore be an inert
        no-op rather than a silent shift, but the graph should still never
        produce that half-wired state.
        """
        wg = self._build(silicon_structure, smooth=False)
        interpolate = {task.name: task for task in wg.tasks}["interpolate_occ"]
        assert not interpolate.inputs["dft_ham_file"]._links
        assert not interpolate.inputs["dft_smooth_ham_file"]._links

    def test_a_one_block_manifold_passes_its_hamiltonian_through(self, silicon_structure):
        """Nothing to combine, so no combining task is added."""
        names = _task_names(self._build(silicon_structure, smooth=True))
        assert not [name for name in names if name.startswith("merge_occ")]

    def test_a_multi_block_manifold_combines_both_hamiltonians(self, silicon_structure):
        """Coarse and dense are each block-diagonal over the manifold's blocks."""
        wg = self._build(silicon_structure, smooth=True, blocks_per_manifold=2)
        by_name = {task.name: task for task in wg.tasks}
        assert "merge_occ_dft_hamiltonian" in by_name
        assert "merge_occ_smooth_dft_hamiltonian" in by_name
        # Band order travels as the key order: one linked socket per block.
        for name in ("merge_occ_dft_hamiltonian", "merge_occ_smooth_dft_hamiltonian"):
            combined = by_name[name]
            assert combined.inputs["b00"]._links
            assert combined.inputs["b01"]._links

    def test_the_spec_order_is_the_band_order(self, silicon_structure):
        """``b00`` is the spec's first block, whatever the enumeration does internally.

        ``interpolate_manifold`` enumerates a manifold's ``blocks`` once for
        the centres and once for each Hamiltonian merge; reversing (or
        otherwise reordering) that enumeration would put the Koopmans
        Hamiltonian's rows and the centre table's entries on different
        Wannier functions, with no error anywhere.
        """
        from aiida_koopmans.workgraphs.ui.band_structure import KoopmansBandStructureTask

        # Reverse alphabetical, so code that sorted the labels rather than
        # following the spec's own list would stack them the other way round.
        labels = ["occ_b", "occ_a"]
        blocks = {label: block_wannierization(label, num_wann=2) for label in labels}
        smooth_blocks = {
            label: block_wannierization(f"{label}_smooth", num_wann=2) for label in labels
        }
        wg = KoopmansBandStructureTask.build(
            structure=silicon_structure,
            koopmans_ham_retrieved=_retrieved_with_hamiltonians(["ham_occ_1.dat", "ham_emp_1.dat"]),
            manifolds=[
                {"filled": True, "spin": "none", "filename": "ham_occ_1.dat", "blocks": labels},
                {"filled": False, "spin": "none", "filename": "ham_emp_1.dat", "blocks": ["emp"]},
            ],
            block_wannierizations={**blocks, "emp": block_wannierization("emp", num_wann=2)},
            smooth_block_wannierizations={
                **smooth_blocks,
                "emp": block_wannierization("emp_smooth", num_wann=2),
            },
            kgrid=[2, 2, 2],
            kpath=_kpath(),
        )
        by_name = {task.name: task for task in wg.tasks}

        for step, source in (
            ("merge_occ_dft_hamiltonian", blocks),
            ("merge_occ_smooth_dft_hamiltonian", smooth_blocks),
        ):
            merged = by_name[step].inputs
            for index, label in enumerate(labels):
                assert merged[f"b{index:02d}"].value.uuid == source[label]["hr_file"].uuid, (
                    f"{step} put {label} on the wrong row"
                )

        centres = by_name["collect_occ_centres"].inputs["output_parameters"]
        for index, label in enumerate(labels):
            assert centres[f"b{index:02d}"].value.uuid == blocks[label]["output_parameters"].uuid, (
                f"collect_occ_centres put {label} on the wrong row"
            )

    def test_the_key_order_is_the_band_order(self, aiida_profile):
        """``b00`` before ``b01`` on the diagonal, whatever order they arrive in."""
        from aiida_koopmans.workgraphs.ui import helpers as ui_helpers
        from aiida_koopmans.workgraphs.ui.band_structure import manifold_hamiltonian
        from aiida_koopmans.workgraphs.utils.wannier_merge import (
            generate_wannier_hr_file_contents,
        )

        def _block(value):
            content = generate_wannier_hr_file_contents(
                np.array([[[value + 0.0j]]]), np.array([[0, 0, 0]]), [1]
            )
            return orm.SinglefileData(io.StringIO(content), filename="aiida_hr.dat").store()

        # Passed b01 first: the ordering must come from the keys, not the
        # call order.
        merged = manifold_hamiltonian._callable(b01=_block(-2.0), b00=_block(-1.0))
        hr, _rvect, _weights, _nrpts = ui_helpers.parse_hr_file_contents(merged.get_content("r"))
        assert np.allclose(hr.reshape(2, 2).diagonal(), [-1.0, -2.0])

    def test_the_graph_survives_a_dict_round_trip(self, silicon_structure):
        """The smooth input is a typed dynamic namespace, which has broken this before."""
        wg = self._build(silicon_structure, smooth=True)
        assert_graph_roundtrips(wg)
