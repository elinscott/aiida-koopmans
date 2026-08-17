"""Tests for the ΔSCF band-structure stage in ``workgraphs/ui/dscf.py``.

Two seams: the final KI printing and retrieving its Koopmans Hamiltonians,
and the per-manifold fan-out that interpolates them into one band structure.
"""

from __future__ import annotations

import numpy as np
import pytest
from aiida import orm

from aiida_koopmans.functionals import Correction
from aiida_koopmans.spin import SpinChannel
from aiida_koopmans.variational_orbitals import VariationalOrbitalType
from aiida_koopmans.workgraphs.kcp import _validate_scope
from tests.fixtures import block_wannierization, occ_emp_merge_groups


def _task_names(wg) -> list[str]:
    """Return every task name in a built graph, walking nested graphs."""
    names: list[str] = []

    def _walk(tasks):
        for task in tasks:
            names.append(task.name)
            children = getattr(task, "children", None)
            if children:
                _walk(children)

    _walk(wg.tasks)
    return names


# ----------------------------------------------------------------------
# Seam 1: kcp.x prints and retrieves the Koopmans Hamiltonians
# ----------------------------------------------------------------------


class TestFinalKiWritesTheHamiltonians:
    """``write_hr`` is what makes the one indispensable UI input exist."""

    def _build(self, *, ozone_structure, kcp_code, ozone_pseudo_family, write_hr):
        from aiida_pseudo.groups.family import PseudoPotentialFamily

        from aiida_koopmans.workgraphs.kcp import RunFinalKI

        family = (
            orm.QueryBuilder()
            .append(PseudoPotentialFamily, filters={"label": ozone_pseudo_family})
            .one()[0]
        )
        return RunFinalKI.build(
            kcp_code=kcp_code,
            structure=ozone_structure,
            pseudos=family.get_pseudos(structure=ozone_structure),
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            nelec=18,
            nelup=9,
            neldw=9,
            correction=Correction.KI,
            alphas={"filled": {SpinChannel.NONE: [0.6] * 9}, "empty": {SpinChannel.NONE: [0.6]}},
            parent_folder=orm.RemoteData(remote_path="/nonexistent/fake"),
            write_hr=write_hr,
        )

    @staticmethod
    def _kcp_task(wg):
        return next(task for task in wg.tasks if "Kcp" in task.identifier)

    def test_write_hr_asks_kcp_for_the_hamiltonians_and_keeps_them(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        wg = self._build(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            write_hr=True,
        )
        task = self._kcp_task(wg)
        parameters = task.inputs["parameters"].value
        assert parameters["CONTROL"]["write_hr"] is True
        settings = task.inputs["settings"].value.get_dict()
        assert settings["additional_retrieve_list"] == ["ham_occ_*.dat", "ham_emp_*.dat"]

    def test_without_write_hr_nothing_is_printed_or_kept(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Negative control: the default final KI is untouched by the new knob."""
        wg = self._build(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            write_hr=False,
        )
        task = self._kcp_task(wg)
        assert task.inputs["parameters"].value["CONTROL"]["write_hr"] is False
        assert task.inputs["settings"].value is None


class TestHamiltonianFilenames:
    """The names kcp.x writes, per QE ``CPV/write_hamiltonian.f90``."""

    def test_names_follow_the_manifold_and_spin_index(self):
        from aiida_koopmans.workgraphs.kcp import koopmans_hamiltonian_filename

        assert koopmans_hamiltonian_filename(filled=True, spin_index=1) == "ham_occ_1.dat"
        assert koopmans_hamiltonian_filename(filled=False, spin_index=1) == "ham_emp_1.dat"
        assert koopmans_hamiltonian_filename(filled=True, spin_index=2) == "ham_occ_2.dat"
        assert koopmans_hamiltonian_filename(filled=False, spin_index=2) == "ham_emp_2.dat"

    def test_a_third_spin_index_is_rejected(self):
        from aiida_koopmans.workgraphs.kcp import koopmans_hamiltonian_filename

        with pytest.raises(ValueError, match="spin_index"):
            koopmans_hamiltonian_filename(filled=True, spin_index=3)


# ----------------------------------------------------------------------
# Seam 2: the per-manifold fan-out and the merge
# ----------------------------------------------------------------------


def _kpath():
    kpath = orm.KpointsData()
    kpath.set_kpoints(np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]))
    return kpath


def _retrieved_with_hamiltonians(names):
    import io

    folder = orm.FolderData()
    for name in names:
        folder.base.repository.put_object_from_filelike(io.BytesIO(b"h"), name)
    return folder.store()


class TestManifoldFanOut:
    """One interpolation per (filling, spin), merged into one band structure."""

    def _build(self, silicon_structure, **overrides):
        from aiida_koopmans.workgraphs.ui.dscf import DscfBandStructureTask

        inputs = {
            "structure": silicon_structure,
            "merge_groups": occ_emp_merge_groups(),
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
        return DscfBandStructureTask.build(**inputs)

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
        merge_groups = occ_emp_merge_groups("up") + occ_emp_merge_groups("down")
        merge_groups[0]["blocks"] = [{"label": "occ_up"}]
        merge_groups[1]["blocks"] = [{"label": "emp_up"}]
        merge_groups[2]["blocks"] = [{"label": "occ_down"}]
        merge_groups[3]["blocks"] = [{"label": "emp_down"}]
        wg = self._build(
            silicon_structure,
            merge_groups=merge_groups,
            spin_polarized=True,
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

    def test_a_missing_manifold_names_itself(self, silicon_structure):
        """Interpolating needs an occupied and an empty manifold per channel."""
        with pytest.raises(ValueError, match="occupied and an empty projection manifold"):
            self._build(
                silicon_structure,
                merge_groups=[{"filled": True, "spin": "none", "blocks": [{"label": "occ"}]}],
            )


class TestRunAgainstTheSiliconReference:
    """Execute the whole stage in-process on the stored silicon fixtures.

    Every task here is pure python, so the graph runs end to end. The same
    Hamiltonian and centres stand in for both manifolds, so the merged band
    structure must be one manifold's eigenvalues concatenated with
    themselves — which no partial wiring produces. The eigenvalues
    themselves come from the interpolation helper the ``test_ui_helpers``
    suite pins against the stored reference; what is under test here is
    the extraction, the centre threading and the merge around it.
    """

    def test_the_merged_bands_are_the_manifolds_concatenated(self, aiida_profile, si_reference):
        import io
        from pathlib import Path

        from aiida_koopmans.workgraphs.ui import helpers as ui_helpers
        from aiida_koopmans.workgraphs.ui.dscf import DscfBandStructureTask

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

        wg = DscfBandStructureTask.build(
            structure=structure,
            merge_groups=occ_emp_merge_groups(),
            block_wannierizations={
                label: {**block_wannierization(label), "output_parameters": parsed}
                for label in ("occ", "emp")
            },
            koopmans_ham_retrieved=retrieved,
            kgrid=list(si_reference["kgrid"]),
            kpath=kpath,
            do_dos=False,
        )
        wg.run()

        expected = ui_helpers.unfold_and_interpolate(
            hr_content=ham,
            centers=centres,
            cell=np.asarray(si_reference["cell"], dtype=float),
            kgrid=tuple(int(n) for n in si_reference["kgrid"]),
            kpath_kpts=np.asarray(si_reference["kpath_kpts"], dtype=float),
        )
        bands = wg.tasks.build_band_structure.outputs.result.value
        assert np.allclose(
            bands.get_bands(), np.concatenate([expected, expected], axis=1), atol=1e-10
        )
        assert bands.units == "eV"
        # The valence-band maximum is the top of the occupied manifold.
        assert wg.tasks.merge_manifold_energies.outputs.reference.value == pytest.approx(
            float(expected.max())
        )


class TestMergeManifoldEnergies:
    """The concatenation and the reference energy."""

    @staticmethod
    def _merge(**kwargs):
        from aiida_koopmans.workgraphs.ui.dscf import merge_manifold_energies

        return merge_manifold_energies._callable(**kwargs)

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

    def test_manifolds_on_different_paths_are_refused(self):
        with pytest.raises(ValueError, match="different k-paths"):
            self._merge(occupied=[[1.0], [1.1]], empty=[[5.0]])


# ----------------------------------------------------------------------
# The workflow-level gate
# ----------------------------------------------------------------------


class TestCalculateBandsScope:
    def test_the_molecular_route_cannot_serve_it(self, ozone_structure):
        with pytest.raises(NotImplementedError, match="Wannier basis"):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                fix_spin_contamination=False,
                structure=ozone_structure,
                calculate_bands=True,
            )

    def test_a_path_is_required(self, periodic_ozone_structure, kmesh):
        from tests.fixtures import ozone_projection_blocks

        with pytest.raises(ValueError, match="needs the band path"):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals=VariationalOrbitalType.MLWFS,
                fix_spin_contamination=False,
                structure=periodic_ozone_structure,
                blocks=ozone_projection_blocks(),
                kgrid=[2, 1, 1],
                kpoints=kmesh,
                calculate_bands=True,
            )

    def test_a_path_satisfies_it(self, periodic_ozone_structure, kmesh):
        from tests.fixtures import ozone_projection_blocks

        _validate_scope(
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            fix_spin_contamination=False,
            structure=periodic_ozone_structure,
            blocks=ozone_projection_blocks(),
            kgrid=[2, 1, 1],
            kpoints=kmesh,
            calculate_bands=True,
            kpath=_kpath(),
        )


class TestInterpolationKnobs:
    @staticmethod
    def _resolve(knobs, *, calculate_bands=True):
        from aiida_koopmans.workgraphs.kcp import _resolve_band_interpolation_knobs

        return _resolve_band_interpolation_knobs(knobs, calculate_bands=calculate_bands)

    def test_smooth_interpolation_is_refused_by_name(self):
        with pytest.raises(NotImplementedError, match="smooth_int_factor"):
            self._resolve({"smooth_int_factor": [2, 2, 2]})

    def test_an_unscaled_factor_passes_and_the_knobs_come_through(self):
        assert self._resolve({"smooth_int_factor": [1, 1, 1], "do_dos": False}) == (True, False)

    def test_the_defaults_are_ws_distance_and_a_dos(self):
        assert self._resolve(None) == (True, True)

    def test_knobs_without_the_stage_are_refused(self):
        with pytest.raises(ValueError, match="calculate_bands is off"):
            self._resolve({"do_dos": True}, calculate_bands=False)


class TestInterpolateBandsCentres:
    """The interpolation takes centres, never a ``.wout`` to re-parse."""

    def test_an_unread_centre_is_named(self, aiida_profile, silicon_structure):
        from aiida_koopmans.workgraphs.ui import interpolate_bands

        with pytest.raises(ValueError, match="unread coordinate"):
            interpolate_bands._callable(
                kc_ham_file=orm.SinglefileData.from_string("x"),
                centres=[[0.0, None, 0.0]],
                structure=silicon_structure,
                kpath=_kpath(),
                kgrid=[1, 1, 1],
            )
