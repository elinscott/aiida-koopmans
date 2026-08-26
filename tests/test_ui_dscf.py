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
from tests.fixtures import block_wannierization, explicit_block, occ_emp_merge_groups


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
        from aiida_koopmans.workgraphs.kcp_files import kcp_hamiltonian_filename

        assert kcp_hamiltonian_filename(filled=True, spin_index=1) == "ham_occ_1.dat"
        assert kcp_hamiltonian_filename(filled=False, spin_index=1) == "ham_emp_1.dat"
        assert kcp_hamiltonian_filename(filled=True, spin_index=2) == "ham_occ_2.dat"
        assert kcp_hamiltonian_filename(filled=False, spin_index=2) == "ham_emp_2.dat"

    def test_a_third_spin_index_is_rejected(self):
        from aiida_koopmans.workgraphs.kcp_files import kcp_hamiltonian_filename

        with pytest.raises(ValueError, match="spin_index"):
            kcp_hamiltonian_filename(filled=True, spin_index=3)


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


class TestSmoothWannierizationGraph:
    """The denser-mesh wannierization: dense where it must be, coarse where it must not."""

    @staticmethod
    def _build(wannier_codes, silicon_structure):
        from aiida_koopmans.workgraphs.ui.smooth import SmoothWannierization

        coarse = orm.KpointsData()
        coarse.set_kpoints_mesh([2, 2, 2])
        dense = orm.KpointsData()
        dense.set_kpoints(np.zeros((8, 3)))
        return SmoothWannierization.build(
            codes=wannier_codes,
            structure=silicon_structure,
            blocks=[
                explicit_block("block_1", range(1, 5), filled=True),
                explicit_block("block_2", range(5, 9), filled=False),
            ],
            smooth_kpoints=dense,
            smooth_mp_grid=[4, 4, 4],
            scf_kpoints=coarse,
            pseudo_family="SSSP/1.3/PBE/efficiency",
        )

    def test_the_scf_keeps_the_coarse_mesh_while_wannier90_takes_the_dense_one(
        self, wannier_codes, silicon_structure
    ):
        """Only the nscf and the wannierization are denser; re-converging the density is not.

        The ``scf_kpoints`` assertion is the discriminating one: scaling
        both meshes runs a needlessly expensive scf and still produces
        plausible bands.
        """
        wg = self._build(wannier_codes, silicon_structure)
        wannierize = {task.name: task for task in wg.tasks}["wannierize"]
        assert wannierize.inputs["mp_grid"].value == [4, 4, 4]
        assert wannierize.inputs["scf_kpoints"].value.get_kpoints_mesh()[0] == [2, 2, 2]
        assert len(wannierize.inputs["kpoints"].value.get_kpoints()) == 8

    def test_the_blocks_namespace_is_the_only_output(self, wannier_codes, silicon_structure):
        """Downstream needs the per-block Hamiltonians and nothing else."""
        wg = self._build(wannier_codes, silicon_structure)
        assert [socket._name for socket in wg.outputs] == ["blocks"]


class TestSmoothInterpolationWiring:
    """A second, denser-mesh wannierization swaps in the smooth correction."""

    @staticmethod
    def _build(silicon_structure, *, smooth: bool, blocks_per_manifold: int = 1):
        from aiida_koopmans.workgraphs.ui.dscf import DscfBandStructureTask

        labels = {
            manifold: [
                manifold if blocks_per_manifold == 1 else f"{manifold}_{index}"
                for index in range(blocks_per_manifold)
            ]
            for manifold in ("occ", "emp")
        }
        merge_groups = [
            {"filled": True, "spin": "none", "blocks": [{"label": x} for x in labels["occ"]]},
            {"filled": False, "spin": "none", "blocks": [{"label": x} for x in labels["emp"]]},
        ]
        every_label = labels["occ"] + labels["emp"]
        inputs = {
            "structure": silicon_structure,
            "merge_groups": merge_groups,
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
        return DscfBandStructureTask.build(**inputs)

    def test_both_dft_hamiltonians_reach_the_interpolation(self, silicon_structure):
        wg = self._build(silicon_structure, smooth=True)
        interpolate = {task.name: task for task in wg.tasks}["interpolate_occ"]
        assert interpolate.inputs["dft_ham_file"]._links
        assert interpolate.inputs["dft_smooth_ham_file"]._links

    def test_without_a_smooth_run_neither_reaches_it(self, silicon_structure):
        """Negative control: the coarse Hamiltonian alone would be a silent bug.

        ``helpers.calc_bands`` subtracts whatever coarse Hamiltonian it is
        given, so wiring one without its dense counterpart shifts every
        band. Neither socket may be linked.
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

    def test_the_key_order_is_the_band_order(self, aiida_profile):
        """``b00`` before ``b01`` on the diagonal, whatever order they arrive in."""
        import io

        from aiida_koopmans.workgraphs.ui import helpers as ui_helpers
        from aiida_koopmans.workgraphs.ui.dscf import manifold_hamiltonian
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
        from aiida_workgraph import WorkGraph

        wg = self._build(silicon_structure, smooth=True)
        restored = WorkGraph.from_dict(wg.to_dict())
        assert sorted(task.name for task in restored.tasks) == sorted(
            task.name for task in wg.tasks
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
# The workflow-level gate
# ----------------------------------------------------------------------


class TestBandPathScope:
    """A ``kpath`` asks for the interpolation; only the Wannier route serves it."""

    def test_the_molecular_route_cannot_serve_a_path(self, ozone_structure):
        with pytest.raises(NotImplementedError, match="Wannier basis"):
            _validate_scope(
                correction=Correction.KI,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                fix_spin_contamination=False,
                structure=ozone_structure,
                kpath=_kpath(),
            )

    def test_the_molecular_route_without_a_path_is_untouched(self, ozone_structure):
        """Negative control: it is the path that the molecular route refuses."""
        _validate_scope(
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            fix_spin_contamination=False,
            structure=ozone_structure,
        )

    def test_the_wannier_route_takes_a_path(self, periodic_ozone_structure, kmesh):
        from tests.fixtures import ozone_projection_blocks

        _validate_scope(
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            fix_spin_contamination=False,
            structure=periodic_ozone_structure,
            blocks=ozone_projection_blocks(),
            kgrid=[2, 1, 1],
            kpoints=kmesh,
            kpath=_kpath(),
        )


class TestInterpolationKnobs:
    @staticmethod
    def _resolve(knobs, **smooth):
        """Resolve ``knobs`` for a run that asks for bands."""
        from aiida_koopmans.workgraphs.kcp import _resolve_band_interpolation_knobs

        return _resolve_band_interpolation_knobs(
            knobs,
            kpath=_kpath(),
            smooth_kpoints=smooth.get("smooth_kpoints"),
            smooth_mp_grid=smooth.get("smooth_mp_grid"),
        )

    @staticmethod
    def _dense_mesh():
        mesh = orm.KpointsData()
        mesh.set_kpoints_mesh([4, 4, 4])
        return mesh

    def test_a_scaled_factor_with_its_mesh_asks_for_smooth_interpolation(self):
        assert self._resolve(
            {"smooth_int_factor": [2, 2, 2]},
            smooth_kpoints=self._dense_mesh(),
            smooth_mp_grid=[4, 4, 4],
        ) == (True, True, True)

    def test_a_scaled_factor_without_its_mesh_names_what_is_missing(self):
        with pytest.raises(ValueError, match="smooth_kpoints and smooth_mp_grid"):
            self._resolve({"smooth_int_factor": [2, 2, 2]})

    def test_a_mesh_without_a_scaled_factor_is_refused(self):
        """A denser mesh nothing interpolates against would take no effect."""
        with pytest.raises(ValueError, match="smooth_int_factor"):
            self._resolve(
                {"smooth_int_factor": [1, 1, 1]},
                smooth_kpoints=self._dense_mesh(),
                smooth_mp_grid=[4, 4, 4],
            )

    def test_an_unscaled_factor_passes_and_the_knobs_come_through(self):
        assert self._resolve({"smooth_int_factor": [1, 1, 1], "do_dos": False}) == (
            True,
            False,
            False,
        )

    def test_the_defaults_are_ws_distance_and_a_dos(self):
        assert self._resolve(None) == (True, True, False)

    def test_knobs_without_a_path_are_refused(self):
        from aiida_koopmans.workgraphs.kcp import _resolve_band_interpolation_knobs

        with pytest.raises(ValueError, match="without a `kpath`"):
            _resolve_band_interpolation_knobs(
                {"do_dos": True}, kpath=None, smooth_kpoints=None, smooth_mp_grid=None
            )

    def test_no_knobs_and_no_path_is_silent(self):
        """Negative control: it is the settings, not the missing path, that raise."""
        from aiida_koopmans.workgraphs.kcp import _resolve_band_interpolation_knobs

        assert _resolve_band_interpolation_knobs(
            None, kpath=None, smooth_kpoints=None, smooth_mp_grid=None
        ) == (True, True, False)


class TestTheWorkflowGatesOnTheBandPath:
    """The whole ΔSCF workflow adds the stage exactly when it is given a path."""

    @staticmethod
    def _build(*, periodic_ozone_structure, kcp_code, mlwf_codes, ozone_pseudo_family, kmesh, **kw):
        from aiida_koopmans.workgraphs.kcp import KoopmansDSCFWorkflow
        from tests.fixtures import ozone_projection_blocks

        return KoopmansDSCFWorkflow.build(
            structure=periodic_ozone_structure,
            pseudo_family=ozone_pseudo_family,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.MLWFS,
            codes={**mlwf_codes, "kcp": kcp_code},
            blocks=ozone_projection_blocks(),
            kgrid=[2, 1, 1],
            kpoints=kmesh,
            **kw,
        )

    def test_a_path_adds_the_stage_and_prints_the_hamiltonians(
        self, periodic_ozone_structure, kcp_code, mlwf_codes, ozone_pseudo_family, kmesh
    ):
        wg = self._build(
            periodic_ozone_structure=periodic_ozone_structure,
            kcp_code=kcp_code,
            mlwf_codes=mlwf_codes,
            ozone_pseudo_family=ozone_pseudo_family,
            kmesh=kmesh,
            kpath=_kpath(),
        )
        names = [task.name for task in wg.tasks]
        assert "interpolate_band_structure" in names, names
        # The stage reads Hamiltonians that exist only because the final KI
        # was asked to print them.
        final_ki = next(task for task in wg.tasks if task.name.startswith("RunFinalKI"))
        assert final_ki.inputs["write_hr"].value is True

    def test_without_a_path_neither_happens(
        self, periodic_ozone_structure, kcp_code, mlwf_codes, ozone_pseudo_family, kmesh
    ):
        """Negative control: the same route, one input short, builds neither."""
        wg = self._build(
            periodic_ozone_structure=periodic_ozone_structure,
            kcp_code=kcp_code,
            mlwf_codes=mlwf_codes,
            ozone_pseudo_family=ozone_pseudo_family,
            kmesh=kmesh,
        )
        names = [task.name for task in wg.tasks]
        assert "interpolate_band_structure" not in names, names
        final_ki = next(task for task in wg.tasks if task.name.startswith("RunFinalKI"))
        assert final_ki.inputs["write_hr"].value is False


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

    def test_centres_that_are_not_three_vectors_are_named(self, aiida_profile, silicon_structure):
        """A per-band table of the wrong width cannot be a set of centres."""
        from aiida_koopmans.workgraphs.ui import interpolate_bands

        with pytest.raises(ValueError, match=r"one \[x, y, z\] per Wannier function"):
            interpolate_bands._callable(
                kc_ham_file=orm.SinglefileData.from_string("x"),
                centres=[[0.0, 0.0]],
                structure=silicon_structure,
                kpath=_kpath(),
                kgrid=[1, 1, 1],
            )


class TestExtractKoopmansHamiltonian:
    """Lifting one printed Hamiltonian out of the retrieved folder."""

    def test_a_missing_file_names_the_folder_contents(self, aiida_profile):
        """The run that did not print them is what the reader has to fix."""
        from aiida_koopmans.workgraphs.ui.dscf import extract_koopmans_hamiltonian

        retrieved = _retrieved_with_hamiltonians(["ham_occ_1.dat"])
        with pytest.raises(ValueError, match=r"ham_emp_1\.dat"):
            extract_koopmans_hamiltonian._callable(
                retrieved=retrieved, filename=orm.Str("ham_emp_1.dat")
            )

    def test_a_present_file_comes_out_under_its_own_name(self, aiida_profile):
        """Negative control: the same folder yields the file it does hold."""
        from aiida_koopmans.workgraphs.ui.dscf import extract_koopmans_hamiltonian

        retrieved = _retrieved_with_hamiltonians(["ham_occ_1.dat"])
        lifted = extract_koopmans_hamiltonian._callable(
            retrieved=retrieved, filename=orm.Str("ham_occ_1.dat")
        )
        assert lifted.filename == "ham_occ_1.dat"
        assert lifted.get_content() == "h"
