"""Tests for the trajectory (ML) workgraph builders in ``workgraphs/ml.py``.

Graph-construction tests mirror ``test_kcp_workgraph.py``: nothing is
executed against a real kcp.x — the fan-out topology is inspected at build
time.
"""

from __future__ import annotations

from typing import TypedDict

import pytest
from aiida_workgraph import task

from aiida_koopmans.functionals import Correction
from aiida_koopmans.ml import MLDescriptor
from aiida_koopmans.variational_orbitals import VariationalOrbitalType


class PreRenameDataset(TypedDict):
    """The dataset shape as it was before the screening column was renamed."""

    descriptors: list
    alphas: list
    filled: list
    labels: list


@task
def pre_rename_dataset() -> PreRenameDataset:
    """Emit a dataset whose screening column reuses the namespace name."""
    return {"descriptors": [[1.0]], "alphas": [0.6], "filled": [True], "labels": ["orb_1"]}


# ----------------------------------------------------------------------
# extract_snapshot_dataset — plain-python callable
# ----------------------------------------------------------------------


class TestExtractSnapshotDataset:
    @staticmethod
    def _call(parameters, alphas):
        from aiida_koopmans.workgraphs.ml import extract_snapshot_dataset

        return extract_snapshot_dataset._callable(  # type: ignore[attr-defined]
            parameters=parameters, alphas=alphas
        )

    def test_pairs_self_hartree_with_alphas(self):
        parameters = {"orbital_data": {"self-Hartree": [[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]]}}
        alphas = {"filled": {"none": [0.6, 0.7]}, "empty": {"none": [0.5]}}
        dataset = self._call(parameters, alphas)
        assert dataset["descriptors"] == [[-1.0], [-2.0], [-3.0]]
        assert dataset["alpha_targets"] == [0.6, 0.7, 0.5]
        assert dataset["filled"] == [True, True, False]

    def test_missing_orbital_data_raises(self):
        with pytest.raises(ValueError, match="No self-Hartree data"):
            self._call({"energy": -1.0}, {"filled": {"none": [0.6]}, "empty": {}})


# ----------------------------------------------------------------------
# TrajectoryWorkflow graph build — structural inspection only
# ----------------------------------------------------------------------


def _all_task_names(wg) -> list[str]:
    """Walk every task (recursing into sub-graphs) and collect names."""
    names: list[str] = []

    def _walk(tasks):
        for t in tasks:
            names.append(t.name)
            children = getattr(t, "children", None)
            if children:
                _walk(children)

    _walk(wg.tasks)
    return names


class TestTrajectoryGraphBuild:
    def _build_wg(
        self, *, ozone_structure, kcp_code, ozone_pseudo_family, n_snapshots=2, **ml_kwargs
    ):
        from aiida_koopmans.workgraphs.ml import TrajectoryWorkflow

        snapshots = {f"snapshot_{i + 1}": ozone_structure for i in range(n_snapshots)}
        ml_kwargs.setdefault("init_orbitals", VariationalOrbitalType.KOHN_SHAM)
        return TrajectoryWorkflow.build(
            code=kcp_code,
            snapshots=snapshots,
            pseudo_family=ozone_pseudo_family,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            tot_magnetization=None,
            correction=Correction.KI,
            alpha_numsteps=1,
            fix_spin_contamination=False,
            initial_alpha=0.6,
            spin_polarized=False,
            **ml_kwargs,
        )

    def test_train_mode_fans_out_and_gathers(self, ozone_structure, kcp_code, ozone_pseudo_family):
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="train",
        )
        names = _all_task_names(wg)

        # One DSCF sub-graph per snapshot (call_link_label carries the key).
        assert any("dscf_snapshot_1" in n for n in names), names
        assert any("dscf_snapshot_2" in n for n in names), names
        # Per-snapshot dataset extraction (alphas come straight off the
        # DSCF outputs — no provenance-walk task anymore).
        assert sum(1 for n in names if "extract_snapshot_dataset" in n) == 2, names
        # The SnapshotDataset return fans out into one output socket per key.
        extract = next(t for t in wg.tasks if "extract_snapshot_dataset" in t.name)
        socket_names = {s._name for s in extract.outputs}
        assert {"descriptors", "alpha_targets", "filled", "labels"} <= socket_names, socket_names
        # Exactly one gather/fit task.
        assert sum(1 for n in names if "train_screening_model" in n) == 1, names
        assert not any("evaluate_screening_model" in n for n in names), names

    def test_none_mode_skips_ml_layer(self, ozone_structure, kcp_code, ozone_pseudo_family):
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="none",
        )
        names = _all_task_names(wg)
        assert any("dscf_snapshot_1" in n for n in names), names
        for forbidden in (
            "extract_final_alphas",
            "extract_snapshot_dataset",
            "train_screening_model",
            "evaluate_screening_model",
        ):
            assert not any(forbidden in n for n in names), (forbidden, names)

    def test_test_mode_wires_evaluation(self, ozone_structure, kcp_code, ozone_pseudo_family):
        from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

        model = ml_helpers.fit_screening_model(
            {
                "descriptors": [[-1.0], [-2.0]],
                "alpha_targets": [0.6, 0.7],
                "filled": [True, False],
                "labels": ["orb_1", "orb_2"],
            },
            "linear_regression",
        )
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="test",
            ml_model=model,
        )
        names = _all_task_names(wg)
        assert sum(1 for n in names if "evaluate_screening_model" in n) == 1, names
        assert not any("train_screening_model" in n for n in names), names

    def test_test_mode_without_model_raises(self, ozone_structure, kcp_code, ozone_pseudo_family):
        with pytest.raises(ValueError, match="requires a trained"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="test",
            )

    def test_predict_mode_skips_the_dataset_layer(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Predict applies the model inside each DSCF; no dataset, fit or score tasks."""
        from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

        model = ml_helpers.fit_screening_model(
            {
                "descriptors": [[-1.0], [-2.0]],
                "alpha_targets": [0.6, 0.7],
                "filled": [True, False],
                "labels": ["orb_1", "orb_2"],
            },
            "linear_regression",
        )
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="predict",
            ml_model=model,
        )
        names = _all_task_names(wg)
        assert any("dscf_snapshot_1" in n for n in names), names
        assert any("dscf_snapshot_2" in n for n in names), names
        for forbidden in (
            "extract_snapshot_dataset",
            "train_screening_model",
            "evaluate_screening_model",
        ):
            assert not any(forbidden in n for n in names), (forbidden, names)

    def test_predict_mode_without_model_raises(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        with pytest.raises(ValueError, match="requires a trained"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="predict",
            )

    def test_predict_mode_rejects_power_spectrum(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        with pytest.raises(NotImplementedError, match="self_hartree"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="predict",
                ml_model={"descriptor": "power_spectrum", "submodels": {}},
                descriptor="power_spectrum",
            )

    def test_power_spectrum_on_molecular_route_raises(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """The KS-init route wannierizes nothing, so it cannot feed the decompose pass."""
        with pytest.raises(ValueError, match="init_orbitals"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="train",
                descriptor="power_spectrum",
            )

    def test_power_spectrum_without_code_raises(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """The Wannier route still needs a decompose-capable pw2wannier90.x."""
        with pytest.raises(ValueError, match="pw2wannier90_code"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="train",
                descriptor="power_spectrum",
                init_orbitals=VariationalOrbitalType.MLWFS,
            )

    def test_power_spectrum_routes_to_decompose_segment(
        self, ozone_structure, kcp_code, ozone_pseudo_family, aiida_local_code_factory
    ):
        """`power_spectrum` swaps the self-Hartree extraction for the decompose segment.

        The discriminator against a guard flip that silently keeps the old
        descriptor: assert the per-snapshot dataset comes from
        ``PowerSpectrumDatasetWorkflow`` and that no
        ``extract_snapshot_dataset`` survives anywhere in the graph.
        """
        p2w = aiida_local_code_factory(
            executable="true", entry_point="koopmans.pw2wannier_decompose"
        )
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="train",
            descriptor="power_spectrum",
            init_orbitals=VariationalOrbitalType.MLWFS,
            pw2wannier90_code=p2w,
        )
        names = _all_task_names(wg)

        assert any("descriptors_snapshot_1" in n for n in names), names
        assert any("descriptors_snapshot_2" in n for n in names), names
        assert not any("extract_snapshot_dataset" in n for n in names), names
        assert sum(1 for n in names if "train_screening_model" in n) == 1, names

    def test_unknown_ml_mode_raises(self, ozone_structure, kcp_code, ozone_pseudo_family):
        with pytest.raises(ValueError, match="ml_mode"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="predict",
            )

    def test_bad_snapshot_label_raises(self, ozone_structure, kcp_code, ozone_pseudo_family):
        # Snapshot keys become socket / link-label components; node-graph
        # validates them at input construction, before the graph body runs.
        from aiida_koopmans.workgraphs.ml import TrajectoryWorkflow

        with pytest.raises(ValueError, match="letters, digits and underscores"):
            TrajectoryWorkflow.build(
                code=kcp_code,
                snapshots={"snapshot-1": ozone_structure},
                pseudo_family=ozone_pseudo_family,
                ecutwfc=65.0,
                ecutrho=260.0,
                nbnd=10,
            )


class TestSharedOutputSpecCollision:
    """Dataset columns must not be shadowed by the screening namespace ports.

    Every python task in one daemon worker validates its outputs against a
    single, process-wide port specification. Emitting a namespace output
    leaves a namespace port behind on it under that name, and a namespace
    port only accepts a mapping -- so any later task emitting a plain list
    under the same name is rejected. The screening layer emits ``alphas``
    and ``errors`` as namespaces, so the dataset's flat columns must not
    reuse either name.
    """

    SCREENING_NAMESPACES = ("alphas", "errors")

    @staticmethod
    def _shared_output_ports():
        from aiida_pythonjob.calculations.pyfunction import PyFunction

        return PyFunction.spec().outputs

    @staticmethod
    def _run_pre_rename(name):
        """Run the pre-rename dataset shape and return its process node."""
        from aiida_workgraph import WorkGraph

        wg = WorkGraph(name)
        wg.add_task(pre_rename_dataset, name="dataset")
        wg.run()
        children = [link.node for link in wg.process.base.links.get_outgoing().all()]
        return next(node for node in children if hasattr(node, "exception"))

    def test_dataset_runs_with_screening_namespace_ports_present(self, aiida_profile_clean):
        from aiida_workgraph import WorkGraph

        from aiida_koopmans.workgraphs.ml import extract_snapshot_dataset

        shared = self._shared_output_ports()
        for name in self.SCREENING_NAMESPACES:
            shared.get_port(name, create_dynamically=True)
        try:
            wg = WorkGraph("dataset_after_screening_namespaces")
            wg.add_task(
                extract_snapshot_dataset,
                name="extract",
                parameters={"orbital_data": {"self-Hartree": [[-1.0, -2.0, -3.0]]}},
                alphas={"filled": {"none": [0.6, 0.7]}, "empty": {"none": [0.5]}},
            )
            wg.run()
            children = [link.node for link in wg.process.base.links.get_outgoing().all()]
            extract = next(node for node in children if hasattr(node, "is_finished_ok"))
            assert extract.is_finished_ok, extract.exception
        finally:
            for name in self.SCREENING_NAMESPACES:
                shared.ports.pop(name, None)

    def test_the_old_column_name_is_rejected_in_that_state(self, aiida_profile_clean):
        """Positive control: the state injected above really is hostile.

        Without this, a passing sibling test could mean the injected ports
        are inert rather than that the column rename dodges them.
        """
        shared = self._shared_output_ports()
        for name in self.SCREENING_NAMESPACES:
            shared.get_port(name, create_dynamically=True)
        # Run the rejected case first and the accepted case second: a
        # successful run is a valid cache source, so the opposite order
        # would serve the second run from the cache and prove nothing.
        try:
            blocked = self._run_pre_rename("pre_rename_with_screening_namespaces")
            assert blocked.exception is not None, "pre-rename shape unexpectedly succeeded"
            assert "not sub class of `Mapping`" in blocked.exception, blocked.exception
        finally:
            for name in self.SCREENING_NAMESPACES:
                shared.ports.pop(name, None)

        allowed = self._run_pre_rename("pre_rename_without_screening_namespaces")
        assert allowed.is_finished_ok, allowed.exception

    def test_dataset_columns_avoid_the_screening_namespace_names(self):
        from aiida_koopmans.workgraphs.ml.helpers import SnapshotDataset

        columns = set(SnapshotDataset.__annotations__)
        assert not columns & set(self.SCREENING_NAMESPACES), columns


class TestTrainedModelArtifact:
    """The fitted model is a stored ``orm.Dict`` — the canonical artifact.

    A prediction run references it by PK/UUID (the koopmans ``ml.model``
    input), so the payload must live in the profile as one ``Dict`` node,
    stamps included, not as an exploded namespace or an unstored value.
    """

    def test_train_task_stores_the_model_as_one_dict_node(self, aiida_profile_clean):
        from aiida import orm
        from aiida_workgraph import WorkGraph

        from aiida_koopmans.workgraphs.ml import train_screening_model

        wg = WorkGraph("train_model_artifact")
        wg.add_task(
            train_screening_model,
            name="train",
            datasets={
                "snapshot_1": {
                    "descriptors": [[-1.0], [-2.0]],
                    "alpha_targets": [0.5, 0.6],
                    "filled": [True, False],
                    "labels": ["orb_1", "orb_2"],
                }
            },
            estimator="linear_regression",
            occ_and_emp_together=True,
            descriptor=MLDescriptor.SELF_HARTREE,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
        )
        wg.run()

        children = [link.node for link in wg.process.base.links.get_outgoing().all()]
        train = next(node for node in children if hasattr(node, "is_finished_ok"))
        assert train.is_finished_ok, train.exception

        model_node = train.outputs.model
        assert isinstance(model_node, orm.Dict), type(model_node)
        assert model_node.is_stored
        model = model_node.get_dict()
        assert model["descriptor"] == "self_hartree"
        assert model["correction"] == "ki"
        assert model["init_orbitals"] == "kohn-sham"
        assert set(model["submodels"]) == {"all"}


class TestTestModeTwin:
    """TEST mode runs the twin final KIs and gathers observable deltas."""

    def _build_wg(self, *, ozone_structure, kcp_code, ozone_pseudo_family, ml_mode, n=2):
        from aiida_koopmans.workgraphs.ml import TrajectoryWorkflow, helpers

        model = helpers.fit_screening_model(
            {
                "descriptors": [[-1.0], [-2.0]],
                "alpha_targets": [0.6, 0.7],
                "filled": [True, False],
                "labels": ["orb_1", "orb_2"],
            },
            "linear_regression",
            correction="ki",
            init_orbitals="kohn-sham",
        )
        return TrajectoryWorkflow.build(
            code=kcp_code,
            snapshots={f"snapshot_{i + 1}": ozone_structure for i in range(n)},
            pseudo_family=ozone_pseudo_family,
            ecutwfc=65.0,
            ecutrho=260.0,
            nbnd=10,
            nspin=2,
            tot_magnetization=None,
            correction=Correction.KI,
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
            alpha_numsteps=1,
            fix_spin_contamination=False,
            initial_alpha=0.6,
            spin_polarized=False,
            ml_mode=ml_mode,
            ml_model=model,
            descriptor=MLDescriptor.SELF_HARTREE,
        )

    def test_test_mode_gathers_per_snapshot_deltas(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """One delta task per snapshot feeds the evaluation gather."""
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="test",
        )
        names = _all_task_names(wg)
        assert any("final_ki_deltas_snapshot_1" in n for n in names), names
        assert any("final_ki_deltas_snapshot_2" in n for n in names), names
        assert sum(1 for n in names if "evaluate_screening_model" in n) == 1, names

    def test_predict_mode_builds_no_delta_layer(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Predict applies the model; there is no computed arm to compare against."""
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="predict",
        )
        names = _all_task_names(wg)
        assert not any("final_ki_deltas" in n for n in names), names
