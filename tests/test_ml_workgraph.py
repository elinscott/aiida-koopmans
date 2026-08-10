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


def _power_spectrum_model(**basis):
    """Fit a two-wide model stamped as ``power_spectrum`` on the given basis."""
    from aiida_koopmans.ml import resolve_radial_basis
    from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

    return ml_helpers.fit_screening_model(
        {
            "descriptors": [[1.0, 0.0], [0.0, 1.0]],
            "alpha_targets": [0.6, 0.7],
            "filled": [True, False],
            "labels": ["orb_1", "orb_2"],
        },
        "linear_regression",
        descriptor="power_spectrum",
        correction="ki",
        init_orbitals="mlwfs",
        radial_basis=resolve_radial_basis(
            {f"decompose_{key}": value for key, value in basis.items()}
        ),
    )


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
        ml_kwargs.setdefault("spin_polarized", False)
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
            **ml_kwargs,
        )

    def test_train_mode_fans_out_and_gathers(self, ozone_structure, kcp_code, ozone_pseudo_family):
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="train",
            descriptor=MLDescriptor.SELF_HARTREE,
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

    def test_a_non_ml_trajectory_submits_without_a_descriptor(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """An omitted ``descriptor`` must not block a non-ML submission.

        A build-time assertion cannot see this: the graph builds either
        way. ``check_before_run`` is what a submission runs first, and it
        is where a socket wrongly marked required surfaces.
        """
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="none",
        )
        assert wg.inputs["descriptor"]._metadata.required is False
        assert wg.inputs["descriptor"].value is None
        assert wg.check_before_run() is None

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
            descriptor=MLDescriptor.SELF_HARTREE,
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
            descriptor=MLDescriptor.SELF_HARTREE,
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

    def test_predict_mode_threads_the_descriptor_route_into_each_dscf(
        self, ozone_structure, kcp_code, ozone_pseudo_family, aiida_local_code_factory
    ):
        """Predict on ``power_spectrum`` builds, with the decompose inputs on each DSCF.

        The DSCF sub-graph body is deferred, so the discriminator at this
        level is that every snapshot's DSCF receives the descriptor, the
        code and the basis — the three the prediction site needs and which
        the trajectory previously kept to itself.
        """
        p2w = aiida_local_code_factory(
            executable="true", entry_point="koopmans.pw2wannier_decompose"
        )
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="predict",
            ml_model=_power_spectrum_model(),
            descriptor="power_spectrum",
            init_orbitals=VariationalOrbitalType.MLWFS,
            pw2wannier90_code=p2w,
            decompose_parameters={"decompose_n_max": 6, "decompose_l_max": 6},
        )
        dscf = next(t for t in wg.tasks if "dscf_snapshot_1" in t.name)
        assert dscf.inputs["descriptor"].value == MLDescriptor.POWER_SPECTRUM
        assert dscf.inputs["pw2wannier90_code"].value.uuid == p2w.uuid
        assert dict(dscf.inputs["decompose_parameters"].value)["decompose_n_max"] == 6

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

    def test_power_spectrum_on_a_spin_polarized_run_raises(
        self, ozone_structure, kcp_code, ozone_pseudo_family, aiida_local_code_factory
    ):
        """The descriptor is closed-shell only, and says so before any snapshot runs.

        The message states the gap and names the one descriptor that does
        work for spin-polarized runs; it must not tell the user to change
        ``spin``, which describes their system rather than a choice.
        """
        p2w = aiida_local_code_factory(
            executable="true", entry_point="koopmans.pw2wannier_decompose"
        )
        with pytest.raises(NotImplementedError, match="spin='collinear'") as excinfo:
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="train",
                descriptor="power_spectrum",
                init_orbitals=VariationalOrbitalType.MLWFS,
                pw2wannier90_code=p2w,
                spin_polarized=True,
            )
        assert "self_hartree" in str(excinfo.value)
        assert "spin='none'" not in str(excinfo.value)

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


class TestDescriptorIsNeverAssumed:
    """A run that never names a descriptor must not be given one.

    Both entry points leave ``descriptor`` unset by default. These two
    validators are where a working mode's missing descriptor is refused,
    and both are plain functions precisely so the path is testable
    without building a graph.
    """

    @pytest.mark.parametrize("ml_mode", ["train", "test", "predict"])
    def test_a_working_mode_without_a_descriptor_raises(self, ml_mode):
        from aiida_koopmans.workgraphs.ml import require_ml_mode_inputs

        with pytest.raises(ValueError, match="needs a `descriptor`"):
            require_ml_mode_inputs(
                ml_mode=ml_mode,
                descriptor=None,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                pw2wannier90_code=None,
                ml_model={"submodels": {}},
            )

    def test_mode_none_needs_no_descriptor(self):
        """Negative control: without ML there is nothing to describe."""
        from aiida_koopmans.workgraphs.ml import require_ml_mode_inputs

        assert (
            require_ml_mode_inputs(
                ml_mode="none",
                descriptor=None,
                init_orbitals=VariationalOrbitalType.KOHN_SHAM,
                pw2wannier90_code=None,
                ml_model=None,
            )
            is None
        )

    def test_a_dscf_carrying_a_model_without_a_descriptor_raises(self):
        from aiida_koopmans.workgraphs.kcp import _validate_ml_model_inputs

        with pytest.raises(ValueError, match="predicts from a descriptor"):
            _validate_ml_model_inputs(
                ml_model={"submodels": {}},
                ml_test=False,
                calculate_alpha=True,
                alpha_numsteps=1,
                descriptor=None,
            )

    def test_a_dscf_without_a_model_needs_no_descriptor(self):
        """Negative control: the descriptor only matters once a model reads it."""
        from aiida_koopmans.workgraphs.kcp import _validate_ml_model_inputs

        assert (
            _validate_ml_model_inputs(
                ml_model=None,
                ml_test=False,
                calculate_alpha=True,
                alpha_numsteps=1,
                descriptor=None,
            )
            is None
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
    """TEST mode runs both final KIs and gathers their deltas."""

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
        assert any("alpha_and_eigenvalue_deltas_snapshot_1" in n for n in names), names
        assert any("alpha_and_eigenvalue_deltas_snapshot_2" in n for n in names), names
        assert sum(1 for n in names if "evaluate_screening_model" in n) == 1, names

    def test_test_mode_graph_roundtrips(self, ozone_structure, kcp_code, ozone_pseudo_family):
        """The test-mode comparison graph survives the to_dict/from_dict round trip."""
        from tests.fixtures import assert_graph_roundtrips

        assert_graph_roundtrips(
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="test",
            )
        )

    def test_predict_mode_builds_no_delta_layer(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        """Predict applies the model; there are no computed alphas to compare against."""
        wg = self._build_wg(
            ozone_structure=ozone_structure,
            kcp_code=kcp_code,
            ozone_pseudo_family=ozone_pseudo_family,
            ml_mode="predict",
        )
        names = _all_task_names(wg)
        assert not any("compute_alpha_and_eigenvalue_deltas" in n for n in names), names


class TestEvaluateStampCheck:
    """``mode: test`` scores a model only when it describes the run's quantity.

    The estimators broadcast a wrong-width row instead of complaining, so
    a score against a mismatched model would come back as a number.
    """

    DATASETS = {  # noqa: RUF012
        "snapshot_1": {
            "descriptors": [[-1.0], [-2.0]],
            "alpha_targets": [0.6, 0.7],
            "filled": [True, False],
            "labels": ["orb_1", "orb_2"],
        }
    }

    def _call(self, model, **overrides):
        from aiida_koopmans.workgraphs.ml import evaluate_screening_model

        kwargs = {
            "descriptor": MLDescriptor.SELF_HARTREE,
            "correction": Correction.KI,
            "init_orbitals": VariationalOrbitalType.KOHN_SHAM,
        }
        kwargs.update(overrides)
        return evaluate_screening_model._callable(  # type: ignore[attr-defined]
            datasets=self.DATASETS, model=model, **kwargs
        )

    def test_a_matching_model_scores(self):
        from aiida_koopmans.workgraphs.ml import helpers as ml_helpers

        model = ml_helpers.fit_screening_model(
            self.DATASETS["snapshot_1"],
            "linear_regression",
            correction="ki",
            init_orbitals="kohn-sham",
        )
        outputs = self._call(model)
        assert outputs["evaluation"]["metrics"]["n_samples"] == 2

    def test_a_power_spectrum_model_scored_on_self_hartree_rows_raises(self):
        from aiida_koopmans.ml import ModelMismatchError

        with pytest.raises(ModelMismatchError, match="trained on `power_spectrum`"):
            self._call(_power_spectrum_model())

    def test_a_basis_mismatch_raises(self):
        from aiida_koopmans.ml import ModelMismatchError, resolve_radial_basis

        with pytest.raises(ModelMismatchError, match="radial basis"):
            self._call(
                _power_spectrum_model(r_min=1.0),
                descriptor=MLDescriptor.POWER_SPECTRUM,
                init_orbitals=VariationalOrbitalType.MLWFS,
                radial_basis=resolve_radial_basis(None),
            )
