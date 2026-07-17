"""Tests for the trajectory (ML) workgraph builders in ``workgraphs/ml.py``.

Graph-construction tests mirror ``test_kcp_workgraph.py``: nothing is
executed against a real kcp.x — the fan-out topology is inspected at build
time.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.types import Correction, VariationalOrbitalType

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
        assert dataset["alphas"] == [0.6, 0.7, 0.5]
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
            init_orbitals=VariationalOrbitalType.KOHN_SHAM,
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
        assert {"descriptors", "alphas", "filled", "labels"} <= socket_names, socket_names
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
        from aiida_koopmans import ml_helpers

        model = ml_helpers.fit_screening_model(
            {
                "descriptors": [[-1.0], [-2.0]],
                "alphas": [0.6, 0.7],
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

    def test_orbital_density_descriptor_raises(
        self, ozone_structure, kcp_code, ozone_pseudo_family
    ):
        with pytest.raises(NotImplementedError, match="orbital_density"):
            self._build_wg(
                ozone_structure=ozone_structure,
                kcp_code=kcp_code,
                ozone_pseudo_family=ozone_pseudo_family,
                ml_mode="train",
                descriptor="orbital_density",
            )

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
