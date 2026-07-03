"""Machine-learning (trajectory) workflow: per-snapshot fan-out + model train/test.

Port of the legacy ``koopmans/workflows/_trajectory.py``
(``TrajectoryWorkflow``) plus the train/predict layer of ``koopmans/ml/``.
Each snapshot runs the full :func:`KoopmansDSCFWorkflow` (frozen public
interface, treated as a black box); the fan-out is the native for-loop over
a dynamic ``snapshots`` input namespace inside the ``@task.graph`` body (no
Map zone). Per-snapshot ``(descriptor, alpha)`` pairs are then gathered into
a single training/evaluation ``@task`` that consumes the dynamic namespace.

Scope notes (MVP):

* **Descriptor**: ``self_hartree`` only. The legacy default
  ``orbital_density`` (power-spectrum) descriptor needs the trial KI's
  real-space orbital-density files, which the ``KcpCalculation`` does not
  currently print/retrieve — the descriptor math is already ported in
  :mod:`aiida_koopmans.ml_helpers` (``compute_decomposition`` /
  ``compute_power_spectrum``) and can be wired once retrieval lands.
* **Modes**: ``train`` (fit a model on the computed alphas) and ``test``
  (compare a previously trained model's predictions against freshly
  computed alphas). Legacy ``predict`` mode (inject predicted alphas and
  skip the Delta-SCF refinement) is blocked on the frozen
  ``KoopmansDSCFWorkflow`` interface, which accepts only a scalar
  ``initial_alpha``.
* **Alphas**: the frozen ``KoopmansDSCFOutputs`` does not expose the
  converged screening parameters, so :func:`extract_final_alphas` recovers
  them from provenance — the final KI ``remote_folder``'s creator CalcJob
  received them as its ``alphas.filled`` / ``alphas.empty`` inputs.
* ``train_on_the_fly`` has no analogue here: snapshots run concurrently,
  so the model is fitted once on the gathered data (matching legacy
  ``train_on_the_fly=False`` behaviour).
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans import ml_helpers
from aiida_koopmans.types import Correction, VariationalOrbitalType
from aiida_koopmans.workgraphs.kcp import (
    KoopmansDSCFOutputs,
    KoopmansDSCFOverrides,
    KoopmansDSCFWorkflow,
)

ML_DESCRIPTOR_TYPES = ("self_hartree", "orbital_density")
ML_MODES = ("none", "train", "test")


class TrainOutputs(TypedDict):
    """Outputs of :func:`train_screening_model`.

    * ``model`` — the fitted, JSON-serialisable screening model (see
      :func:`aiida_koopmans.ml_helpers.fit_screening_model`).
    * ``metrics`` — training-set error metrics (a sanity indicator, not a
      validation score: the model is evaluated on its own training data).
    """

    model: dict
    metrics: dict


class TrajectoryOutputs(TypedDict):
    """Outputs of :func:`TrajectoryWorkflow`.

    * ``snapshots`` — dynamic namespace keyed by snapshot label; each entry
      is the full :class:`KoopmansDSCFOutputs` of that snapshot.
    * ``datasets`` — dynamic namespace keyed by snapshot label; each entry
      pairs per-orbital descriptors with computed alphas (empty when
      ``ml_mode == "none"``).
    * ``model`` — the trained model (``train``), the supplied model
      (``test``), or ``{}``.
    * ``evaluation`` — training-set metrics (``train``), test metrics plus
      per-orbital predictions (``test``), or ``{}``.
    """

    snapshots: Annotated[dict, dynamic(KoopmansDSCFOutputs)]
    datasets: Annotated[dict, dynamic(dict)]
    model: dict
    evaluation: dict


@task.calcfunction
def extract_final_alphas(remote_folder: orm.RemoteData) -> orm.Dict:
    """Recover the screening parameters the final KI ran with.

    The frozen :class:`KoopmansDSCFOutputs` interface does not expose the
    converged alphas, but its ``remote_folder`` was created by the final KI
    ``KcpCalculation``, which consumed them as ``alphas.filled`` /
    ``alphas.empty`` ``Dict`` inputs — walk one provenance step back and
    repackage them. A calcfunction (not a plain ``@task``) so the returned
    ``Dict`` is linked to the ``RemoteData`` in provenance.
    """
    creator = remote_folder.creator
    if creator is None:
        raise ValueError(
            f"RemoteData<{remote_folder.pk}> has no creator CalcJob; "
            "cannot recover the screening parameters"
        )
    if "alphas" not in creator.inputs:
        raise ValueError(
            f"{creator.process_label}<{creator.pk}> (creator of RemoteData<{remote_folder.pk}>) "
            "has no `alphas` input namespace; expected the final KI kcp.x calculation"
        )
    return orm.Dict(
        dict={
            "filled": creator.inputs.alphas.filled.get_dict(),
            "empty": creator.inputs.alphas.empty.get_dict(),
        }
    )


@task
def extract_snapshot_dataset(parameters: dict, alphas: dict) -> dict:
    """Pair one snapshot's self-Hartree descriptors with its screening parameters.

    ``parameters`` is the final KI's parsed output (its
    ``orbital_data["self-Hartree"]`` per-spin blocks list filled orbitals
    first, then empty — the same layout as the per-spin alpha lists).
    """
    orbital_data = parameters.get("orbital_data") or {}
    self_hartrees = orbital_data.get("self-Hartree") or []
    if not self_hartrees:
        raise ValueError(
            "No self-Hartree data found in the kcp.x output parameters; the final KI "
            "run did not print per-orbital data"
        )
    return ml_helpers.build_snapshot_dataset(self_hartrees, alphas)


@task
def train_screening_model(
    datasets: Annotated[dict, dynamic(dict)],
    estimator: str,
    occ_and_emp_together: bool,
    descriptor: str,
) -> TrainOutputs:
    """Gather every snapshot's dataset and fit the screening model.

    The single gather point of the workflow: consumes the dynamic
    per-snapshot namespace so the fit sees all ``(descriptor, alpha)``
    pairs at once (legacy trains after all snapshots when
    ``train_on_the_fly`` is off).
    """
    merged = ml_helpers.concatenate_datasets(datasets)
    model = ml_helpers.fit_screening_model(
        merged,
        estimator_type=estimator,
        occ_and_emp_together=occ_and_emp_together,
        descriptor=descriptor,
    )
    predicted = ml_helpers.predict_screening(model, merged)
    metrics = ml_helpers.evaluate_predictions(merged["alphas"], predicted)
    return TrainOutputs(model=model, metrics=metrics)


class EvaluateOutputs(TypedDict):
    """Outputs of :func:`evaluate_screening_model`.

    * ``evaluation`` — ``metrics`` (error metrics of predicted vs computed
      alphas) plus ``predictions`` (per-orbital ``labels`` / ``computed`` /
      ``predicted`` lists, labels being ``<snapshot>:<orbital>`` keys).
    * ``model`` — the supplied model, echoed so the graph can surface it as
      a socket (graph outputs must be task sockets, not raw input values).
    """

    evaluation: dict
    model: dict


@task
def evaluate_screening_model(
    datasets: Annotated[dict, dynamic(dict)],
    model: dict,
) -> EvaluateOutputs:
    """Gather every snapshot's dataset and score a trained model against it."""
    merged = ml_helpers.concatenate_datasets(datasets)
    predicted = ml_helpers.predict_screening(model, merged)
    evaluation = {
        "metrics": ml_helpers.evaluate_predictions(merged["alphas"], predicted),
        "predictions": {
            "labels": merged["labels"],
            "computed": merged["alphas"],
            "predicted": predicted,
        },
    }
    return EvaluateOutputs(evaluation=evaluation, model=model)


@task.graph
def TrajectoryWorkflow(
    code: orm.AbstractCode,
    snapshots: Annotated[dict, dynamic(orm.StructureData)],
    pseudo_family: str,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int = 2,
    tot_magnetization: int | None = None,
    correction: Correction = Correction.KI,
    init_orbitals: VariationalOrbitalType = VariationalOrbitalType.KOHN_SHAM,
    alpha_numsteps: int = 1,
    fix_spin_contamination: bool = False,
    initial_alpha: float = 0.6,
    spin_polarized: bool = False,
    orbital_groups_self_hartree_tol: float | None = None,
    overrides: KoopmansDSCFOverrides | None = None,
    options: dict[str, Any] | None = None,
    ml_mode: str = "none",
    ml_model: dict | None = None,
    estimator: str = "ridge_regression",
    descriptor: str = "self_hartree",
    occ_and_emp_together: bool = True,
) -> TrajectoryOutputs:
    """Run the Koopmans DSCF workflow on every snapshot, then train/test an ML model.

    ``snapshots`` is a dynamic namespace ``{label: StructureData}``; labels
    become link-label components, so they must match ``[A-Za-z0-9_]+``
    (e.g. ``snapshot_1``). Every snapshot fans out into an independent
    :func:`KoopmansDSCFWorkflow` (all DSCF inputs besides ``structure`` are
    shared), mirroring the legacy ``TrajectoryWorkflow`` which re-ran
    ``KoopmansDSCFWorkflow.fromparent`` with updated positions.

    ``ml_mode``:

    * ``"none"`` — just run the snapshots (legacy trajectory task without
      any ``ml`` flags).
    * ``"train"`` — additionally extract per-orbital ``(self-Hartree,
      alpha)`` pairs from every snapshot and fit a screening model; the
      fitted model is the ``model`` output.
    * ``"test"`` — extract the same pairs and score the supplied
      ``ml_model`` against the computed alphas (legacy ``ml.test``).
    """
    if ml_mode not in ML_MODES:
        raise ValueError(f"ml_mode must be one of {ML_MODES}, not `{ml_mode}`")
    if ml_mode != "none":
        if descriptor not in ML_DESCRIPTOR_TYPES:
            raise ValueError(f"`{descriptor}` is not implemented as a valid descriptor.")
        if descriptor != "self_hartree":
            raise NotImplementedError(
                "The `orbital_density` (power-spectrum) descriptor is not wired up yet: "
                "the kcp.x CalcJob does not retrieve the trial KI's real-space "
                "orbital-density files. The descriptor math is available in "
                "`aiida_koopmans.ml_helpers`. Use `descriptor='self_hartree'`."
            )
    if ml_mode == "test" and ml_model is None:
        raise ValueError("ml_mode='test' requires a trained `ml_model`")

    snapshot_outputs: dict[str, KoopmansDSCFOutputs] = {}
    datasets: dict[str, dict] = {}
    # Snapshot labels become socket/link-label components; node-graph
    # validates them upstream (letters, digits and underscores only).
    for label, structure in snapshots.items():
        dscf = KoopmansDSCFWorkflow(
            code=code,
            structure=structure,
            pseudo_family=pseudo_family,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            nbnd=nbnd,
            nspin=nspin,
            tot_magnetization=tot_magnetization,
            correction=correction,
            init_orbitals=init_orbitals,
            alpha_numsteps=alpha_numsteps,
            fix_spin_contamination=fix_spin_contamination,
            initial_alpha=initial_alpha,
            spin_polarized=spin_polarized,
            orbital_groups_self_hartree_tol=orbital_groups_self_hartree_tol,
            overrides=overrides,
            options=options,
            metadata={"call_link_label": f"dscf_{label}"},
        )
        snapshot_outputs[label] = KoopmansDSCFOutputs(
            parameters=dscf["parameters"],
            eigenvalues=dscf["eigenvalues"],
            lambdas=dscf["lambdas"],
            bare_lambdas=dscf["bare_lambdas"],
            remote_folder=dscf["remote_folder"],
        )

        if ml_mode != "none":
            alphas = extract_final_alphas(
                remote_folder=dscf["remote_folder"],
                metadata={"call_link_label": f"alphas_{label}"},
            )
            dataset = extract_snapshot_dataset(
                parameters=dscf["parameters"],
                alphas=alphas.result,
            )
            datasets[label] = dataset.result

    if ml_mode == "train":
        trained = train_screening_model(
            datasets=datasets,
            estimator=estimator,
            occ_and_emp_together=occ_and_emp_together,
            descriptor=descriptor,
        )
        model_output: dict = trained["model"]
        evaluation: dict = trained["metrics"]
    elif ml_mode == "test":
        evaluated = evaluate_screening_model(datasets=datasets, model=ml_model)
        model_output = evaluated["model"]
        evaluation = evaluated["evaluation"]
    else:
        model_output = {}
        evaluation = {}

    return TrajectoryOutputs(
        snapshots=snapshot_outputs,
        datasets=datasets,
        model=model_output,
        evaluation=evaluation,
    )
