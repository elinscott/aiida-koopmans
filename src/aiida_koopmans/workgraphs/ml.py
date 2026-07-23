"""Machine-learning (trajectory) workflow: per-snapshot fan-out + model train/test.

Each snapshot runs the full :func:`KoopmansDSCFWorkflow` (treated as a
black box); the fan-out is the native for-loop over a dynamic
``snapshots`` input namespace inside the ``@task.graph`` body. Per-snapshot
``(descriptor, alpha)`` pairs are then gathered into a single
training/evaluation ``@task`` that consumes the dynamic namespace.

Scope notes:

* **Descriptor**: ``self_hartree`` only. The ``orbital_density``
  (power-spectrum) descriptor needs the trial KI's real-space
  orbital-density files, which the ``KcpCalculation`` does not currently
  print/retrieve — the descriptor math lives in
  :mod:`aiida_koopmans.ml_helpers` (``compute_decomposition`` /
  ``compute_power_spectrum``) and can be wired once retrieval lands.
* **Modes**: ``train`` (fit a model on the computed alphas) and ``test``
  (compare a previously trained model's predictions against freshly
  computed alphas). ``predict`` mode (inject predicted alphas and skip the
  Delta-SCF refinement) is blocked on the ``KoopmansDSCFWorkflow``
  interface, which accepts only a scalar ``initial_alpha``.
* **Alphas**: read directly from ``KoopmansDSCFOutputs["alphas"]`` — the
  converged screening parameters the final KI consumed, exposed at the
  DSCF workflow level.
* Snapshots run concurrently, so the model is fitted once on the gathered
  data (no train-on-the-fly).
"""

from __future__ import annotations

import io
from typing import Annotated, Any, TypedDict, cast

from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans import ml_helpers
from aiida_koopmans.calculations.pw2wannier_decompose import Pw2wannierDecomposeCalculation
from aiida_koopmans.ml_helpers import SnapshotDataset
from aiida_koopmans.types import (
    AlphaScreening,
    Correction,
    ParallelizationDict,
    VariationalOrbitalType,
)
from aiida_koopmans.workgraphs import apply_parallelization
from aiida_koopmans.workgraphs.block_wannierize import WannierizeBlockOutputs
from aiida_koopmans.workgraphs.kcp import (
    KoopmansDSCFOutputs,
    KoopmansDSCFOverrides,
    KoopmansDSCFWorkflow,
)

# pw2wannier90.x ``wan_mode='decompose'`` wrapped as a workgraph task.
DecomposeTask = task(Pw2wannierDecomposeCalculation)

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
      is a :class:`~aiida_koopmans.ml_helpers.SnapshotDataset` namespace
      pairing per-orbital descriptors with computed alphas (empty when
      ``ml_mode == "none"``).
    * ``model`` — the trained model (``train``), the supplied model
      (``test``), or ``{}``.
    * ``evaluation`` — training-set metrics (``train``), test metrics plus
      per-orbital predictions (``test``), or ``{}``.
    """

    snapshots: Annotated[dict, dynamic(KoopmansDSCFOutputs)]
    datasets: Annotated[dict, dynamic(SnapshotDataset)]
    model: dict
    evaluation: dict


@task
def extract_snapshot_dataset(parameters: dict, alphas: AlphaScreening) -> SnapshotDataset:
    """Pair one snapshot's self-Hartree descriptors with its screening parameters.

    ``parameters`` is the final KI's parsed output (its
    ``orbital_data["self-Hartree"]`` per-spin blocks list filled orbitals
    first, then empty — the same layout as the per-spin alpha lists).

    The ``SnapshotDataset`` return fans out into one output socket per key
    (``descriptors`` / ``alphas`` / ``filled`` / ``labels``).
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
    datasets: Annotated[dict, dynamic(SnapshotDataset)],
    estimator: str,
    occ_and_emp_together: bool,
    descriptor: str,
) -> TrainOutputs:
    """Gather every snapshot's dataset and fit the screening model.

    The single gather point of the workflow: consumes the dynamic
    per-snapshot namespace so the fit sees all ``(descriptor, alpha)``
    pairs at once.
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
    datasets: Annotated[dict, dynamic(SnapshotDataset)],
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


# ----------------------------------------------------------------------
# Orbital-density descriptor via pw2wannier90 ``wan_mode='decompose'``
# ----------------------------------------------------------------------
#
# The ``orbital_density`` power-spectrum descriptor is built from a second
# pw2wannier90.x pass that decomposes each Wannier-function density (and the
# group density about each Wannier centre) onto a Gaussian x spherical-harmonic
# basis. The segment below turns a per-snapshot wannierization's retrieved
# folder plus the shared nscf scratch into per-orbital descriptors and a
# :class:`SnapshotDataset`. The alpha source stays route-generic (the dataset
# builder takes ``alphas`` as an input): kcp.x's converged alphas for the DSCF
# route today, kcw.x ``screen_parameters`` for the DFPT route later.


@task.calcfunction(outputs=["u_mat", "centres_xyz", "centres_file"])
def extract_decompose_inputs(hr_retrieved: orm.FolderData) -> dict:
    """Lift the wannier90 read-back files out of a block's retrieved folder.

    The per-block wannierization (with the ``wannier-product-retrieval``
    settings) forces ``aiida_u.mat`` and ``aiida_centres.xyz`` into the
    ``retrieved`` ``FolderData``. This calcfunction re-emits them as
    ``SinglefileData`` and, from the Wannier centres, synthesises the
    group-density ``centres_file`` (every Wannier centre) so the group
    density is decomposed about each orbital's own centre.
    """
    names = hr_retrieved.base.repository.list_object_names()
    for filename in ("aiida_u.mat", "aiida_centres.xyz"):
        if filename not in names:
            raise FileNotFoundError(
                f"``{filename}`` is missing from the wannier90 retrieved folder — check "
                "that the block wannierization forced its retrieval "
                "(write_u_matrices / write_xyz)."
            )

    with hr_retrieved.base.repository.open("aiida_u.mat", "rb") as handle:
        u_mat = orm.SinglefileData(handle, filename="aiida_u.mat")
    with hr_retrieved.base.repository.open("aiida_centres.xyz", "rb") as handle:
        centres_xyz = orm.SinglefileData(handle, filename="aiida_centres.xyz")

    xyz_content = hr_retrieved.base.repository.get_object_content("aiida_centres.xyz", mode="r")
    centres = ml_helpers.parse_wannier_centres_xyz(xyz_content)
    if not centres:
        raise ValueError(
            "No Wannier centres (``X`` rows) found in aiida_centres.xyz; cannot build "
            "the group-density centres file."
        )
    centres_content = ml_helpers.format_group_centres_file(centres)
    centres_file = orm.SinglefileData(
        io.BytesIO(centres_content.encode()), filename="gc_centres.dat"
    )

    return {"u_mat": u_mat, "centres_xyz": centres_xyz, "centres_file": centres_file}


class OrbitalDensityDatasetOutputs(TypedDict):
    """Outputs of :func:`OrbitalDensityDatasetWorkflow`.

    * ``dataset`` — the per-orbital :class:`SnapshotDataset` for one snapshot,
      its rows aligned with the snapshot's screening parameters across every
      projection block.
    """

    dataset: SnapshotDataset


@task
def compute_block_descriptors(
    coefficients: orm.ArrayData,
    group_coefficients: orm.ArrayData,
    output_parameters: dict,
) -> orm.ArrayData:
    """Cross-power descriptor matrix for one block's Wannier functions.

    Wraps :func:`ml_helpers.cross_power_spectra` on the block's decompose
    parser arrays; the ``(num_wann, descriptor_dim)`` result is stored under
    the ``descriptors`` array so the gather step can stack blocks by label.
    """
    n_max = int(output_parameters["n_max"])
    l_max = int(output_parameters["l_max"])
    coeff = coefficients.get_array("coefficients")
    group = group_coefficients.get_array("group_coefficients")
    power = ml_helpers.cross_power_spectra(coeff, group, n_max, l_max)
    out = orm.ArrayData()
    out.set_array("descriptors", power)
    return out


@task
def align_block_descriptors(
    block_descriptors: Annotated[dict, dynamic(orm.ArrayData)],
    merge_groups: list,
    alphas: dict,
) -> SnapshotDataset:
    """Gather the per-block descriptors and align them with the alphas.

    The single gather point of the orbital-density route: consumes the
    per-block descriptor namespace and returns a :class:`SnapshotDataset`
    whose row order matches the ``AlphaScreening`` convention (see
    :func:`ml_helpers.assemble_orbital_density_dataset`).
    """
    descriptors_by_label = {
        label: node.get_array("descriptors").tolist() for label, node in block_descriptors.items()
    }
    return ml_helpers.assemble_orbital_density_dataset(
        descriptors_by_label, merge_groups, cast("AlphaScreening", alphas)
    )


def require_wannier_route_inputs(
    nscf_remote_folder: Any,
    block_wannierizations: dict,
    merge_groups: list,
) -> None:
    """Guard the orbital_density route's Wannier-initialised-route requirement.

    The decompose descriptor route consumes the shared nscf scratch
    (``nscf_remote_folder``) and the per-block wannierizations
    (``block_wannierizations``) that :class:`KoopmansDSCFOutputs` carries **only**
    on the Wannier-initialised DSCF route; on the molecular (KS-init) route those
    keys are absent (see the KoopmansDSCFOutputs docstring). Raise a ValueError
    that names the requirement rather than letting a bare ``KeyError`` (or a
    ``None`` ``parent_folder`` downstream) surface. Kept as a plain function so
    the failure path is unit-testable without building the graph.
    """
    if nscf_remote_folder is None:
        raise ValueError(
            "The orbital_density descriptor route requires `nscf_remote_folder`, "
            "the shared nscf scratch that KoopmansDSCFOutputs exposes only on the "
            "Wannier-initialised DSCF route; it is absent on the molecular "
            "(KS-init) route. Use descriptor='self_hartree' for such snapshots."
        )
    missing = [
        block["label"]
        for group in merge_groups
        for block in group["blocks"]
        if block["label"] not in block_wannierizations
    ]
    if missing:
        raise ValueError(
            "The orbital_density descriptor route requires a per-block "
            f"wannierization for every merge-group block, but {missing} "
            "are absent from `block_wannierizations`. These are produced only by "
            "the Wannier-initialised DSCF route; the molecular (KS-init) route "
            "does not wannierize. Use descriptor='self_hartree'."
        )


@task.graph
def OrbitalDensityDatasetWorkflow(
    code: orm.AbstractCode,
    nscf_remote_folder: orm.RemoteData,
    block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)],
    merge_groups: list,
    alphas: dict,
    decompose_parameters: dict | None = None,
    parallelization: ParallelizationDict | None = None,
) -> OrbitalDensityDatasetOutputs:
    """Build one snapshot's orbital-density dataset from its Wannierisation.

    Fans a ``wan_mode='decompose'`` pw2wannier90.x pass out over every
    projection block (each block's ``hr_retrieved`` folder from
    ``block_wannierizations``, all against the shared ``nscf_remote_folder``),
    then gathers the per-block power-spectrum descriptors and aligns them with
    ``alphas`` in ``merge_groups`` order.

    ``merge_groups`` is the ``(filled, spin, blocks)`` partition (each block a
    ``{"label": ...}`` mapping); ``alphas`` is the snapshot's screening
    parameters in ``AlphaScreening`` shape.

    Raises ``ValueError`` at graph-build time if the Wannier-initialised-route
    inputs (``nscf_remote_folder`` / ``block_wannierizations``) are missing —
    i.e. this descriptor route was requested for a molecular (KS-init) snapshot.
    """
    require_wannier_route_inputs(nscf_remote_folder, block_wannierizations, merge_groups)
    block_descriptors: dict[str, orm.ArrayData] = {}
    for group in merge_groups:
        for block in group["blocks"]:
            label = block["label"]
            products = extract_decompose_inputs(block_wannierizations[label]["hr_retrieved"])
            decompose_inputs: dict[str, Any] = {
                "code": code,
                "parent_folder": nscf_remote_folder,
                "u_mat": products["u_mat"],
                "centres_xyz": products["centres_xyz"],
                "centres_file": products["centres_file"],
                "metadata": {"call_link_label": f"decompose_{label}"},
            }
            if decompose_parameters is not None:
                decompose_inputs["parameters"] = decompose_parameters
            apply_parallelization(decompose_inputs, parallelization, "pw2wannier90")
            decompose = DecomposeTask(**decompose_inputs)
            block_descriptors[label] = compute_block_descriptors(
                coefficients=decompose["coefficients"],
                group_coefficients=decompose["group_coefficients"],
                output_parameters=decompose["output_parameters"],
            ).result

    dataset = align_block_descriptors(
        block_descriptors=block_descriptors,
        merge_groups=merge_groups,
        alphas=alphas,
    )
    return OrbitalDensityDatasetOutputs(dataset=dataset)


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
    parallelization: ParallelizationDict | None = None,
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
    shared).

    ``ml_mode``:

    * ``"none"`` — just run the snapshots.
    * ``"train"`` — additionally extract per-orbital ``(self-Hartree,
      alpha)`` pairs from every snapshot and fit a screening model; the
      fitted model is the ``model`` output.
    * ``"test"`` — extract the same pairs and score the supplied
      ``ml_model`` against the computed alphas.
    """
    if ml_mode not in ML_MODES:
        raise ValueError(f"ml_mode must be one of {ML_MODES}, not `{ml_mode}`")
    if ml_mode != "none":
        if descriptor not in ML_DESCRIPTOR_TYPES:
            raise ValueError(f"`{descriptor}` is not implemented as a valid descriptor.")
        if descriptor != "self_hartree":
            raise NotImplementedError(
                "The `orbital_density` (power-spectrum) descriptor is implemented "
                "but gated pending live alignment validation. The full route is "
                "built and unit-tested — `OrbitalDensityDatasetWorkflow` fans a "
                "pw2wannier90 wan_mode='decompose' pass out over the per-block "
                "wannierizations now exposed on `KoopmansDSCFOutputs` "
                "(`nscf_remote_folder` / `block_wannierizations`), and "
                "`ml_helpers.assemble_orbital_density_dataset` aligns the per-block "
                "descriptors with the alphas. The decompose math is reproduced to "
                "machine precision, but the per-block Wannier-function-to-alpha "
                "ordering has not yet been confirmed by a live daemon regression "
                "against the legacy reference, so the guard stays until it is. Use "
                "`descriptor='self_hartree'` in the meantime."
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
            parallelization=parallelization,
            metadata={"call_link_label": f"dscf_{label}"},
        )
        snapshot_outputs[label] = KoopmansDSCFOutputs(
            parameters=dscf["parameters"],
            eigenvalues=dscf["eigenvalues"],
            lambdas=dscf["lambdas"],
            bare_lambdas=dscf["bare_lambdas"],
            remote_folder=dscf["remote_folder"],
            alphas=dscf["alphas"],
        )

        if ml_mode != "none":
            # The whole SnapshotDataset output namespace becomes the entry
            # (one socket per key), mirroring the channel-keyed DFPT wiring.
            datasets[label] = extract_snapshot_dataset(
                parameters=dscf["parameters"],
                alphas=dscf["alphas"],
            )

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
