"""Machine-learning (trajectory) workflow: per-snapshot fan-out + model train/test.

Each snapshot runs the full :func:`KoopmansDSCFWorkflow` (treated as a
black box); the fan-out is the native for-loop over a dynamic
``snapshots`` input namespace inside the ``@task.graph`` body. Per-snapshot
``(descriptor, alpha)`` pairs are then gathered into a single
training/evaluation ``@task`` that consumes the dynamic namespace.

Scope notes:

* **Descriptor**: ``self_hartree`` reads the per-orbital self-Hartree
  energies straight off the final KI's parsed output;
  ``power_spectrum`` instead runs
  :func:`PowerSpectrumDatasetWorkflow` per snapshot, which needs the
  Wannier-initialised DSCF route (``init_orbitals`` in ``mlwfs`` /
  ``projwfs``) and a pw2wannier90.x code carrying
  ``wan_mode='decompose'``.
* **Modes**: ``train`` (fit a model on the computed alphas) and ``test``
  (compare a previously trained model's predictions against freshly
  computed alphas). ``predict`` mode (inject predicted alphas and skip the
  Delta-SCF refinement) is not wired here yet — the
  ``KoopmansDSCFWorkflow`` side is ready (per-orbital ``initial_alphas``
  + ``calculate_alpha=False``), but this workflow does not build the
  per-snapshot predictions to feed it.
* **Alphas**: read directly from ``KoopmansDSCFOutputs["alphas"]`` — the
  converged screening parameters the final KI consumed, exposed at the
  DSCF workflow level.
* Snapshots run concurrently, so the model is fitted once on the gathered
  data (no train-on-the-fly).
"""

from __future__ import annotations

import io
from typing import Annotated, Any, TypedDict, cast

import numpy as np
from aiida import orm
from aiida_workgraph import dynamic, task

from aiida_koopmans import ml_helpers
from aiida_koopmans.calculations.pw2wannier_decompose import Pw2wannierDecomposeCalculation
from aiida_koopmans.ml_helpers import SnapshotDataset
from aiida_koopmans.types import (
    AlphaScreening,
    Correction,
    MLDescriptor,
    MLMode,
    ParallelizationDict,
    SpinChannel,
    VariationalOrbitalType,
)
from aiida_koopmans.workgraphs import (
    Codes,
    merge_parallelization_into_inputs,
    validate_parallelization,
)
from aiida_koopmans.workgraphs.block_wannierize import (
    WannierizeBlockOutputs,
    WannierizeOverrides,
)
from aiida_koopmans.workgraphs.kcp import (
    KoopmansDSCFOutputs,
    KoopmansDSCFOverrides,
    KoopmansDSCFWorkflow,
)

# pw2wannier90.x ``wan_mode='decompose'`` wrapped as a workgraph task.
DecomposeTask = task(Pw2wannierDecomposeCalculation)

#: Fallback ``metadata.options`` for the decompose CalcJob this module creates
#: directly. A CalcJob cannot run without ``resources``, and the descriptor
#: route must not depend on the caller supplying a parallelization block.
_DEFAULT_CALCJOB_OPTIONS: dict[str, Any] = {"resources": {"num_machines": 1}}

ML_DESCRIPTOR_TYPES = tuple(descriptor.value for descriptor in MLDescriptor)
ML_MODES = tuple(mode.value for mode in MLMode)


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
    (``descriptors`` / ``alpha_targets`` / ``filled`` / ``labels``).
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
    metrics = ml_helpers.evaluate_predictions(merged["alpha_targets"], predicted)
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
        "metrics": ml_helpers.evaluate_predictions(merged["alpha_targets"], predicted),
        "predictions": {
            "labels": merged["labels"],
            "computed": merged["alpha_targets"],
            "predicted": predicted,
        },
    }
    return EvaluateOutputs(evaluation=evaluation, model=model)


# ----------------------------------------------------------------------
# Power-spectrum descriptor via pw2wannier90 ``wan_mode='decompose'``
# ----------------------------------------------------------------------
#
# The ``power_spectrum`` descriptor is built from a second
# pw2wannier90.x pass that decomposes each Wannier-function density (and the
# group density about each Wannier centre) onto a Gaussian x spherical-harmonic
# basis. The segment below turns a per-snapshot wannierization's retrieved
# folder plus the shared nscf scratch into per-orbital descriptors and a
# :class:`SnapshotDataset`. The alpha source stays route-generic (the dataset
# builder takes ``alphas`` as an input): kcp.x's converged alphas for the DSCF
# route today, kcw.x ``screen_parameters`` for the DFPT route later.


@task.calcfunction(outputs=["u_mat", "centres_xyz", "centres_file"])
def extract_decompose_inputs(retrieved: orm.FolderData) -> dict:
    """Lift the wannier90 read-back files out of a block's retrieved folder.

    The per-block wannierization (with the ``wannier-product-retrieval``
    settings) forces ``aiida_u.mat`` and ``aiida_centres.xyz`` into the
    ``retrieved`` ``FolderData``. This calcfunction re-emits them as
    ``SinglefileData`` and, from the Wannier centres, synthesises the
    group-density ``centres_file`` (every Wannier centre) so the group
    density is decomposed about each orbital's own centre.
    """
    names = retrieved.base.repository.list_object_names()
    for filename in ("aiida_u.mat", "aiida_centres.xyz"):
        if filename not in names:
            raise FileNotFoundError(
                f"``{filename}`` is missing from the wannier90 retrieved folder — check "
                "that the block wannierization forced its retrieval "
                "(write_u_matrices / write_xyz)."
            )

    with retrieved.base.repository.open("aiida_u.mat", "rb") as handle:
        u_mat = orm.SinglefileData(handle, filename="aiida_u.mat")
    with retrieved.base.repository.open("aiida_centres.xyz", "rb") as handle:
        centres_xyz = orm.SinglefileData(handle, filename="aiida_centres.xyz")

    xyz_content = retrieved.base.repository.get_object_content("aiida_centres.xyz", mode="r")
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


@task.calcfunction
def extract_u_dis_mat(retrieved: orm.FolderData) -> orm.SinglefileData:
    """Lift the wannier90 disentanglement matrix out of a block's retrieved folder.

    Called only for a manifold the caller's block metadata marks as
    disentangling (``num_bands`` > ``num_wann``); the decompose pass errors
    without ``<seed>_u_dis.mat`` in that case, so a missing file here is a
    hard error rather than an optional-input skip.
    """
    names = retrieved.base.repository.list_object_names()
    if "aiida_u_dis.mat" not in names:
        raise FileNotFoundError(
            "``aiida_u_dis.mat`` is missing from the wannier90 retrieved folder, but "
            "the block metadata marks the manifold as disentangling "
            "(num_bands > num_wann). Check that the block wannierization forced its "
            "retrieval (write_u_matrices)."
        )
    with retrieved.base.repository.open("aiida_u_dis.mat", "rb") as handle:
        return orm.SinglefileData(handle, filename="aiida_u_dis.mat")


def _block_disentangles(block: dict) -> bool:
    """Return whether a manifold disentangles, from the block's own counts.

    Structural authority: the decision to stage ``u_dis`` is read from the
    caller's ``num_bands`` / ``num_wann`` metadata, never probed from the
    retrieved folder's contents.
    """
    num_bands = block.get("num_bands")
    num_wann = block.get("num_wann")
    return num_bands is not None and num_wann is not None and num_bands > num_wann


def _spin_component(group_spin: Any) -> str | None:
    """Map a manifold's spin channel to the decompose ``spin_component`` value.

    ``SpinChannel.UP`` / ``SpinChannel.DOWN`` become ``'up'`` / ``'down'`` so
    the decompose pass reads one channel of an nspin=2 scratch; ``NONE``
    (nspin=1) returns ``None`` and the key is omitted, letting QE default to
    the single channel.
    """
    spin = group_spin if isinstance(group_spin, SpinChannel) else SpinChannel(group_spin)
    return None if spin == SpinChannel.NONE else spin.value


class PowerSpectrumDatasetOutputs(TypedDict):
    """Outputs of :func:`PowerSpectrumDatasetWorkflow`.

    * ``dataset`` — the per-orbital :class:`SnapshotDataset` for one snapshot,
      its rows aligned with the snapshot's screening parameters across every
      projection block.
    """

    dataset: SnapshotDataset


def array_payload(value: Any, name: str) -> np.ndarray:
    """Return the ``name`` array of *value*, node or already-deserialized.

    A single-array ``orm.ArrayData`` socket reaches a task body as a bare
    ``numpy`` array, because ``aiida-pythonjob`` deserializes it on the way
    in; the node itself arrives when the callable is invoked directly.
    """
    if hasattr(value, "get_array"):
        return value.get_array(name)
    return np.asarray(value)


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
    coeff = array_payload(coefficients, "coefficients")
    group = array_payload(group_coefficients, "group_coefficients")
    power = ml_helpers.cross_power_spectra(coeff, group, n_max, l_max)
    out = orm.ArrayData()
    out.set_array("descriptors", power)
    return out


@task
def align_block_descriptors(
    block_descriptors: Annotated[dict, dynamic(orm.ArrayData)],
    merge_groups: list,
    alphas: AlphaScreening,
) -> SnapshotDataset:
    """Gather the per-block descriptors and align them with the alphas.

    The single gather point of the orbital-density route: consumes the
    per-block descriptor namespace and returns a :class:`SnapshotDataset`
    whose row order matches the ``AlphaScreening`` convention (see
    :func:`ml_helpers.assemble_power_spectrum_dataset`).
    """
    descriptors_by_label = {
        label: array_payload(node, "descriptors").tolist()
        for label, node in block_descriptors.items()
    }
    return ml_helpers.assemble_power_spectrum_dataset(
        descriptors_by_label, merge_groups, cast("AlphaScreening", alphas)
    )


def require_wannier_route_inputs(
    nscf_remote_folder: Any,
    block_wannierizations: dict,
    merge_groups: list,
) -> None:
    """Guard the power_spectrum route's Wannier-initialised-route requirement.

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
            "The power_spectrum descriptor route requires `nscf_remote_folder`, "
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
            "The power_spectrum descriptor route requires a per-block "
            f"wannierization for every merge-group block, but {missing} "
            "are absent from `block_wannierizations`. These are produced only by "
            "the Wannier-initialised DSCF route; the molecular (KS-init) route "
            "does not wannierize. Use descriptor='self_hartree'."
        )


@task.graph
def PowerSpectrumDatasetWorkflow(
    code: orm.AbstractCode,
    nscf_remote_folder: orm.RemoteData,
    block_wannierizations: Annotated[dict, dynamic(WannierizeBlockOutputs)],
    merge_groups: list,
    alphas: AlphaScreening,
    decompose_parameters: dict | None = None,
    parallelization: ParallelizationDict | None = None,
) -> PowerSpectrumDatasetOutputs:
    """Build one snapshot's orbital-density dataset from its Wannierization.

    Fans a ``wan_mode='decompose'`` pw2wannier90.x pass out over every
    projection block (each block's ``retrieved`` folder and ``nnkp_file``
    from ``block_wannierizations``, all against the shared
    ``nscf_remote_folder``), then gathers the per-block power-spectrum
    descriptors and aligns them with ``alphas`` in ``merge_groups`` order.

    Each block's decompose pass is staged from that block's own Wannierization:
    the required ``nnkp`` file threads straight from ``nnkp_file``; the U_dis
    matrix is lifted from ``retrieved`` and wired only when the block's
    metadata marks the manifold as disentangling (``num_bands`` > ``num_wann``);
    and the ``spin_component`` namelist key is set per group from the manifold's
    spin channel so an nspin=2 scratch is read one channel at a time.

    ``merge_groups`` is the ``(filled, spin, blocks)`` partition (each block a
    ``{"label", "num_wann", "num_bands", ...}`` mapping); ``alphas`` is the
    snapshot's screening parameters in ``AlphaScreening`` shape.

    Raises ``ValueError`` at graph-build time if the Wannier-initialised-route
    inputs (``nscf_remote_folder`` / ``block_wannierizations``) are missing —
    i.e. this descriptor route was requested for a molecular (KS-init) snapshot.
    """
    require_wannier_route_inputs(nscf_remote_folder, block_wannierizations, merge_groups)
    block_descriptors: dict[str, orm.ArrayData] = {}
    for group in merge_groups:
        spin_component = _spin_component(group["spin"])
        for block in group["blocks"]:
            label = block["label"]
            products = extract_decompose_inputs(block_wannierizations[label]["retrieved"])
            decompose_inputs: dict[str, Any] = {
                "code": code,
                "parent_folder": nscf_remote_folder,
                "nnkp": block_wannierizations[label]["nnkp_file"],
                "u_mat": products["u_mat"],
                "centres_xyz": products["centres_xyz"],
                "centres_file": products["centres_file"],
                "metadata": {"call_link_label": f"decompose_{label}"},
            }
            # Per-block namelist: the manifold's spin channel (structural
            # authority) fixes ``spin_component``, overriding any shared value
            # because one ``decompose_parameters`` dict spans both channels.
            block_parameters = dict(decompose_parameters or {})
            if spin_component is not None:
                block_parameters["spin_component"] = spin_component
            if block_parameters:
                decompose_inputs["parameters"] = block_parameters
            # A disentangling manifold needs its U_dis matrix or the QE
            # decompose pass errors; the decision is read from the block's
            # counts, never probed from the retrieved folder.
            if _block_disentangles(block):
                decompose_inputs["u_dis_mat"] = extract_u_dis_mat(
                    block_wannierizations[label]["retrieved"]
                ).result
            # Seed the resources first so the pass is runnable with no
            # parallelization block; a supplied one overwrites them.
            decompose_inputs["metadata"]["options"] = dict(_DEFAULT_CALCJOB_OPTIONS)
            merge_parallelization_into_inputs(decompose_inputs, parallelization, "pw2wannier90")
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
    return PowerSpectrumDatasetOutputs(dataset=dataset)


def require_power_spectrum_route(
    init_orbitals: Any,
    pw2wannier90_code: Any,
) -> None:
    """Guard the trajectory-level requirements of the power_spectrum descriptor.

    The descriptor is built from a pw2wannier90.x ``wan_mode='decompose'``
    pass over each snapshot's per-block Wannierizations, so it needs both a
    Wannier-initialised DSCF route to produce them and a code to run the
    pass with. Raise here — at graph build, before any snapshot is
    launched — rather than letting the requirement surface per snapshot
    once the fan-out is already running. Kept as a plain function so both
    failure paths are unit-testable without building the graph.
    """
    orbitals = VariationalOrbitalType(init_orbitals)
    if orbitals not in (VariationalOrbitalType.MLWFS, VariationalOrbitalType.PROJWFS):
        raise ValueError(
            f"descriptor='power_spectrum' requires the Wannier-initialised DSCF "
            f"route (init_orbitals='mlwfs' or 'projwfs'), but init_orbitals="
            f"{orbitals.value!r}. That route is what produces the per-block "
            f"Wannierizations the decompose pass decomposes; the molecular "
            f"(Kohn-Sham) route wannierizes nothing. Use "
            f"descriptor='self_hartree' for such snapshots."
        )
    if pw2wannier90_code is None:
        raise ValueError(
            "descriptor='power_spectrum' requires `pw2wannier90_code`, a "
            "pw2wannier90.x code built with wan_mode='decompose'. "
            "Use descriptor='self_hartree' if no such code is available."
        )


def require_ml_mode_inputs(
    *,
    ml_mode: str,
    descriptor: str,
    init_orbitals: Any,
    pw2wannier90_code: Any,
    ml_model: dict | None,
) -> None:
    """Guard the ``ml_mode`` / ``descriptor`` / ``ml_model`` combinations.

    ``test`` and ``predict`` need a trained ``ml_model``; ``predict``
    supports only ``self_hartree`` (the decompose pass that builds the
    power-spectrum descriptors is not wired into the DSCF's screening
    stage, where the prediction runs); a ``power_spectrum`` run must
    satisfy :func:`require_power_spectrum_route`.
    """
    if ml_mode not in ML_MODES:
        raise ValueError(f"ml_mode must be one of {ML_MODES}, not `{ml_mode}`")
    if ml_mode != MLMode.NONE:
        if descriptor not in ML_DESCRIPTOR_TYPES:
            raise ValueError(f"`{descriptor}` is not implemented as a valid descriptor.")
        if ml_mode == MLMode.PREDICT and descriptor != MLDescriptor.SELF_HARTREE:
            raise NotImplementedError(
                f"ml_mode='predict' supports only the 'self_hartree' descriptor, not "
                f"`{descriptor}`: the decompose pass that builds the power-spectrum "
                "descriptors is not wired into the DSCF's screening stage, where the "
                "prediction runs. Use descriptor='self_hartree'."
            )
        if descriptor == MLDescriptor.POWER_SPECTRUM:
            require_power_spectrum_route(init_orbitals, pw2wannier90_code)
    if ml_mode in (MLMode.TEST, MLMode.PREDICT) and ml_model is None:
        raise ValueError(f"ml_mode='{ml_mode}' requires a trained `ml_model`")


def build_snapshot_dataset(
    descriptor: str,
    dscf: Any,
    *,
    label: str,
    pw2wannier90_code: orm.AbstractCode | None,
    decompose_parameters: dict | None,
    parallelization: ParallelizationDict | None,
) -> Any:
    """Wire one snapshot's ``(descriptor, alpha)`` dataset for the chosen descriptor.

    ``self_hartree`` reads the pairs off the final KI's parsed output;
    ``power_spectrum`` runs the decompose segment over the snapshot's
    per-block Wannierizations. Both return a :class:`SnapshotDataset`
    namespace whose rows follow the snapshot's alpha order.
    """
    if descriptor == MLDescriptor.POWER_SPECTRUM:
        return PowerSpectrumDatasetWorkflow(
            code=pw2wannier90_code,
            nscf_remote_folder=dscf["nscf_remote_folder"],
            block_wannierizations=dscf["block_wannierizations"],
            merge_groups=dscf["merge_groups"],
            alphas=dscf["alphas"],
            decompose_parameters=decompose_parameters,
            parallelization=parallelization,
            metadata={"call_link_label": f"descriptors_{label}"},
        )["dataset"]
    return extract_snapshot_dataset(parameters=dscf["parameters"], alphas=dscf["alphas"])


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
    codes: Codes | None = None,
    blocks: list | None = None,
    kgrid: list[int] | None = None,
    kpoints: orm.KpointsData | None = None,
    gamma_only: bool = False,
    wannier_protocol: str | None = None,
    wannier_overrides: WannierizeOverrides | None = None,
    mp_correction: bool | None = None,
    eps_inf: float | None = None,
    overrides: KoopmansDSCFOverrides | None = None,
    parallelization: ParallelizationDict | None = None,
    ml_mode: str = "none",
    ml_model: dict | None = None,
    estimator: str = "ridge_regression",
    descriptor: str = "self_hartree",
    occ_and_emp_together: bool = True,
    pw2wannier90_code: orm.AbstractCode | None = None,
    decompose_parameters: dict | None = None,
) -> TrajectoryOutputs:
    """Run the Koopmans DSCF workflow on every snapshot, then train/test an ML model.

    ``snapshots`` is a dynamic namespace ``{label: StructureData}``; labels
    become link-label components, so they must match ``[A-Za-z0-9_]+``
    (e.g. ``snapshot_1``). Every snapshot fans out into an independent
    :func:`KoopmansDSCFWorkflow` (all DSCF inputs besides ``structure`` are
    shared).

    ``ml_mode``:

    * ``"none"`` — just run the snapshots.
    * ``"train"`` — additionally extract per-orbital ``(descriptor,
      alpha)`` pairs from every snapshot and fit a screening model; the
      fitted model is the ``model`` output.
    * ``"test"`` — extract the same pairs and score the supplied
      ``ml_model`` against the computed alphas.
    * ``"predict"`` — inject ``ml_model`` into every snapshot's DSCF:
      the per-orbital Delta-SCF refinement is replaced by a model
      prediction from the trial KI's self-Hartree descriptors (so only
      ``descriptor='self_hartree'`` is supported), and the final KI
      applies the predicted alphas.

    ``descriptor`` selects what those pairs are built from.
    ``'self_hartree'`` reads the per-orbital self-Hartree energies off the
    final KI's parsed output. ``'power_spectrum'`` instead runs
    :func:`PowerSpectrumDatasetWorkflow` per snapshot, decomposing each
    block's Wannier functions with a pw2wannier90.x
    ``wan_mode='decompose'`` pass; it therefore needs ``pw2wannier90_code``
    (a decompose-capable build) and the Wannier-initialised DSCF route, and
    accepts the basis settings (``n_max`` / ``l_max`` / ``r_min`` /
    ``r_max``) through ``decompose_parameters``. Both descriptors return
    rows in the same per-orbital order, the one the snapshot's ``alphas``
    are reported in.
    """
    validate_parallelization(parallelization)

    require_ml_mode_inputs(
        ml_mode=ml_mode,
        descriptor=descriptor,
        init_orbitals=init_orbitals,
        pw2wannier90_code=pw2wannier90_code,
        ml_model=ml_model,
    )

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
            ml_model=ml_model if ml_mode == MLMode.PREDICT else None,
            spin_polarized=spin_polarized,
            orbital_groups_self_hartree_tol=orbital_groups_self_hartree_tol,
            codes=codes,
            blocks=blocks,
            kgrid=kgrid,
            kpoints=kpoints,
            gamma_only=gamma_only,
            wannier_protocol=wannier_protocol,
            wannier_overrides=wannier_overrides,
            mp_correction=mp_correction,
            eps_inf=eps_inf,
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

        if ml_mode in (MLMode.TRAIN, MLMode.TEST):
            # The whole SnapshotDataset output namespace becomes the entry
            # (one socket per key), mirroring the channel-keyed DFPT wiring.
            # Predict mode builds no dataset: the model is *applied* inside
            # each snapshot's DSCF, and there are no computed alphas to
            # pair descriptors with.
            datasets[label] = build_snapshot_dataset(
                descriptor,
                dscf,
                label=label,
                pw2wannier90_code=pw2wannier90_code,
                decompose_parameters=decompose_parameters,
                parallelization=parallelization,
            )

    if ml_mode == MLMode.TRAIN:
        trained = train_screening_model(
            datasets=datasets,
            estimator=estimator,
            occ_and_emp_together=occ_and_emp_together,
            descriptor=descriptor,
        )
        model_output: dict = trained["model"]
        evaluation: dict = trained["metrics"]
    elif ml_mode == MLMode.TEST:
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
