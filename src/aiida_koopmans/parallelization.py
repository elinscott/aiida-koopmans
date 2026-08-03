"""Per-code parallelization: the code vocabulary and the option/cmdline merging.

A workflow's ``parallelization`` input is one mapping keyed by QE code
name (:data:`ParallelizationDict`); the helpers here turn each entry into
``metadata.options`` / ``settings.cmdline`` on the CalcJob steps a graph
builds.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, TypedDict, get_args

# The QE code vocabulary, defined once. ``CODE_NAMES`` is the runtime tuple
# (the koopmans2 parallelization schema imports it for its ``ALL_CODES``); the
# ``CodeName`` ``Literal`` types dict keys / helper args so a typo is a static
# error, and ``validate_parallelization`` catches one that slips in at runtime.
CodeName = Literal["pw", "kcp", "kcw", "ph", "projwfc", "pw2wannier90", "wann2kcp", "wannier90"]
CODE_NAMES: tuple[str, ...] = get_args(CodeName)


class CodeParallelization(TypedDict, total=False):
    """One code's parallelization directive: MPI ranks, k-point pools, pencil decomp, threads.

    ``ntasks`` sets ``metadata.options.resources`` (``num_mpiprocs_per_machine``);
    ``npool`` becomes ``-npool`` and ``pd`` becomes ``-pd true`` on the QE
    command line; ``omp`` sets the per-rank OpenMP/BLAS thread count via a
    ``metadata.options.prepend_text`` export block (overriding the
    computer-level pin of one thread). Every field is optional
    (``total=False``); an absent one means the QE/AiiDA default. Mirrors the
    koopmans2 ``CodeParallelization`` pydantic model that produces these dicts.
    """

    ntasks: int
    npool: int
    pd: bool
    omp: int


# Per-code parallelization mapping threaded into every top-level graph: a plain
# dict keyed by code name, each value a :class:`CodeParallelization`. A dict
# alias (not a fixed-key TypedDict) so ``aiida-workgraph`` keeps it as one
# opaque input socket rather than expanding a typed namespace, and so a dynamic
# ``code`` lookup types cleanly. Which flags each code accepts is enforced by
# ``POOL_SUPPORTING_CODES`` / ``PD_SUPPORTING_CODES`` below.
ParallelizationDict = dict[CodeName, CodeParallelization]


# QE codes that accept ``-npool`` (k-point pools) and ``-pd`` (pencil
# decomposition) on the command line. Source-verified against Quantum ESPRESSO:
# ``Modules/command_line_options.f90`` parses ``-nk``/``-npool`` and ``-pd``
# globally for the modern binaries (pw, ph, projwfc, pw2wannier90, kcw); the
# koopmans-kcp fork (kcp.x and wann2kcp.x) predates that parser and reads no CLI
# flags at all, and wannier90 has no pool/pd concept. ``kcw`` accepts pools only
# for its wann2kc / screen steps, not ham (``KCW/src/kcw_readin.f90`` rejects
# pools for calculation='ham') — that per-step split is the ``pools`` argument
# below, not a code-level fact.
POOL_SUPPORTING_CODES = frozenset({"pw", "ph", "projwfc", "pw2wannier90", "kcw"})
PD_SUPPORTING_CODES = frozenset({"pw", "ph", "projwfc", "pw2wannier90", "kcw"})

# Exported via metadata.options.prepend_text, which aiida-core assembles last
# so it overrides the computer-level pin of 1 (an ``environment_variables``
# entry would instead lose — those are emitted before any prepend_text).
_OMP_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def omp_prepend_text(nthreads: int) -> str:
    """Return the ``export OMP_NUM_THREADS=...`` block pinning all BLAS thread pools."""
    return "\n".join(f"export {var}={int(nthreads)}" for var in _OMP_THREAD_VARS)


def validate_parallelization(parallelization: ParallelizationDict | None) -> None:
    """Raise if the mapping names a code outside the known vocabulary.

    The runtime guard behind the ``CodeName`` type: graph inputs arrive as plain
    dicts whatever the annotation, so a typo'd code name (``"pww"``) would
    otherwise silently no-op — the merge helpers skip codes they do not
    recognise — which violates the explicit-failure standard. Call once at each
    graph's merge entry.

    Raises:
        ValueError: If any key is not one of :data:`CODE_NAMES`.
    """
    if not parallelization:
        return
    unknown = sorted(name for name in dict(parallelization) if name not in CODE_NAMES)
    if unknown:
        raise ValueError(
            f"unknown parallelization code name(s): {unknown}; "
            f"valid codes are {sorted(CODE_NAMES)}."
        )


def resolve_parallelization(
    parallelization: ParallelizationDict | None, code: CodeName, *, pools: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(options, settings)`` for ``code`` from a parallelization mapping.

    ``parallelization`` is keyed by code name; each value is a plain dict with
    optional ``ntasks`` (MPI ranks -> ``metadata.options.resources``), ``npool``
    (k-point pools -> ``-npool``), ``pd`` (pencil decomposition -> ``-pd true``),
    and ``omp`` (per-rank BLAS threads -> ``metadata.options.prepend_text``).
    The two command-line flags are emitted npool-before-pd. Unlike npool/pd,
    ``omp`` is a plain environment knob accepted for every code — no support
    matrix.

    ``pools=False`` suppresses ``-npool`` for a step whose executable takes no
    pools even though the code generally does (the kcw.x ham step).
    """
    if not parallelization:
        return {}, {}
    cfg_entry = parallelization.get(code)
    if not cfg_entry:
        return {}, {}
    # Rebuild the entry into a plain dict: a wrapt-proxied graph input must
    # never reach a namespace socket as a TaggedValue, which rejects it.
    cfg: dict[str, Any] = dict(cfg_entry)
    options: dict[str, Any] = {}
    ntasks = cfg.get("ntasks")
    npool = cfg.get("npool")
    pd = cfg.get("pd")
    omp = cfg.get("omp")
    if ntasks is not None:
        # ``num_machines`` + ``num_mpiprocs_per_machine`` is the one resource
        # shape every scheduler in play accepts: native for the node-counting
        # schedulers, and hyperqueue's back-compat path maps it to num_cpus
        # (its own class ignores ``tot_num_mpiprocs``, silently yielding
        # single-rank jobs).
        options = {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": int(ntasks)}}
    if omp is not None:
        options["prepend_text"] = omp_prepend_text(omp)
    cmdline: list[str] = []
    if npool is not None and pools:
        if code not in POOL_SUPPORTING_CODES:
            raise ValueError(
                f"'npool' was requested for {code!r}, which does not parallelize over "
                f"k-point pools; pools are only valid for {sorted(POOL_SUPPORTING_CODES)}."
            )
        cmdline += ["-npool", str(int(npool))]
    if pd:
        if code not in PD_SUPPORTING_CODES:
            raise ValueError(
                f"'pd' (pencil decomposition) was requested for {code!r}, which does not "
                f"support it; pd is only valid for {sorted(PD_SUPPORTING_CODES)}."
            )
        cmdline += ["-pd", "true"]
    settings: dict[str, Any] = {"cmdline": cmdline} if cmdline else {}
    return options, settings


def _merge_into_namespace(
    namespace: dict[str, Any], options: dict[str, Any], settings: dict[str, Any]
) -> None:
    """Merge ``metadata.options`` / ``settings`` into a CalcJob-input namespace, in place.

    Preserves an existing ``metadata`` (e.g. a ``call_link_label``), an existing
    ``metadata.options`` (e.g. a ``prepend_text`` already set by the step), and
    an existing ``settings`` (e.g. ``additional_retrieve_list``). A
    ``prepend_text`` in ``options`` is *appended* (newline-joined) to any already
    present rather than clobbering it, so the omp export block adds to — never
    replaces — a step's own prepend.
    """
    if options:
        metadata = dict(namespace.get("metadata") or {})
        merged_options = dict(metadata.get("options") or {})
        incoming = dict(options)
        incoming_prepend = incoming.pop("prepend_text", None)
        merged_options.update(incoming)
        if incoming_prepend:
            existing_prepend = merged_options.get("prepend_text")
            merged_options["prepend_text"] = (
                f"{existing_prepend}\n{incoming_prepend}" if existing_prepend else incoming_prepend
            )
        metadata["options"] = merged_options
        namespace["metadata"] = metadata
    if settings:
        merged = dict(namespace.get("settings") or {})
        merged.update(settings)
        namespace["settings"] = merged


def merge_parallelization_into_inputs(
    step_inputs: dict[str, Any],
    parallelization: ParallelizationDict | None,
    code: CodeName,
    *,
    pools: bool = True,
) -> None:
    """Inject ``code``'s ``metadata.options`` / ``settings.cmdline`` into a CalcJob step's inputs.

    Operates in place on ``step_inputs``. Pass ``pools=False`` for a step whose
    executable takes no ``-npool`` even though the code generally does (the
    kcw.x ham step).
    """
    options, settings = resolve_parallelization(parallelization, code, pools=pools)
    _merge_into_namespace(step_inputs, options, settings)


def merge_parallelization_into_overrides(
    overrides: dict[str, Any],
    parallelization: ParallelizationDict | None,
    mapping: Iterable[tuple[tuple[str, ...], CodeName]],
) -> None:
    """Merge per-code parallelization into WorkChain ``overrides`` namespaces, in place.

    ``mapping`` pairs each calcjob-namespace *path* with the code driving it.
    The path locates the calcjob namespace inside ``overrides``: e.g.
    ``(("scf", "pw"), "pw")`` for a nested PwBaseWorkChain step,
    ``(("projwfc",), "projwfc")`` for a direct calcjob namespace. For each
    pair the code's ``metadata.options`` and ``settings.cmdline`` are merged
    under ``overrides[path...]``.
    """
    for path, code in mapping:
        options, settings = resolve_parallelization(parallelization, code)
        if not options and not settings:
            continue
        namespace = overrides
        for part in path:
            namespace = namespace.setdefault(part, {})
        _merge_into_namespace(namespace, options, settings)


def merge_parallelization_into_existing_namespaces(
    data: dict[str, Any],
    parallelization: ParallelizationDict | None,
    mapping: Iterable[tuple[tuple[str, ...], CodeName]],
) -> None:
    """Merge per-code parallelization into ``data`` namespaces that already exist.

    Like :func:`merge_parallelization_into_overrides` but never creates a namespace: a path
    absent from ``data`` (e.g. the ``projwfc`` step the workchain isn't running)
    is skipped. For post-builder ``data`` dicts where the present namespaces
    depend on the run.
    """
    for path, code in mapping:
        options, settings = resolve_parallelization(parallelization, code)
        if not options and not settings:
            continue
        namespace: object = data
        for part in path:
            if not isinstance(namespace, dict) or part not in namespace:
                namespace = None
                break
            namespace = namespace[part]
        if isinstance(namespace, dict):
            _merge_into_namespace(namespace, options, settings)
