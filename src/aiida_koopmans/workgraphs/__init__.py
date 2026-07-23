"""WorkGraph-based workflows for koopmans calculations.

Naming convention: case encodes what a call creates. PascalCase names
create process nodes — verb-first ``@task.graph`` builders
(``WannierizeBlock``, ``RunScfNscf``; ``Workflow`` suffix reserved for the
dispatcher entry points) and ``Step``-suffixed ``task(WorkChain/CalcJob)``
constants (``KcpStep``, ``PwBaseStep``). snake_case names are in-process
leaf ``@task`` / calcfunction / workfunction computations.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypedDict

from aiida import orm


class Codes(TypedDict, total=False):
    """Code instances used across koopmans workgraphs."""

    pw: orm.AbstractCode
    pw2wannier90: orm.AbstractCode
    wannier90: orm.AbstractCode
    projwfc: orm.AbstractCode
    dos: orm.AbstractCode
    kcp: orm.AbstractCode
    kcw: orm.AbstractCode
    ph: orm.AbstractCode
    wann2kcp: orm.AbstractCode
    merge_evc: orm.AbstractCode


def inject_pseudo_family(
    overrides: dict, pseudo_family: str | None, namespaces: Iterable[str]
) -> None:
    """Set ``pseudo_family`` under each of ``namespaces`` in ``overrides``, in place.

    The protocol-based ``PwBaseWorkChain`` / ``PwBandsWorkChain`` /
    ``PdosWorkChain`` builders take the pseudo family as a per-sub-workchain
    override (``overrides["scf"]["pseudo_family"]``, …) rather than a
    top-level argument, so each caller has to seed it under every namespace
    it drives. ``setdefault`` preserves an explicit family already present in
    the overrides. A ``None`` family is a no-op (the protocol default applies).
    """
    if pseudo_family is None:
        return
    for namespace in namespaces:
        overrides.setdefault(namespace, {}).setdefault("pseudo_family", pseudo_family)


# QE codes that accept ``-npool`` (k-point pools) and ``-pd`` (pencil
# decomposition) on the command line. Ground truth is the legacy
# ``koopmans.commands`` per-executable config classes; mirrors the koopmans2
# schema. ``kcw`` accepts pools only for its wann2kc / screen steps, not ham —
# that per-step split is the ``pools`` argument below, not a code-level fact.
POOL_SUPPORTING_CODES = frozenset({"pw", "projwfc", "kcw"})
PD_SUPPORTING_CODES = frozenset({"pw", "pw2wannier90", "projwfc", "kcw"})


def resolve_parallelization(
    parallelization: dict[str, Any] | None, code: str, *, pools: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(options, settings)`` for ``code`` from a parallelization mapping.

    ``parallelization`` is keyed by code name; each value is a plain dict with
    optional ``ntasks`` (MPI ranks -> ``metadata.options.resources``), ``npool``
    (k-point pools -> ``-npool``), and ``pd`` (pencil decomposition ->
    ``-pd true``). The two flags are emitted npool-before-pd, matching the
    legacy command rendering. Everything is rebuilt into fresh plain dicts so a
    wrapt-proxied graph input (a ``TaggedValue``) never reaches a namespace
    socket, which rejects it.

    ``pools=False`` suppresses ``-npool`` for a step whose executable takes no
    pools even though the code generally does (the kcw.x ham step).
    """
    if not parallelization:
        return {}, {}
    cfg = dict(parallelization).get(code)
    if not cfg:
        return {}, {}
    cfg = dict(cfg)
    options: dict[str, Any] = {}
    ntasks = cfg.get("ntasks")
    npool = cfg.get("npool")
    pd = cfg.get("pd")
    if ntasks is not None:
        options = {"resources": {"num_machines": 1, "tot_num_mpiprocs": int(ntasks)}}
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

    Preserves an existing ``metadata`` (e.g. a ``call_link_label``) and an
    existing ``settings`` (e.g. ``additional_retrieve_list``).
    """
    if options:
        metadata = dict(namespace.get("metadata") or {})
        metadata["options"] = options
        namespace["metadata"] = metadata
    if settings:
        merged = dict(namespace.get("settings") or {})
        merged.update(settings)
        namespace["settings"] = merged


def apply_parallelization(
    step_inputs: dict[str, Any],
    parallelization: dict[str, Any] | None,
    code: str,
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


def inject_parallelization(
    overrides: dict[str, Any],
    parallelization: dict[str, Any] | None,
    mapping: Iterable[tuple[tuple[str, ...], str]],
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


def apply_parallelization_present(
    data: dict[str, Any],
    parallelization: dict[str, Any] | None,
    mapping: Iterable[tuple[tuple[str, ...], str]],
) -> None:
    """Merge per-code parallelization into ``data`` namespaces that already exist.

    Like :func:`inject_parallelization` but never creates a namespace: a path
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
