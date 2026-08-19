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
from enum import Enum
from typing import Any

from aiida import orm
from aiida_workgraph import task

from aiida_koopmans.owned_keywords import owned


def unwrap_enum[EnumT: Enum](value: Any, enum_cls: type[EnumT]) -> EnumT | None:
    """Return ``value`` as a member of ``enum_cls``, or ``None`` for ``None``.

    Accepts a member, a proxy wrapping one, or the bare value string. A
    graph input is always a proxy, and a protocol builder that branches on
    ``argument is SomeEnum.MEMBER`` takes the wrong branch for one:
    ``==`` forwards through the proxy, ``is`` does not. Call this on an
    enum whose destination builder branches by identity, whether or not
    the calling body runs eagerly today — where a graph sits in the call
    tree is the caller's choice.
    """
    if value is None:
        return None
    return enum_cls(getattr(value, "value", value))


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


def enforce_step_calculation(params: dict[str, Any], step: str, expected: str) -> dict[str, Any]:
    """Stamp the ``CONTROL.calculation`` a step owns, raising on a conflicting explicit value.

    Each step in a multi-step graph (scf, nscf, bands, ...) owns its
    ``CONTROL.calculation`` mode. After the protocol defaults and caller
    overrides are merged, assert the merged parameters carry no *different*
    explicit calculation and set ``expected`` in place. A matching explicit
    value is accepted; a genuine conflict raises so no override is dropped
    silently.

    Args:
        params: The pw ``parameters`` namelist dict (mutated in place).
        step: The step name, used only for the error message.
        expected: The calculation mode this step requires.

    Returns:
        The same ``params`` dict, with ``CONTROL.calculation`` set to ``expected``.

    Raises:
        ValueError: If ``params`` already sets a different ``CONTROL.calculation``.
    """
    control = params.setdefault("CONTROL", {})
    found = control.get("calculation")
    if found is not None and found != expected:
        raise ValueError(
            f"the {step!r} step requires CONTROL.calculation={expected!r}, but the "
            f"merged parameters set calculation={found!r}; remove the conflicting override."
        )
    control["calculation"] = expected
    return params


def name_step(inputs: dict[str, Any], display: str) -> None:
    """Name one process for a reader, via ``metadata.label``, in place.

    The label is what a progress display and ``verdi process list`` show
    instead of the class name, so it reads as the step it stands for
    (``SCF``, ``Trial KI``) rather than as an identifier. It is mutable
    metadata and takes no part in the caching hash: ``ProcessNode``
    excludes ``metadata_inputs`` from the hashed attributes, and the node
    label is a column rather than an attribute.

    Args:
        inputs: The inputs of one process, or of one exposed namespace of
            a workchain that launches it (mutated in place).
        display: The name to show.
    """
    inputs.setdefault("metadata", {})["label"] = display


@task.calcfunction
def stamp_render_intent(kind: str) -> orm.Bool:
    """Set ``koopmans_render`` extras on the process that called this task.

    AiiDA's ``metadata`` input namespace is spec'd (``label``,
    ``description``, ``call_link_label``, ``store_provenance``,
    ``disable_cache``) and takes no arbitrary keys, so a render-intent flag
    cannot ride through ``metadata`` the way :func:`name_step` rides
    ``metadata.label``. Extras have no such namespace: they are free-form
    and settable on any stored node. Runs as a genuine (in-process)
    ``@task.calcfunction`` so ``Process.current()`` resolves to itself and
    ``.caller`` to whichever process invoked it — the row a reader actually
    sees, whichever of its call sites ran.

    Args:
        kind: ``"transparent"`` (render invisible, children hoisted onto
            the parent) or ``"numbered"`` (positional "Iteration N"
            numbering rather than a name).
    """
    from aiida.engine import Process

    current = Process.current()
    caller = None if current is None else current.node.caller  # type: ignore[attr-defined]
    if caller is not None:
        caller.base.extras.set("koopmans_render", {getattr(kind, "value", kind): True})
    return orm.Bool(True)


def mark_step(*, transparent: bool = False, numbered: bool = False) -> None:
    """Stamp render intent on the enclosing ``@task.graph`` run.

    Call once, unconditionally, near the top of the ``@task.graph`` body
    whose rendering this marks — not at each of its call sites. The intent
    is a property of which function ran (:func:`stamp_render_intent` reads
    it off its own ``ProcessNode.caller`` at run time), not of who called
    it, so one call per graph body covers every call site at once.
    """
    if transparent:
        stamp_render_intent(kind="transparent")
    if numbered:
        stamp_render_intent(kind="numbered")


def force_pw_verbosity(pw_inputs: dict[str, Any]) -> None:
    """Set the ``CONTROL.verbosity`` every koopmans pw.x calculation runs at.

    Applied after the protocol defaults and caller overrides are merged, so
    the forced value replaces any the merged parameters carry. Call it on
    every pw.x step a route assembles.

    Args:
        pw_inputs: One built ``PwCalculation`` input namespace; its
            ``parameters`` node is replaced (mutated in place).
    """
    parameters = pw_inputs["parameters"].get_dict()
    parameters.setdefault("CONTROL", {}).update(owned("pw.CONTROL", {"verbosity": "high"}))
    pw_inputs["parameters"] = orm.Dict(parameters)


def pin_kpoints(inputs: dict[str, Any], kpoints: orm.KpointsData | None) -> None:
    """Replace a ``PwBaseWorkChain`` step's protocol mesh with ``kpoints``, in place.

    The workchain accepts exactly one of ``kpoints`` and
    ``kpoints_distance``, so the protocol's distance has to go rather than
    be overruled — as does ``kpoints_force_parity``, which only qualifies
    the distance. A ``None`` mesh leaves the protocol in charge.

    Args:
        inputs: One ``PwBaseWorkChain`` step's flattened inputs (mutated in place).
        kpoints: The mesh (or explicit list) the step samples.
    """
    if kpoints is None:
        return
    inputs.pop("kpoints_distance", None)
    inputs.pop("kpoints_force_parity", None)
    inputs["kpoints"] = kpoints
