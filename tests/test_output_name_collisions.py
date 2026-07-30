"""Package-wide guard against leaf-vs-namespace output-name collisions.

Every python task in a daemon worker validates its outputs against one
shared, process-wide port specification. A cache hit on a task with
namespace outputs leaves a ``PortNamespace`` behind on it under each
namespace name, and a namespace port only accepts a mapping — so a later
task emitting a plain (leaf) output under the same name fails validation
with ``not sub class of `Mapping```. The only robust contract is
package-wide: an output name is either always a namespace or always a
leaf, never both.

``tests/test_ml_workgraph.py::TestSharedOutputSpecCollision`` reproduces
the failure live for one pair; this module enforces the naming invariant
statically for every task the package defines.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections import defaultdict
from typing import TypedDict

from aiida_workgraph import task
from node_graph.socket_spec import SocketSpec, SocketView

import aiida_koopmans

NAMESPACE_IDENTIFIER = "workgraph.namespace"


class Clash(TypedDict):
    """The nested namespace of :func:`emits_namespace`."""

    first: int
    second: int


class NamespaceOutputs(TypedDict):
    """One side of the deliberate collision: ``shared_name`` as a namespace."""

    shared_name: Clash


class LeafOutputs(TypedDict):
    """The other side of the deliberate collision: ``shared_name`` as a leaf."""

    shared_name: int


# Module scope, not inside the test: the decorator resolves the (stringified,
# via ``from __future__ import annotations``) return annotation against the
# module globals, where function-local TypedDicts are invisible.
@task
def emits_namespace() -> NamespaceOutputs:
    """Emit ``shared_name`` as a namespace output."""
    return {"shared_name": {"first": 1, "second": 2}}


@task
def emits_leaf() -> LeafOutputs:
    """Emit ``shared_name`` as a leaf output."""
    return {"shared_name": 3}


NameRegistry = dict[str, dict[str, set[str]]]


def _record_output_names(spec: SocketSpec, owner: str, names: NameRegistry) -> None:
    """Record every output name under ``spec`` as ``namespace`` or ``leaf``.

    Recurses through nested namespaces and through the item spec of dynamic
    namespaces (whose keys are runtime data, but whose item fields become
    real port names).
    """
    for name, child in (spec.fields or {}).items():
        kind = "namespace" if child.identifier == NAMESPACE_IDENTIFIER else "leaf"
        names[name][kind].add(owner)
        _record_output_names(child, owner, names)
    if spec.item is not None:
        _record_output_names(spec.item, owner, names)


def _collisions(names: NameRegistry) -> dict[str, dict[str, list[str]]]:
    """Return the names recorded both as a namespace and as a leaf."""
    return {
        name: {kind: sorted(owners) for kind, owners in kinds.items()}
        for name, kinds in sorted(names.items())
        if len(kinds) == 2
    }


def _package_output_names() -> NameRegistry:
    """Walk every task the package defines and register its output names.

    Covers ``@task`` functions, ``@task.graph`` graphs and module-scope
    ``task(Process)`` wrappers alike: each is a handle whose ``outputs``
    attribute is a :class:`SocketView` over its output spec.
    """
    names: NameRegistry = defaultdict(lambda: defaultdict(set))
    seen: set[int] = set()
    for module_info in pkgutil.walk_packages(aiida_koopmans.__path__, prefix="aiida_koopmans."):
        module = importlib.import_module(module_info.name)
        for attribute, obj in vars(module).items():
            view = getattr(obj, "outputs", None)
            if not isinstance(view, SocketView) or id(obj) in seen:
                continue
            seen.add(id(obj))
            _record_output_names(view.to_spec(), f"{module_info.name}.{attribute}", names)
    if not names:
        raise RuntimeError("The package walk found no task output specs.")
    return names


def test_no_output_name_is_both_namespace_and_leaf():
    """No output name may be a namespace in one task and a leaf in another."""
    offenders = _collisions(_package_output_names())
    assert not offenders, (
        "Output names used both as a namespace and as a leaf (a cache hit on "
        "the namespace side poisons the shared output spec for the leaf "
        f"side — rename one of them): {offenders}"
    )


def test_walker_registers_the_known_namespace_and_leaf_families():
    """The walk sees the screening sockets it exists to keep apart.

    ``alphas`` is the kcp.x screening namespace (``filled`` / ``empty``);
    ``alpha_values`` is the kcw.x screening leaf list. If the walker stopped
    finding either, the sibling test would pass vacuously.
    """
    names = _package_output_names()
    assert set(names["alphas"]) == {"namespace"}
    assert set(names["alpha_values"]) == {"leaf"}


def test_walker_flags_a_deliberate_collision():
    """The collision detector reports a namespace/leaf pair it is fed.

    A pair of tasks reusing one name across the two kinds must be flagged —
    otherwise a clean package result proves nothing.
    """
    names: NameRegistry = defaultdict(lambda: defaultdict(set))
    _record_output_names(emits_namespace.outputs.to_spec(), "emits_namespace", names)
    _record_output_names(emits_leaf.outputs.to_spec(), "emits_leaf", names)

    assert _collisions(names) == {
        "shared_name": {"leaf": ["emits_leaf"], "namespace": ["emits_namespace"]}
    }
