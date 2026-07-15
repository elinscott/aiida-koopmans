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
from typing import TypedDict

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
