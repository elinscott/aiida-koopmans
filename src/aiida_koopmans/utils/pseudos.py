"""Pseudopotential-family resolution for workgraph tasks."""

from __future__ import annotations

from typing import Annotated

from aiida import orm
from aiida_pseudo.data.pseudo.upf import UpfData
from aiida_workgraph import dynamic, task


def resolve_pseudo_family(family_label: str, structure: orm.StructureData) -> dict[str, UpfData]:
    """Resolve an ``aiida-pseudo`` family label into a ``{kind_name: UpfData}`` dict.

    Args:
        family_label: The ``label`` of a stored ``PseudoPotentialFamily`` group
            (e.g. ``"SG15/1.2/PBE/SR"``).
        structure: The :class:`~aiida.orm.StructureData` whose kinds need pseudos.

    Returns:
        A dict mapping each kind name in ``structure`` to its ``UpfData`` node.

    Raises:
        :class:`~aiida.common.exceptions.NotExistent`: if no group with that
            label exists in the current profile.
        :class:`~aiida.common.exceptions.MultipleObjectsError`: if more than
            one group shares the label.
    """
    family = orm.load_group(family_label)
    return family.get_pseudos(structure=structure)


@task.workfunction()
def resolve_pseudo_family_task(
    family_label: orm.Str,
    structure: orm.StructureData,
) -> Annotated[dict, dynamic(UpfData)]:
    """Workfunction variant of :func:`resolve_pseudo_family`.

    A ``@task.workfunction`` (not ``@task``) because the body returns
    already-stored ``UpfData`` nodes from the family group — calcfunctions
    (and ``aiida_pythonjob.PyFunction``, which is a calcfunction-style
    process) reject that under provenance rules. Workfunctions are
    explicitly allowed to *select* existing nodes.

    A side-effect: workfunction inputs arrive as AiiDA Data, so
    ``family_label`` is an ``orm.Str``; reach the underlying string via
    ``.value`` (NOT ``str(...)``, which returns the node's
    ``"uuid: ... value: ..."`` repr and silently breaks the QueryBuilder
    filter). ``structure`` passes through as a ``StructureData`` node —
    no manual conversion needed.

    Single-output convention: consumers wire the resolved pseudos via
    ``resolve_pseudo_family_task(...).result``.
    """
    return resolve_pseudo_family(family_label.value, structure)
