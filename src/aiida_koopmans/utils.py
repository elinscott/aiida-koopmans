"""Shared helpers used across aiida-koopmans workgraphs and CalcJobs.

Keep this module free of kcp/wannier/pw-specific logic — those live in their
respective modules. Anything here should be generic enough that a future
workgraph would import it too.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from aiida import orm
from aiida.plugins import DataFactory
from aiida_workgraph import dynamic, task

UpfData = DataFactory("pseudo.upf")


class ElectronCountOutputs(TypedDict):
    """Outputs of :func:`count_electrons_task`.

    ``nelup`` and ``neldw`` are ``None`` when ``nspin == 1``; downstream
    consumers gate on ``nspin`` (a build-time scalar) before reading
    them.
    """

    nelec: int
    nelup: int | None
    neldw: int | None


class FilledEmptyCountOutputs(TypedDict):
    """Outputs of :func:`filled_and_empty_counts_task`."""

    n_filled: int
    n_empty: int


def passthrough_node(node):
    """Identity deserializer that keeps an AiiDA Data socket as a node.

    ``aiida_pythonjob``'s default deserializer eagerly converts known Data
    types (e.g. ``StructureData → ase.Atoms``); for types it doesn't know
    (e.g. ``UpfData``) it raises because they have no ``.value`` and no
    registered deserializer. koopmans tasks that need the AiiDA node
    (e.g. for ``family.get_pseudos(structure=...)``,
    ``structure.sites`` access, or passing pseudos to ``KcpCalculation``)
    register this passthrough via ``@task(deserializers=...)``.
    """
    return node


# Plug in via ``@task(deserializers=KOOPMANS_NODE_DESERIALIZERS)`` (or
# extended copies thereof) on PyFunction tasks that take AiiDA Data
# inputs but want the node, not its deserialized payload.
KOOPMANS_NODE_DESERIALIZERS = {
    "aiida.orm.nodes.data.structure.StructureData": ("aiida_koopmans.utils.passthrough_node"),
    "aiida_pseudo.data.pseudo.upf.UpfData": ("aiida_koopmans.utils.passthrough_node"),
}


def resolve_pseudo_family(family_label: str, structure: orm.StructureData) -> dict[str, UpfData]:
    """Resolve an ``aiida-pseudo`` family label into a ``{kind_name: UpfData}`` dict.

    Args:
        family_label: The ``label`` of a stored ``PseudoPotentialFamily`` group
            (e.g. ``"SG15/1.2/PBE/SR"``).
        structure: The :class:`~aiida.orm.StructureData` whose kinds need pseudos.

    Returns:
        A dict mapping each kind name in ``structure`` to its ``UpfData`` node.

    Raises:
        :class:`~aiida.common.exceptions.NotExistent`: if no family with that
            label exists in the current profile.
    """
    from aiida_pseudo.groups.family.pseudo import PseudoPotentialFamily

    family = (
        orm.QueryBuilder().append(PseudoPotentialFamily, filters={"label": family_label}).one()[0]
    )
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


def count_electrons(
    structure: orm.StructureData,
    pseudos: dict[str, UpfData],
    *,
    nspin: int,
    tot_magnetization: int | None = None,
) -> tuple[int, int | None, int | None]:
    """Sum ``z_valence`` across sites to get ``(nelec, nelup, neldw)``.

    For ``nspin == 1`` the per-spin counts are returned as ``None``; for
    ``nspin == 2`` they are computed from ``nelec`` and ``tot_magnetization``
    (which defaults to 0 — i.e., closed shell).

    Raises:
        ValueError: if the total valence charge is non-integer (indicates a
            charged structure or a bad pseudo), or if the given
            ``tot_magnetization`` is inconsistent with ``nelec``.
    """
    nelec_total = 0.0
    for site in structure.sites:
        nelec_total += float(pseudos[site.kind_name].z_valence)
    nelec = round(nelec_total)
    if nelec != nelec_total:
        raise ValueError(
            f"Non-integer total valence charge {nelec_total} from pseudos — "
            "structure may be charged or a pseudo has a non-integer z_valence."
        )
    if nspin == 1:
        return nelec, None, None

    m = tot_magnetization if tot_magnetization is not None else 0
    if (nelec + m) % 2 or (nelec - m) % 2:
        raise ValueError(f"nelec={nelec}, tot_magnetization={m} give non-integer spin populations.")
    return nelec, (nelec + m) // 2, (nelec - m) // 2


@task(deserializers=KOOPMANS_NODE_DESERIALIZERS)
def count_electrons_task(
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    nspin: int,
    tot_magnetization: int | None = None,
) -> ElectronCountOutputs:
    """Runtime task variant of :func:`count_electrons`.

    Outputs are emitted as three named sockets so downstream
    ``@task.graph`` builders can wire ``nelec`` / ``nelup`` / ``neldw``
    independently. ``nelup`` and ``neldw`` are ``None`` when
    ``nspin == 1``.
    """
    nelec, nelup, neldw = count_electrons(
        structure, pseudos, nspin=nspin, tot_magnetization=tot_magnetization
    )
    return {"nelec": nelec, "nelup": nelup, "neldw": neldw}


@task
def filled_and_empty_counts_task(
    nspin: int,
    nbnd: int,
    nelec: int,
    nelup: int | None = None,
    neldw: int | None = None,
) -> FilledEmptyCountOutputs:
    """Runtime task variant of :func:`filled_and_empty_counts`.

    For the DSCF refinement loop, where the totals must come out of
    socket-valued ``nelec`` / ``nelup`` / ``neldw``.
    """
    n_filled, n_empty = filled_and_empty_counts(
        nspin=nspin, nbnd=nbnd, nelec=nelec, nelup=nelup, neldw=neldw
    )
    return {"n_filled": n_filled, "n_empty": n_empty}


def filled_and_empty_counts(
    *,
    nspin: int,
    nbnd: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
) -> tuple[int, int]:
    """Return total filled / empty orbital counts across spin channels.

    Used for sizing the ``file_alpharef[_empty].txt`` screening-parameter files.

    For ``nspin == 2`` ``nelup`` and ``neldw`` must be provided.
    """
    if nspin == 2:
        if nelup is None or neldw is None:
            raise ValueError("nelup and neldw are required when nspin=2")
        n_filled = nelup + neldw
        n_empty = max(0, nbnd - nelup) + max(0, nbnd - neldw)
    else:
        n_per_spin = nelec // 2
        n_filled = n_per_spin
        n_empty = max(0, nbnd - n_per_spin)
    return n_filled, n_empty
