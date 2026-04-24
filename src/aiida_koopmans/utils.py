"""Shared helpers used across aiida-koopmans workgraphs and CalcJobs.

Keep this module free of kcp/wannier/pw-specific logic — those live in their
respective modules. Anything here should be generic enough that a future
workgraph would import it too.
"""

from __future__ import annotations

from aiida import orm
from aiida.plugins import DataFactory

UpfData = DataFactory("pseudo.upf")


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
