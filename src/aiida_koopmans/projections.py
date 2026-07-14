"""Projection-spec accounting shared by the workgraph builders.

Converts user projection specs (``wannier90_input`` ``Projection`` models)
into Wannier90 ``.win`` projection strings and Wannier-function counts.
"""

from __future__ import annotations

from typing import Any

from aiida import orm


def projection_win_string(projection: Any) -> str:
    """Format one projection as a Wannier90 ``.win`` projections line.

    ``projection`` is duck-typed on the ``wannier90_input`` ``Projection``
    model. Element-labelled sites render as ``<element>:<ang_mtm>``;
    single-point sites use Wannier90's ``f=x,y,z`` (crystal) / ``c=x,y,z``
    (Cartesian) forms. The ``ang_mtm`` quantum numbers stringify to
    Wannier90's own syntax (``l=-3`` for sp3, ...).
    """
    if projection.site is not None:
        return f"{projection.site}:{projection.ang_mtm}"
    fractional = getattr(projection, "fractional_site", None)
    if fractional is not None:
        return f"f={','.join(str(c) for c in fractional)}:{projection.ang_mtm}"
    cartesian = getattr(projection, "cartesian_site", None)
    if cartesian is not None:
        return f"c={','.join(str(c) for c in cartesian)}:{projection.ang_mtm}"
    raise ValueError(f"Projection {projection!r} defines no site.")


def projection_num_wann(structure: orm.StructureData, projection: Any) -> int:
    """Count the Wannier functions of one projection: site multiplicity x (2l+1).

    ``projection`` is duck-typed on the ``wannier90_input`` ``Projection``
    model (``.site`` element label or a ``fractional_site`` /
    ``cartesian_site`` single point, ``.ang_mtm`` quantum numbers).
    """
    if projection.site is not None:
        n_sites = sum(1 for site in structure.sites if site.kind_name == projection.site)
        if n_sites == 0:
            raise ValueError(
                f"Projection site '{projection.site}' does not match any atom in the structure."
            )
    elif (
        getattr(projection, "fractional_site", None) is not None
        or getattr(projection, "cartesian_site", None) is not None
    ):
        # An explicit point hosts exactly one set of orbitals.
        n_sites = 1
    else:
        raise ValueError(f"Projection {projection!r} defines no site.")
    quantum_numbers = projection.ang_mtm
    if quantum_numbers.m_r is not None:
        multiplicity = len(quantum_numbers.m_r)
    else:
        l_value = quantum_numbers.angular.value
        # Hybrids are encoded with negative l: sp=-1 (2 orbitals), sp2=-2 (3),
        # sp3=-3 (4), sp3d=-4 (5), sp3d2=-5 (6).
        multiplicity = 2 * l_value + 1 if l_value >= 0 else 1 - l_value
    return n_sites * multiplicity


def band_range_complement(start: int, end: int, nbnd: int) -> list[int] | None:
    """Return the wannier90 ``exclude_bands`` list complementing ``[start, end]``.

    A list of band indices (not the ``.win`` range string): aiida-wannier90's
    input writer expects integers and does the range compression itself.
    """
    excluded = [*range(1, start), *range(end + 1, nbnd + 1)]
    return excluded or None
