"""Primitive-cell → supercell conversion for Γ-only kcp.x runs.

kcp.x has no k-point sampling: a periodic Koopmans DSCF calculation runs on
the Γ-point of a supercell whose repeat counts equal the primitive-cell
Monkhorst-Pack grid. This module owns
that conversion plus the plain-Python scaling helpers for the *extensive*
kcp.x parameters.

Of the extensive parameters (``nelec``, ``nelup``, ``neldw``, ``nbnd``, ``conv_thr``,
``esic_conv_thr``, ``tot_charge``, ``tot_magnetization``) only ``nbnd`` and
``tot_magnetization`` need explicit scaling here: the electron counts come
out of :func:`~aiida_koopmans.utils.count_electrons_task` evaluated on the
supercell structure (so they scale with the atom count automatically), and
the convergence thresholds are derived as ``1e-9 * nelec`` inside the kcp.x
parameter builders (so they follow ``nelec``).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from aiida import orm
from aiida_workgraph import task


def supercell_size(kgrid: Sequence[int]) -> int:
    """Return the number of primitive cells in the supercell, ``prod(kgrid)``."""
    return math.prod(kgrid)


def scale_extensive(value: int | None, ncells: int) -> int | None:
    """Scale an extensive per-primitive-cell parameter to the supercell.

    ``None`` passes through (e.g. an unset ``tot_magnetization``).
    """
    return None if value is None else value * ncells


@task.calcfunction
def primitive_to_supercell(
    structure: orm.StructureData,
    kgrid: orm.List,
) -> orm.StructureData:
    """Repeat a primitive cell into the supercell matching a k-point grid.

    The transformation matrix is ``diag(kgrid)`` — the supercell contains
    ``prod(kgrid)`` primitive cells, so the Γ-point of the supercell samples
    exactly the primitive Monkhorst-Pack grid. Because the matrix is
    diagonal, the repeat is a plain site replication (no ASE round-trip
    needed, so custom kind names survive): sites are emitted cell-major —
    the full primitive basis for translation (0,0,0) first, then for
    (0,0,1), and so on — matching ASE ``make_supercell``'s default order.

    A ``@task.calcfunction`` so the derived ``StructureData`` carries
    provenance back to the primitive structure. Single-output convention:
    consumers wire the supercell via ``primitive_to_supercell(...).result``.
    """
    counts = [int(n) for n in kgrid.get_list()]
    if len(counts) != 3 or any(n < 1 for n in counts):
        raise ValueError(f"kgrid must be three positive integers, got {kgrid.get_list()!r}")

    cell = structure.cell
    supercell = orm.StructureData(
        cell=[
            [n * component for component in vector] for n, vector in zip(counts, cell, strict=True)
        ],
        pbc=structure.pbc,
    )
    for kind in structure.kinds:
        supercell.append_kind(kind)
    for i, j, k in itertools.product(*(range(n) for n in counts)):
        shift = [i * cell[0][d] + j * cell[1][d] + k * cell[2][d] for d in range(3)]
        for site in structure.sites:
            supercell.append_site(
                orm.Site(
                    kind_name=site.kind_name,
                    position=[p + s for p, s in zip(site.position, shift, strict=True)],
                )
            )
    return supercell
