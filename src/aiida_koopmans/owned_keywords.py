"""Which Quantum ESPRESSO and wannier90 keywords the koopmans routes determine.

A keyword is **owned** when a route's value wins over anything the caller
supplies: the route force-merges it on top of the overrides, so a caller
value is discarded. ``koopmans`` generates its input-file models from these
sets, dropping every owned keyword from the schema, so an input file cannot
state one at all.

A keyword is **seeded** when the route writes a starting value that the
caller's own value replaces. Seeded keywords stay settable.

:func:`owned` and :func:`seeded` guard the route literals: a keyword that
appears in one but is classified in neither raises, so a route cannot start
forcing a keyword that ``koopmans`` still advertises as settable.

Data and plain-Python checks only — no AiiDA imports, so ``koopmans`` can
read this at build time.
"""

from collections.abc import Mapping
from typing import Any

__all__ = [
    "OWNED",
    "ROUTE_CONDITIONAL",
    "SEEDED",
    "owned",
    "seeded",
]

#: Keywords a route always determines, keyed by the ``koopmans`` input-file
#: block they would appear in (``<calculator>`` for wannier90, which has no
#: namelists, ``<calculator>.<NAMELIST>`` otherwise).
OWNED: dict[str, frozenset[str]] = {
    "pw.CONTROL": frozenset(
        {
            # aiida-quantumespresso's PwCalculation writes the scratch and
            # pseudopotential paths and the file prefix for every run.
            "pseudo_dir",
            "outdir",
            "prefix",
            # Every route reads eigenvalues out of the pw.x output, which
            # only prints them all at high verbosity.
            "verbosity",
        }
    ),
    "pw.SYSTEM": frozenset(
        {
            # Derived from the structure.
            "ibrav",
            "nat",
            "ntyp",
            # One cutoff, stated once (calculator_parameters.ecutwfc), since
            # pw.x and kcp.x always share the grid it derives.
            "ecutwfc",
            "ecutrho",
            # workflow.spin says what the spin treatment is; these are the
            # pw.x spelling of it.
            "nspin",
            "noncolin",
            "lspinorb",
        }
    ),
    "ph.INPUTPH": frozenset(
        {
            # aiida-quantumespresso's PhCalculation writes these itself and
            # rejects a parameters Dict that states them.
            "outdir",
            "prefix",
            "fildyn",
            "ldisp",
            "nq1",
            "nq2",
            "nq3",
            "qplot",
            "verbosity",
            # The only ph.x run koopmans makes is the q = Gamma dielectric
            # response the DFPT screening consumes.
            "epsil",
            "trans",
        }
    ),
    "pw2wannier90.INPUTPP": frozenset(
        {
            # Written by aiida-quantumespresso's Pw2wannier90Calculation.
            "prefix",
            "outdir",
            "seedname",
            # workflow.spin says which channels there are; the route runs one
            # pw2wannier90 per channel and selects it.
            "spin_component",
        }
    ),
    "wannier90": frozenset(
        {
            # Band and manifold bookkeeping, derived from the projections.
            "num_wann",
            "num_bands",
            "exclude_bands",
            # The structure and the k-points come from the atoms and kpoints
            # blocks of the input file.
            "unit_cell_cart",
            "atoms_cart",
            "atoms_frac",
            "mp_grid",
            "kpoints",
            # Requested through workflow.auto_projections.
            "auto_projections",
            # The gauge products every downstream step reads: kcw.x needs the
            # U matrices and the Wannier centres, the supercell fold needs
            # the Hamiltonian.
            "write_hr",
            "write_u_matrices",
            "write_xyz",
            # workflow.spin again, in wannier90's spelling.
            "spinors",
            "spin",
        }
    ),
}

#: Keywords a route writes as a starting value that a caller value replaces.
SEEDED: dict[str, frozenset[str]] = {
    "pw.SYSTEM": frozenset(
        {
            # A tiny moment so QE runs its spin-accounting branch; a
            # genuinely magnetic system states its own.
            "starting_magnetization",
        }
    ),
    "wannier90": frozenset(
        {
            # Minimisation settings tuned for reproducible Wannier centres.
            "guiding_centres",
            "num_iter",
            "num_cg_steps",
            "conv_tol",
            "conv_window",
            "dis_conv_tol",
        }
    ),
}

#: Keywords a route forces on one step while leaving them settable on
#: another, so they are neither wholly owned nor merely seeded. Each is a
#: known gap: the input file still advertises them, and stating one is
#: honoured on some steps and silently discarded on others.
#:
#: * ``nosym`` / ``noinv``: forced on the DFPT chain's nscf, which must
#:   sample the unreduced mesh wannier90 expects, and left alone on its scf.
#: * ``tot_magnetization``: forced to zero under ``workflow.spin = 'none'``
#:   (the DFPT scratch is nspin = 2 for a closed-shell system), and taken
#:   from ``calculator_parameters.tot_magnetization`` otherwise.
ROUTE_CONDITIONAL: dict[str, frozenset[str]] = {
    "pw.SYSTEM": frozenset({"nosym", "noinv", "tot_magnetization"}),
}


def owned[T: Mapping[str, Any]](block: str, keywords: T) -> T:
    """Return ``keywords`` after checking every one of them is owned.

    Args:
        block: The input-file block the keywords belong to, as keyed in
            :data:`OWNED`.
        keywords: A namelist literal a route force-merges over the caller's
            overrides.

    Returns:
        ``keywords``, unchanged.

    Raises:
        ValueError: If a keyword is classified neither owned nor conditional.
    """
    _check(block, keywords, OWNED, "forces")
    return keywords


def seeded[T: Mapping[str, Any]](block: str, keywords: T) -> T:
    """Return ``keywords`` after checking every one of them is seeded.

    Args:
        block: The input-file block the keywords belong to, as keyed in
            :data:`SEEDED`.
        keywords: A namelist literal a route merges underneath the caller's
            overrides.

    Returns:
        ``keywords``, unchanged.

    Raises:
        ValueError: If a keyword is classified neither seeded nor conditional.
    """
    _check(block, keywords, SEEDED, "seeds")
    return keywords


def _check(
    block: str,
    keywords: Mapping[str, Any],
    classified: dict[str, frozenset[str]],
    verb: str,
) -> None:
    """Raise if a keyword of ``keywords`` is missing from ``classified``.

    Raises:
        ValueError: If a keyword is in neither ``classified[block]`` nor
            :data:`ROUTE_CONDITIONAL`.
    """
    allowed = classified.get(block, frozenset()) | ROUTE_CONDITIONAL.get(block, frozenset())
    undeclared = sorted(set(keywords) - allowed)
    if undeclared:
        raise ValueError(
            f"the route {verb} {block} {', '.join(undeclared)}, which "
            f"aiida_koopmans.owned_keywords does not classify. Add each to OWNED "
            f"(and to koopmans' reason map, which drops it from the input file) or "
            f"to SEEDED (which leaves it settable)."
        )
