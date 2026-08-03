"""Variational-orbital vocabulary.

Pure data shapes and labels; the partition and enumeration operators
that work on them live in
:mod:`aiida_koopmans.workgraphs.variational_orbitals`. This module must
stay importable without an AiiDA profile or configuration (the koopmans
input-file schema imports it), so nothing here may import
``aiida_workgraph`` or ``aiida_pythonjob``.
"""

from __future__ import annotations

from enum import Enum
from typing import NotRequired, TypedDict

from aiida_koopmans.spin import SpinChannel


class VariationalOrbitalType(str, Enum):
    """Initial variational orbitals to use for the trial KI / KIPZ run.

    * ``PZ``: PZ-initialised variational orbitals.
    * ``KOHN_SHAM``: KS orbitals from the DFT init reused as
      variational (the currently-supported path; produces a
      KS-as-variational overlay so the trial KI's ``evc0N.dat`` is the
      DFT ``evcN.dat``).
    * ``MLWFS``: maximally-localised Wannier functions
      (Wannier90-based; not yet implemented in the kcp.x stream).
    * ``PROJWFS``: projected Wannier functions (not yet implemented in
      the kcp.x stream).
    """

    PZ = "pz"
    KOHN_SHAM = "kohn-sham"
    MLWFS = "mlwfs"
    PROJWFS = "projwfs"


class VariationalOrbital(TypedDict):
    """Structured record for a single variational orbital.

    Carries spin / per-spin 1-based band index / filled-vs-empty plus
    its place in any grouping (``group_id``, ``representative``). The
    key names *are* the structural information — stable and never
    parsed back into parts, unlike a flat string label like
    ``"up_orb_5"``; use :func:`map_key_for` when a string label is
    needed (only at the ``aiida-workgraph`` ``Map`` zone boundary).

    On AiiDA round-trip ``spin`` comes back as a plain ``str`` rather
    than a :class:`SpinChannel` enum, so compare with ``==`` not
    ``is`` (``SpinChannel`` subclasses ``str``, so
    ``o["spin"] == SpinChannel.UP`` holds but ``is`` does not).

    ``manifold`` is the label of the projection block the orbital's
    Wannier function belongs to, set by callers that build orbitals
    from projection blocks. Workflows without blocks have no such
    label, so it is ``NotRequired``; the partition operator
    :func:`~aiida_koopmans.workgraphs.variational_orbitals.refine_by_key`
    demands it on every orbital before splitting on it.
    """

    spin: SpinChannel
    index: int  # 1-based per-spin band index
    filled: bool
    group_id: int
    representative: bool
    manifold: NotRequired[str]  # projection-block label; set by block-building callers


def map_key_for(orbital: VariationalOrbital) -> str:
    """Stable string label used as a key in ``Map`` zone iteration dicts.

    ``aiida-workgraph``'s ``Map`` zone iterates over a dict and uses
    the key as the iteration handle / resulting calc-node link label.
    Strings are required there. This is the *only* place where an
    orbital's identity is encoded as a string; the round-trip back to
    structured form goes through :func:`enumerate_variational_orbitals`
    at the gather boundary, never by parsing the string.

    Uses ``SpinChannel(...).value`` rather than ``str(spin)`` because
    Python 3.12+ changed ``str()`` on ``str``-Enums to return
    ``"<ClassName>.<member>"`` for non-trivial subclasses — explicit
    ``.value`` access stays "up" / "down" / "none" regardless of
    Python version, and also normalises post-AiiDA-round-trip values
    where ``spin`` arrives as a plain ``str`` rather than the enum.
    """
    spin = SpinChannel(orbital["spin"])
    tag = "" if spin is SpinChannel.NONE else f"{spin.value}_"
    return f"{tag}orb_{orbital['index']}"
