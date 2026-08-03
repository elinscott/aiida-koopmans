"""Functional-level vocabulary of the Koopmans workflows."""

from __future__ import annotations

from enum import Enum


class Correction(str, Enum):
    """The Koopmans correction (functional) the workflow applies.

    Members:

    * ``KI``: Koopmans-Integral correction (the default).
    * ``KIPZ``: Koopmans-Integral with Perdew-Zunger self-interaction
      correction on the variational orbitals — different sub-step
      parameters; see ``aiida_koopmans/workgraphs/kcp.py``.
    * ``PKIPZ``: perturbative KIPZ (trial KI, KIPZ correction applied
      post-hoc). Not yet implemented — accepted at the type level but
      not wired through the dispatcher.
    * ``PZ``: plain Perdew-Zunger orbital-dependent functional —
      strictly not a "Koopmans correction" but routed through the
      same orbital-dependent screening machinery
      (:func:`aiida_koopmans.workgraphs.kcp._build_orbdep_parameters`),
      e.g. the empty-orbital ``pz_print`` sub-step of a KI workflow.
    * ``NONE``: no Koopmans correction (plain DFT only).
    * ``ALL``: run KI / KIPZ / PKIPZ together (user-facing workflow
      control).
    """

    KI = "ki"
    KIPZ = "kipz"
    PKIPZ = "pkipz"
    PZ = "pz"
    NONE = "none"
    ALL = "all"
