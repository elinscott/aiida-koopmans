"""WorkGraph-based workflows for koopmans calculations."""

from __future__ import annotations

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
