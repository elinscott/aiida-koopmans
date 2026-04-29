"""Shared types for the koopmans AiiDA workgraph layer.

Cross-module data shapes live here so workgraph builders, the kcp.x
CalcJob, parser, and tests can all import a single canonical definition.
"""

from __future__ import annotations

from enum import Enum
from typing import TypedDict


class SpinChannel(str, Enum):
    """Spin channel index used as a dict key in per-spin data structures.

    Values match the legacy koopmans convention from
    ``koopmans/src/koopmans/utils/_spin.py``
    (``SpinType = Literal[None, "up", "down", "spinor"]``). String-valued
    so AiiDA / JSON round-trips preserve the value (a JSON-deserialised
    ``"up"`` compares equal to ``SpinChannel.UP``).

    Use ``SpinChannel.NONE`` for ``nspin == 1`` calculations (no spin
    polarisation, single channel).
    """

    NONE = "none"
    UP = "up"
    DOWN = "down"

    @property
    def index(self) -> int:
        """Spin index into a stacked-by-spin array (axis-0 of ``(nspin, ...)``).

        ``NONE`` and ``UP`` both live at index 0 (kcp.x's nspin=1 file layout
        and the up channel of nspin=2 share the leading axis); ``DOWN`` is 1.
        """
        return 1 if self is SpinChannel.DOWN else 0


class AlphaScreening(TypedDict):
    """Per-spin per-orbital screening parameters for the kcp.x ``file_alpharef``.

    Both ``filled`` and ``empty`` are dicts keyed by spin channel; each
    value is a list of one ``alpha`` per per-spin orbital, 1-indexed by
    list position.

    For ``nspin == 2``: keys are ``SpinChannel.UP`` and ``SpinChannel.DOWN``; the
    ``KcpCalculation`` flattens them into the kcp.x file format on write
    (block-spin: all ``SpinChannel.UP`` entries first, then ``SpinChannel.DOWN``).

    For ``nspin == 1``: the only key is ``SpinChannel.NONE``.
    """

    filled: dict[SpinChannel, list[float]]
    empty: dict[SpinChannel, list[float]]
