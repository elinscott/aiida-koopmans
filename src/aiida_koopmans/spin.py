"""Spin-channel vocabulary shared by every stream."""

from __future__ import annotations

from enum import Enum


class SpinChannel(str, Enum):
    """Spin channel index used as a dict key in per-spin data structures.

    Use ``SpinChannel.NONE`` for ``nspin == 1`` calculations (no spin
    polarisation, single channel).
    """

    # Declaration order is the canonical channel walk order: iterating
    # the enum IS the ordering authority for representative stamping and
    # orbital emission. The position of NONE relative to UP/DOWN is
    # immaterial — the channels are mutually exclusive spin regimes, so
    # no calculation ever walks NONE alongside UP/DOWN; what matters is
    # only UP before DOWN.
    NONE = "none"
    UP = "up"
    DOWN = "down"
    SPINOR = "spinor"

    @property
    def axis(self) -> int:
        """Spin index into a stacked-by-spin array (axis-0 of ``(nspin, ...)``).

        ``NONE`` and ``UP`` both live at index 0 (kcp.x's nspin=1 file layout
        and the up channel of nspin=2 share the leading axis); ``DOWN`` is 1.
        ``SPINOR`` (noncollinear, nspin=4) has a single band index — 0.
        """
        return 1 if self is SpinChannel.DOWN else 0
