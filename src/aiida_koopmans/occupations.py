"""Per-spin-channel electron and band accounting shared by the QE streams."""

from __future__ import annotations

from aiida_koopmans.types import SpinChannel


def default_channel_nocc(spin_channel: SpinChannel, nelec: int) -> int:
    """Occupied-band count of a channel when the caller supplies none.

    Spinor bands are singly occupied (``nocc = nelec``); the unpolarized
    channel holds electron pairs. Collinear channels have no default — their
    occupations depend on the magnetization, which only the caller knows.
    """
    if spin_channel in (SpinChannel.UP, SpinChannel.DOWN):
        raise ValueError(
            f"spin_channel={spin_channel.value!r} needs an explicit per-channel "
            "nocc (derived from the electron count and the magnetization)."
        )
    if spin_channel == SpinChannel.SPINOR:
        return nelec
    if nelec % 2:
        raise ValueError(
            f"Odd electron count ({nelec}) requires spin='collinear', which "
            "derives per-channel occupations from the magnetization."
        )
    return nelec // 2
