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


class VariationalOrbital(TypedDict):
    """Structured record for a single variational orbital.

    Carries spin / per-spin 1-indexed position / filled-vs-empty plus
    its place in any grouping (``group_id``, ``representative``).
    Defined as a :class:`TypedDict` rather than a ``dataclass`` so
    instances are plain dicts at runtime — ``list[VariationalOrbital]``
    survives ``aiida-workgraph``'s storage path (``orm.List`` →
    ``clean_value``) because ``Mapping`` instances are recursed into,
    with each leaf landing on a primitive (``SpinChannel`` is a
    ``str``-Enum, the rest are ``int`` / ``bool``).

    Access fields by string keys: ``o["spin"]``, ``o["index"]``,
    ``o["filled"]``, ``o["group_id"]``, ``o["representative"]``. The
    key names *are* the structural information — they're stable and
    never parsed back into parts, unlike a flat string label like
    ``"up_orb_5"``. Use :func:`map_key_for` when a string label is
    needed (only at the ``aiida-workgraph`` ``Map`` zone boundary,
    where iteration handles require strings).

    On AiiDA round-trip the ``spin`` value comes back as a plain
    ``str`` rather than a :class:`SpinChannel` enum — ``SpinChannel``
    inherits from ``str`` so ``o["spin"] == SpinChannel.UP`` continues
    to work, but ``o["spin"] is SpinChannel.UP`` does not. Prefer
    ``==`` everywhere (this is also the project-wide
    ``feedback_taggedvalue_is_comparison`` rule for ``@task.graph``
    bodies).
    """

    spin: SpinChannel
    index: int  # 1-indexed per-spin band position
    filled: bool
    group_id: int
    representative: bool


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
