"""Shared types for the koopmans AiiDA workgraph layer.

Cross-module data shapes live here so workgraph builders, the kcp.x
CalcJob, parser, and tests can all import a single canonical definition.
"""

from __future__ import annotations

from enum import Enum
from typing import NotRequired, TypedDict

from aiida_quantumespresso.common.types import SpinType as SpinType
from aiida_wannier90_workflows.common.types import WannierProjectionType


class Correction(str, Enum):
    """The Koopmans correction (functional) the workflow applies.

    Members:

    * ``KI``: Koopmans-Integral correction (the default).
    * ``KIPZ``: Koopmans-Integral with Perdew-Zunger self-interaction
      correction on the variational orbitals — different sub-step
      parameters; see ``aiida_koopmans/workgraphs/kcp.py``.
    * ``PKIPZ``: perturbative KIPZ (trial KI, KIPZ correction applied
      post-hoc). Deferred — accepted at the type level but not yet
      wired through the dispatcher.
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


class VariationalOrbitalType(str, Enum):
    """Initial variational orbitals to use for the trial KI / KIPZ run.

    * ``PZ``: PZ-initialised variational orbitals.
    * ``KOHN_SHAM``: KS orbitals from the DFT init reused as
      variational (the currently-supported path; produces a
      KS-as-variational overlay so the trial KI's ``evc0N.dat`` is the
      DFT ``evcN.dat``).
    * ``MLWFS``: maximally-localised Wannier functions
      (Wannier90-based; deferred).
    * ``PROJWFS``: projected Wannier functions (deferred).
    """

    PZ = "pz"
    KOHN_SHAM = "kohn-sham"
    MLWFS = "mlwfs"
    PROJWFS = "projwfs"


class SpinChannel(str, Enum):
    """Spin channel index used as a dict key in per-spin data structures.

    Use ``SpinChannel.NONE`` for ``nspin == 1`` calculations (no spin
    polarisation, single channel).
    """

    NONE = "none"
    UP = "up"
    DOWN = "down"
    SPINOR = "spinor"

    @property
    def index(self) -> int:
        """Spin index into a stacked-by-spin array (axis-0 of ``(nspin, ...)``).

        ``NONE`` and ``UP`` both live at index 0 (kcp.x's nspin=1 file layout
        and the up channel of nspin=2 share the leading axis); ``DOWN`` is 1.
        ``SPINOR`` (noncollinear, nspin=4) has a single band index — 0.
        """
        return 1 if self is SpinChannel.DOWN else 0


class VariationalOrbital(TypedDict):
    """Structured record for a single variational orbital.

    Carries spin / per-spin 1-indexed position / filled-vs-empty plus
    its place in any grouping (``group_id``, ``representative``). The
    key names *are* the structural information — stable and never
    parsed back into parts, unlike a flat string label like
    ``"up_orb_5"``; use :func:`map_key_for` when a string label is
    needed (only at the ``aiida-workgraph`` ``Map`` zone boundary).

    On AiiDA round-trip ``spin`` comes back as a plain ``str`` rather
    than a :class:`SpinChannel` enum, so compare with ``==`` not
    ``is`` (``SpinChannel`` subclasses ``str``, so
    ``o["spin"] == SpinChannel.UP`` holds but ``is`` does not).
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


class OrbitalDict(TypedDict):
    """A single resolved Wannier orbital as a plain dict.

    A typed view over the dict AiiDA produces via
    ``Orbital.get_orbital_dict()`` for the ``core.realhydrogen`` orbital
    type (the one ``aiida-wannier90``'s ``OrbitalData`` stores);
    :func:`orbital_data_to_dicts` / :func:`dicts_to_orbital_data`
    round-trip it losslessly against ``OrbitalData``. Convert to
    ``OrbitalData`` only at the ``Wannier90WorkChain`` input boundary.

    The keys mirror ``aiida.tools.data.orbital.realhydrogen``'s fields
    exactly; ``test_projection_blocks`` asserts parity so an upstream
    schema change is caught rather than silently drifting.
    """

    _orbital_type: str
    position: list[float]
    angular_momentum: int
    magnetic_number: int
    radial_nodes: int
    kind_name: str
    spin: int
    x_orientation: list[float] | None
    z_orientation: list[float] | None
    spin_orientation: list[float] | None
    diffusivity: float | None


class _ProjectionBlockBase(TypedDict):
    """Band-bookkeeping shared by every projection block (any source).

    The per-block bookkeeping that is independent of *how* the Wannier
    functions are obtained: the block's label, spin, the counts
    (``num_wann`` is the common denominator across all projection
    sources), and which bands it covers. ``projection_type`` is the
    discriminator (a real :class:`WannierProjectionType`, registered for
    AiiDA serialization via the ``aiida.data`` entry points).
    """

    label: str
    spin: SpinChannel
    num_wann: int
    num_bands: int
    include_bands: list[int]
    exclude_bands: NotRequired[list[int] | None]
    projection_type: WannierProjectionType


class ExplicitProjectionBlock(_ProjectionBlockBase):
    """A block whose Wannier functions come from explicit projections.

    ``projection_type`` is ``WannierProjectionType.ANALYTIC``.
    ``projections`` is the resolved ``list[OrbitalDict]`` and is
    required (``num_wann == len(projections)``).
    """

    projections: list[OrbitalDict]


class AutomaticProjectionBlock(_ProjectionBlockBase):
    """A block whose Wannier functions are found automatically.

    For ``projection_type`` in ``{SCDM, ATOMIC_PROJECTORS_QE,
    ATOMIC_PROJECTORS_EXTERNAL, RANDOM}`` there are no explicit projection
    orbitals -- the block is defined by ``num_wann`` (plus the frozen /
    disentanglement windows carried elsewhere).
    """


# Analytic blocks carry ``projections``; automatic blocks do not, so
# ``"projections" in block`` narrows the union to the explicit arm.
ProjectionBlock = ExplicitProjectionBlock | AutomaticProjectionBlock


class MergeGroup(TypedDict):
    """A set of :class:`ProjectionBlock` instances merged into one kcp.x manifold.

    Blocks that share a filling (occupied vs empty) and spin are merged
    together (their per-block ``evcw`` wavefunctions are concatenated by
    ``merge_evc.x``) into a single ``evc_occupied`` / ``evc0_empty`` file
    that seeds the supercell kcp.x run.

    * ``filled``: ``True`` for the occupied manifold, ``False`` for empty.
    * ``spin``: the shared spin channel (``SpinChannel.NONE`` for nspin=1).
    * ``blocks``: the member blocks, in band order.
    """

    filled: bool
    spin: SpinChannel
    blocks: list[ProjectionBlock]


def block_w90_kwargs(block: ProjectionBlock) -> dict:
    """Return the Wannier90 input keywords for a single block.

    The per-block ``num_wann`` / ``num_bands`` / ``exclude_bands`` (and
    ``spin`` when the block is spin-resolved) that distinguish one block's
    Wannier90
    cycle from another's. ``projections`` is included only for an
    :class:`ExplicitProjectionBlock`; automatic blocks rely on
    ``projection_type`` instead. ``exclude_bands`` is omitted when the
    block excludes nothing.
    """
    kwargs: dict = {
        "num_wann": block["num_wann"],
        "num_bands": block["num_bands"],
    }
    exclude = block.get("exclude_bands")
    if exclude is not None:
        kwargs["exclude_bands"] = exclude
    if block["spin"] != SpinChannel.NONE:
        kwargs["spin"] = SpinChannel(block["spin"]).value
    if "projections" in block:
        kwargs["projections"] = block["projections"]
    return kwargs


def group_blocks_to_merge(
    blocks: list[ProjectionBlock],
    num_occ_bands: dict[SpinChannel, int],
) -> list[MergeGroup]:
    """Group blocks into occupied / empty manifolds per spin.

    A block is *occupied* when all of its ``include_bands`` lie at or
    below that spin channel's
    occupied-band count, and *empty* when they all lie above it; a block
    that straddles the boundary is an error (the projections must be split
    so each block is purely occupied or purely empty).

    ``num_occ_bands`` maps each spin channel to its number of occupied
    bands. For ``nspin == 1`` use the single key ``SpinChannel.NONE``.

    Returns one :class:`MergeGroup` per ``(filled, spin)`` that has
    members, preserving the order in which blocks are first encountered so
    the downstream ``merge_evc.x`` concatenation is deterministic.
    """
    groups: list[MergeGroup] = []
    index: dict[tuple[bool, SpinChannel], MergeGroup] = {}
    for block in blocks:
        spin = SpinChannel(block["spin"])
        if spin not in num_occ_bands:
            raise KeyError(
                f"`num_occ_bands` has no entry for spin {spin!r}; provide one "
                f"occupied-band count per spin channel (use SpinChannel.NONE "
                f"for nspin==1)."
            )
        n_occ = num_occ_bands[spin]
        include = block["include_bands"]
        if max(include) <= n_occ:
            filled = True
        elif min(include) > n_occ:
            filled = False
        else:
            raise ValueError(
                f"Block {block['label']!r} spans both the occupied and empty "
                f"manifolds (include_bands={include}, n_occ={n_occ}). Split the "
                f"projections so each block is purely occupied or purely empty."
            )
        key = (filled, spin)
        group = index.get(key)
        if group is None:
            group = MergeGroup(filled=filled, spin=spin, blocks=[])
            index[key] = group
            groups.append(group)
        group["blocks"].append(block)
    return groups


def merge_dest_filename(filled: bool, spin_index: int) -> str:
    """kcp.x-side filename for a merged manifold wavefunction.

    The supercell kcp.x run reads its initial variational orbitals from
    ``evc_occupied{n}.dat`` (occupied manifold) or ``evc0_empty{n}.dat``
    (empty manifold), where ``n`` is the 1-based kcp.x spin index
    (1 = up / unpolarized, 2 = down).
    """
    if spin_index not in (1, 2):
        raise ValueError(f"spin_index must be 1 or 2, got {spin_index!r}")
    if filled:
        return f"evc_occupied{spin_index}.dat"
    return f"evc0_empty{spin_index}.dat"
