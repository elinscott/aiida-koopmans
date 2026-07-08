"""aiida-workgraph builders for kcp.x.

Wraps :class:`~aiida_koopmans.calculations.kcp.KcpCalculation` as a task and
composes it into higher-level workflows.

**Current scope.** Two routes are implemented: the molecular Kohn-Sham-init path
(KI/KIPZ + DSCF) and the periodic Wannier-init path (``init_orbitals``
``'mlwfs'`` / ``'projwfs'``: wannierize → fold-to-supercell → Γ-point
supercell kcp.x; see ``mlwf_init.py``). Unsupported combinations raise
``NotImplementedError`` at build time with a clear message.

The MVP ``KoopmansDSCFWorkflow`` executes **two** kcp.x calls:

1. DFT initialization (``do_orbdep=False``, nspin=2, from scratch)
2. KI final (``do_orbdep=True``, ``which_orbdep='nki'``, restart from step 1,
   initial alphas = ``initial_alpha`` for every orbital)

The legacy implementation for the same inputs executes 20 kcp.x calls
including spin-symmetrization (7), a trial KI pass (1), a Delta SCF loop to
compute the alphas (12), and a final KI (1). Porting the Delta SCF alpha loop
and spin-symmetrization is deferred to later phases — the code below is
structured so those extensions can slot in as additional ``@task.graph``
helpers without reshaping the public ``KoopmansDSCFWorkflow`` signature.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Annotated, Any, TypedDict, cast

import numpy as np
from aiida import orm
from aiida_pseudo.data.pseudo.upf import UpfData
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_workgraph import dynamic, task

from aiida_koopmans.calculations.kcp import KcpCalculation
from aiida_koopmans.types import (
    AlphaScreening,
    Correction,
    SpinChannel,
    VariationalOrbital,
    VariationalOrbitalType,
    map_key_for,
)
from aiida_koopmans.utils import (
    count_electrons_task,
    resolve_pseudo_family_task,
)
from aiida_koopmans.workgraphs import Codes
from aiida_koopmans.workgraphs.convert_spin import convert_spin1_to_spin2
from aiida_koopmans.workgraphs.variational_orbitals import (
    assign_orbital_groups,
    expand_alphas_by_group,
    extract_self_hartree_from_kcp,
)

# ----------------------------------------------------------------------
# Output / override typing
# ----------------------------------------------------------------------
#
# Annotations declare what consumer pyfunctions see *after* aiida-pythonjob's
# auto-deserialization: ``orm.Dict → dict``, single-key ``orm.ArrayData →
# np.ndarray``. Lambdas/bare-lambdas come from the parser as a single
# stacked ``(nspin, n, n)`` matrix (see ``KcpParser._parse_lambdas``); index
# axis-0 by ``SpinChannel.axis``. ``remote_folder`` stays as
# ``orm.RemoteData`` because downstream ``parent_folder`` sockets take the
# node, not its payload.


class DFTCPOutputs(TypedDict):
    """Outputs of a single kcp.x DFT-only run."""

    parameters: dict
    eigenvalues: np.ndarray
    remote_folder: orm.RemoteData


class KIFinalOutputs(TypedDict):
    """Outputs of the final KI run (the application of the converged alphas)."""

    parameters: dict
    eigenvalues: np.ndarray
    lambdas: np.ndarray
    bare_lambdas: np.ndarray
    remote_folder: orm.RemoteData


class KoopmansDSCFOutputs(TypedDict):
    """Outputs of a full KI-DSCF workflow.

    :class:`KIFinalOutputs` plus ``alphas`` — the converged per-orbital
    screening parameters the final KI consumed (in the
    :class:`~aiida_koopmans.types.AlphaScreening` shape). Exposed at the
    workflow level so consumers (e.g. the ML trajectory workflow's
    training targets) read them directly instead of walking provenance.
    A separate ``TypedDict`` from :class:`KIFinalOutputs` because
    ``alphas`` is an *input* of the final KI — :func:`RunFinalKI` cannot
    echo a graph input as an output, so the field is wired at the outer
    workflow level from the screening step's outputs.
    """

    parameters: dict
    eigenvalues: np.ndarray
    lambdas: np.ndarray
    bare_lambdas: np.ndarray
    remote_folder: orm.RemoteData
    alphas: AlphaScreening


@dataclass(frozen=True)
class KcpBaseInputs:
    """Cell-, basis-, and electron-count inputs shared by every kcp.x step.

    Each parameter builder takes one ``KcpBaseInputs`` (built once per
    workgraph from ``structure`` + electron-count outputs) plus its
    step-specific kwargs. ``nbnd`` is intentionally *not* here —
    DFT/KI/PZ steps need it but alpha-step (dft_n±1) builders strip it,
    so it stays a step-level kwarg.

    A frozen ``dataclass`` rather than a ``TypedDict``: aiida-workgraph
    routes dataclass-typed sockets through ``structured_to_dict`` →
    ``dataclasses.asdict`` (which preserves ``None`` fields) and
    reconstructs via ``cls(**value)`` on the receiving side. Plain
    ``dict`` / ``TypedDict`` sockets, by contrast, silently strip
    ``None``-valued entries in transit (e.g. ``tot_magnetization=None``
    for a closed-shell system).
    """

    ecutwfc: float
    ecutrho: float
    nspin: int
    nelec: int
    ntyp: int
    mt_correction: bool
    nelup: int | None = None
    neldw: int | None = None
    tot_magnetization: int | None = None


class KcpNamelistOverrides(TypedDict, total=False):
    """Override shape for a single kcp.x input rendering.

    Each key corresponds to a Fortran namelist. Values are merged on top of
    the default kcp.x namelist values via
    ``aiida_quantumespresso.workflows.protocols.utils.recursive_merge``.
    """

    CONTROL: dict[str, Any]
    SYSTEM: dict[str, Any]
    ELECTRONS: dict[str, Any]
    IONS: dict[str, Any]
    CELL: dict[str, Any]
    EE: dict[str, Any]
    NKSIC: dict[str, Any]


class KoopmansDSCFOverrides(TypedDict, total=False):
    """Per-step overrides for ``KoopmansDSCFWorkflow``.

    ``ki`` is reused for both the trial KI and the final KI.
    The four DSCF sub-step keys override individual orbital sub-runs:
    ``dft_n_minus_1`` for filled orbitals; ``dft_n_plus_1_dummy``,
    ``pz_print``, ``dft_n_plus_1`` for the empty-orbital triplet.
    """

    dft: KcpNamelistOverrides
    ki: KcpNamelistOverrides
    dft_n_minus_1: KcpNamelistOverrides
    dft_n_plus_1_dummy: KcpNamelistOverrides
    pz_print: KcpNamelistOverrides
    dft_n_plus_1: KcpNamelistOverrides


class OrbitalDeltaSCFOutputs(TypedDict):
    """Outputs of one Delta-SCF orbital sub-run.

    ``alpha`` is the new screening parameter for the (spin, band) pair this
    task targets; ``error`` is ``|dE - lambda_a|``, the convergence
    indicator used by the iteration loop's stopping criterion.
    """

    alpha: float
    error: float


class _PerOrbitalAlphaOutputs(TypedDict):
    """Gathered outputs of the per-orbital fan-out, packed into per-spin lists.

    ``alphas`` and ``errors`` share the same per-spin / filled-vs-empty
    layout (see :class:`AlphaScreening`), making the ``alphas`` field
    drop-in for the kcp.x ``alphas`` socket on the final KI step.
    """

    alphas: AlphaScreening
    errors: AlphaScreening


class ScreeningIterationOutputs(TypedDict):
    """Outputs of one alpha-refinement iteration (trial KI + per-orbital DSCF).

    Used to thread the next iteration's inputs through the recursive
    :func:`RefineScreeningParameters`:

    * ``alphas`` — gathered per-orbital screening parameters; becomes the
      next iteration's trial-KI ``alphas`` input.
    * ``errors`` — gathered ``|dE - lambda|`` per orbital; retained for
      diagnostics / convergence reporting.
    * ``trial_remote`` — the trial KI's ``remote_folder``; becomes the
      next iteration's ``parent_folder`` (and, after the loop, the final
      KI's parent).
    * ``max_error`` — convergence indicator; the loop terminates when
      this falls below the ``1e-3 eV`` threshold.
    """

    alphas: AlphaScreening
    errors: AlphaScreening
    trial_remote: orm.RemoteData
    max_error: float


class ScreeningParametersOutputs(TypedDict):
    """Outputs of ``ComputeScreeningParameters``: the converged screening alpha's.

    Semantically distinct from :class:`KoopmansDSCFOutputs` — the latter
    is the *application* of the alpha's (the final KI calculation), the
    former is what's needed to *run* that application.

    * ``alphas`` — the per-orbital screening parameters from the last
      alpha-refinement iteration, in the shape :class:`AlphaScreening`
      that ``KcpCalculation`` accepts at its ``alphas`` input.
    * ``trial_remote`` — the last iteration's trial-KI ``remote_folder``;
      becomes the final KI's ``parent_folder``, so the final KI inherits
      the converged variational orbital basis.
    """

    alphas: AlphaScreening
    trial_remote: orm.RemoteData


# ----------------------------------------------------------------------
# Raw CalcJob as a workgraph task
# ----------------------------------------------------------------------

KcpStep = task(KcpCalculation)


# ----------------------------------------------------------------------
# Pure-Python alpha computation. ``@task.pyfunction`` runs in-process and
# auto-deserialises Node inputs to native Python types so the body stays
# AiiDA-agnostic. Provenance is preserved via the resulting ``PyFunction``
# process node.
# ----------------------------------------------------------------------


@task(outputs=["alpha", "error"])
def compute_alpha_from_dscf(
    *,
    trial_output_parameters: dict,
    perturbed_output_parameters: dict,
    trial_lambdas: np.ndarray,
    trial_bare_lambdas: np.ndarray,
    spin_channel: SpinChannel,
    band_index: int,
    alpha_guess: float,
    filled: bool,
    mp_correction: bool = False,
    eps_inf: float = 1.0,
) -> dict:
    """Compute the new alpha for one orbital from its Delta-SCF perturbed run.

    Implements equation 10 of Nguyen et al. (2018) 10.1103/PhysRevX.8.021051::

        alpha_new = alpha_guess * (dE - lambda_0) / (lambda_a - lambda_0)

    where:

    - ``dE = E_trial - E_dft_n-1`` for filled orbitals,
      ``dE = E_dft_n+1 - E_trial`` for empty orbitals;
    - ``lambda_a`` is the diagonal element of the trial KI's
      orbital-dependent Hamiltonian at ``(band_index, band_index)``;
    - ``lambda_0`` is the same diagonal element of the **bare** Hamiltonian.

    With ``mp_correction=True`` (periodic supercells; legacy default for
    the DSCF method on periodic systems) the Makov-Payne image-interaction
    energies reported by the perturbed N±1 run are subtracted from ``dE``
    scaled by the macroscopic dielectric constant — legacy
    ``_koopmans_dscf.py:932-942``: ``dE -= sign(charge) * (mp1 + mp2) /
    eps_inf`` where ``sign(charge)`` is ``+1`` for a filled orbital (an
    electron removed) and ``-1`` for an empty one (an electron added), and
    ``mp2`` is used only when the run reports it.

    Both energies and lambdas are in eV (the parser converts from Hartree),
    so the units cancel on division. ``error = |dE - lambda_a|`` is the
    convergence indicator the refinement loop monitors. The lambda arrays are
    stacked ``(nspin, n, n)``; ``spin_channel.axis`` selects the spin axis.
    """
    trial_e = trial_output_parameters["energy"]
    perturbed_e = perturbed_output_parameters["energy"]
    spin = spin_channel.axis
    lambda_a = float(trial_lambdas[spin, band_index, band_index].real)
    lambda_0 = float(trial_bare_lambdas[spin, band_index, band_index].real)
    dE = trial_e - perturbed_e if filled else perturbed_e - trial_e  # noqa: N806
    if mp_correction:
        mp1 = perturbed_output_parameters.get("mp1_energy")
        mp2 = perturbed_output_parameters.get("mp2_energy")
        if mp1 is None:
            raise ValueError("Could not find 1st order Makov-Payne energy")
        mp_energy = mp1 if mp2 is None else mp1 + mp2
        sign_of_charge = 1 if filled else -1
        dE -= sign_of_charge * mp_energy / eps_inf  # noqa: N806
    alpha_new = alpha_guess * (dE - lambda_0) / (lambda_a - lambda_0)
    error = abs(dE - lambda_a)
    return {"alpha": alpha_new, "error": error}


@task
def assemble_alpha_screening(
    *,
    orbitals: list[VariationalOrbital],
    filled_alphas: dict | None = None,
    filled_errors: dict | None = None,
    empty_alphas: dict | None = None,
    empty_errors: dict | None = None,
) -> _PerOrbitalAlphaOutputs:
    """Pack per-orbital fan-out outputs into the :class:`AlphaScreening` shape.

    Inputs are flat dicts keyed by :func:`map_key_for` strings — the
    same labels the per-orbital for-loop gather uses — paired with a
    ``list[VariationalOrbital]`` from :func:`assign_orbital_groups`
    that carries each orbital's structured fields (``spin`` /
    ``index`` / ``filled``). The packer looks up each input value by
    ``map_key_for`` and groups by spin using the structured fields
    directly — no string parsing.

    The closed-shell up→down mirror (when kcp.x needs both spin slots
    in ``file_alpharef``) lives in :meth:`KcpCalculation._write_alpha_files`,
    not here.

    Returns a ``{"alphas": AlphaScreening, "errors": AlphaScreening}``
    pair: the ``alphas`` field plugs straight into the kcp.x ``alphas``
    socket of the final KI step.
    """
    # A branch with zero orbitals passes no gather dict (``None``);
    # treat that as the empty dict.
    filled_alphas = filled_alphas or {}
    filled_errors = filled_errors or {}
    empty_alphas = empty_alphas or {}
    empty_errors = empty_errors or {}

    def _pack(flat: dict, *, filled: bool) -> dict[SpinChannel, list[float]]:
        by_spin: dict[SpinChannel, list[tuple[int, float]]] = {}
        for o in orbitals:
            if o["filled"] != filled:
                continue
            key = map_key_for(o)
            if key not in flat:
                continue
            # ``spin`` round-trips through AiiDA storage as plain ``str``;
            # normalise back to the enum so per-spin dicts are keyed
            # uniformly.
            spin = SpinChannel(o["spin"])
            by_spin.setdefault(spin, []).append((o["index"], float(flat[key])))
        out: dict[SpinChannel, list[float]] = {}
        for spin, items in by_spin.items():
            items.sort(key=lambda t: t[0])
            out[spin] = [v for _, v in items]
        return out

    filled_packed = _pack(filled_alphas, filled=True)
    empty_packed = _pack(empty_alphas, filled=False)
    filled_err_packed = _pack(filled_errors, filled=True)
    empty_err_packed = _pack(empty_errors, filled=False)

    alphas: AlphaScreening = {"filled": filled_packed, "empty": empty_packed}
    errors: AlphaScreening = {"filled": filled_err_packed, "empty": empty_err_packed}
    return {"alphas": alphas, "errors": errors}


@task
def max_alpha_error(filled_errors: dict, empty_errors: dict):
    """Convergence indicator for one DSCF iteration.

    Returns ``max |dE - lambda|`` across every per-orbital error in both
    branches. The alpha-refinement loop in ``ComputeScreeningParameters`` stops
    when this falls below ``1e-3 eV``. ``filled_errors`` / ``empty_errors`` are
    the per-spin dicts produced by :func:`assemble_alpha_screening` (each
    keyed by ``SpinChannel`` value mapping to a per-orbital list).
    """
    values: list[float] = []
    for per_spin in (filled_errors, empty_errors):
        for spin_list in per_spin.values():
            values.extend(abs(v) for v in spin_list)
    return max(values) if values else 0.0


# ----------------------------------------------------------------------
# Uniform alpha generator. Runs as a plain ``@task`` so all per-channel
# list arithmetic happens inside a process node, never on raw sockets
# inside a ``@task.graph`` body.
# ----------------------------------------------------------------------


@task
def generate_alphas(
    alpha_guess: float,
    nbnd: int,
    nelup: int,
    neldw: int,
    spin_polarized: bool = False,
) -> AlphaScreening:
    """Build an :class:`AlphaScreening` of uniform ``alpha_guess`` values.

    Used both for the trial-KI uniform alphas and the empty-orbital
    ``pz_print`` alphas. Mirrors the channel-emission rule of
    :func:`build_filled_iter_source`:

    * ``spin_polarized=False`` → emit a single representative channel
      keyed by :attr:`SpinChannel.NONE`. The mirror onto kcp.x's per-spin
      ``file_alpharef`` happens later in
      :meth:`KcpCalculation._write_alpha_files`.
    * ``spin_polarized=True`` → emit both UP and DOWN at their actual
      per-spin sizes (open-shell systems with ``nelup != neldw`` have
      different per-spin empty-manifold sizes — e.g. O2 with
      ``nelup=7, neldw=5, nbnd=8`` has 1 UP empty and 3 DOWN empties).
    """
    n_empty_up = max(0, nbnd - nelup)
    n_empty_dw = max(0, nbnd - neldw)
    if not spin_polarized:
        # Closed-shell: nelup == neldw == nelec/2, so picking either
        # arm as the representative works.
        return {
            "filled": {SpinChannel.NONE: [alpha_guess] * nelup},
            "empty": {SpinChannel.NONE: [alpha_guess] * n_empty_up},
        }
    return {
        "filled": {
            SpinChannel.UP: [alpha_guess] * nelup,
            SpinChannel.DOWN: [alpha_guess] * neldw,
        },
        "empty": {
            SpinChannel.UP: [alpha_guess] * n_empty_up,
            SpinChannel.DOWN: [alpha_guess] * n_empty_dw,
        },
    }


class FilledIterItem(TypedDict):
    """One per-orbital work item for the *filled* Delta-SCF fan-out.

    Built by :func:`build_filled_iter_source` and consumed field-by-field
    by :func:`ComputeOrbitalScreeningParameters`, which scatters one
    :func:`ComputeFilledOrbitalScreeningParameter` per representative filled orbital.
    ``spin_channel`` / ``band_index`` are in the physical frame (they
    index the trial-KI lambda matrices); ``fixed_band`` is the 1-indexed
    kcp.x band position.
    """

    orbital: VariationalOrbital
    fixed_band: int
    spin_channel: SpinChannel
    band_index: int
    alpha_guess: float


class EmptyIterItem(TypedDict):
    """One per-orbital work item for the *empty* Delta-SCF fan-out.

    Built by :func:`build_empty_iter_source` and consumed field-by-field
    by :func:`ComputeOrbitalScreeningParameters`, which scatters one
    :func:`ComputeEmptyOrbitalScreeningParameter` per representative empty orbital. The
    three parameter dicts are fully baked into the (possibly spin-swapped)
    kcp.x frame; ``spin_channel`` / ``band_index`` stay in the physical
    frame for indexing the trial-KI lambda matrices. ``overlay`` is the
    ``pz_print`` save-file map (``{}`` when no spin swap is needed).
    """

    orbital: VariationalOrbital
    spin_channel: SpinChannel
    band_index: int
    alpha_guess: float
    dummy_parameters: dict
    pz_parameters: dict
    n_plus_1_parameters: dict
    overlay: dict


def build_filled_iter_source(
    nelup: int | None,
    neldw: int | None,
    orbitals: list[VariationalOrbital],
    filled_alphas: dict,
) -> dict[str, FilledIterItem]:
    """Materialise the per-orbital items for the *filled* fan-out loop.

    A plain function (not a ``@task``): it runs inside the deferred body
    of :func:`ComputeOrbitalScreeningParameters`, where ``orbitals`` and
    ``filled_alphas`` are already concrete values, and its return dict
    is iterated by a native ``for`` loop.

    Emits one entry per **representative** filled orbital
    (``o["representative"] is True``); non-representative orbitals
    inherit their group's alpha after the gather, via
    :func:`expand_alphas_by_group`. When grouping is disabled
    (:func:`assign_orbital_groups` short-circuit on ``tol is None``)
    every orbital is its own representative and the fan-out is
    unchanged.

    Each emitted item is a :class:`FilledIterItem`; ``band_index`` is
    the 0-indexed numpy index into the trial-KI lambda matrix and
    ``alpha_guess`` is the per-orbital alpha in use this refinement
    iteration.

    ``filled_alphas`` is keyed by spin tag (:class:`SpinChannel`'s
    string values — what survives AiiDA's serializer round-trip) and
    maps to per-channel alpha lists.
    On iteration 1 the caller passes :func:`generate_alphas`'s uniform
    output; on subsequent iterations the previous iteration's gathered
    alphas.

    kcp.x always runs with ``nspin=2`` in our KI flow; closed-shell
    (``spin_polarized=False``) is signalled inside the ``orbitals``
    list by emitting a single :attr:`SpinChannel.NONE` channel.
    """
    if nelup is None or neldw is None:
        raise ValueError("nelup and neldw are required (kcp.x runs at nspin=2)")
    out: dict[str, FilledIterItem] = {}
    for o in orbitals:
        if not o["representative"] or not o["filled"]:
            continue
        spin = SpinChannel(o["spin"])
        index = o["index"]
        alphas_for_spin = filled_alphas[spin.value]
        # kcp.x ``fixed_band`` indexing for spin-polarised systems
        # interleaves the filled blocks before the empty ones —
        # DOWN-channel bands are shifted by ``nelup`` (not by the
        # symmetric halved count, which is wrong when nelup != neldw).
        fixed_band = index + nelup if spin == SpinChannel.DOWN else index
        out[map_key_for(o)] = {
            "orbital": o,
            "fixed_band": fixed_band,
            "spin_channel": spin,
            "band_index": index - 1,
            "alpha_guess": alphas_for_spin[index - 1],
        }
    return out


def build_empty_iter_source(
    base: KcpBaseInputs,
    nbnd: int,
    orbitals: list[VariationalOrbital],
    empty_alphas: dict,
    correction: Correction = Correction.KI,
) -> dict[str, EmptyIterItem]:
    """Materialise the per-orbital items for the *empty* fan-out loop.

    A plain function (not a ``@task``): it runs inside the deferred body
    of :func:`ComputeOrbitalScreeningParameters`, where ``orbitals`` / ``base`` /
    ``empty_alphas`` are already concrete values, so the spin-aware
    branching below evaluates on real ints and enums.

    Empty orbitals come *after* the filled manifold within each per-spin
    block, so ``fixed_band`` and ``band_index`` are offset by
    ``n_filled_per_channel``. ``index_empty_to_save`` is 1-based within
    the per-spin empty manifold (kcp.x convention).

    ``empty_alphas`` follows the same shape as ``filled_alphas`` in
    :func:`build_filled_iter_source`: keyed by spin tag, mapping to
    per-channel alpha lists indexed within the empty manifold (so index
    ``0`` is the first empty orbital, *not* the global band index).

    For each orbital the per-iter dict carries the three fully-baked
    kcp.x parameter dicts (``dummy_parameters``, ``pz_parameters``,
    ``n_plus_1_parameters``) so ``ComputeEmptyOrbitalScreeningParameter`` stays a thin
    three-step pipeline. Spin-aware electron addition and the kcp-frame
    spin-swap decision happen here so a wrong branch can't silently
    produce kcp.x inputs that violate ``nupdwn(1) >= nupdwn(2)`` on the
    DOWN-channel empty orbital. ``spin_channel`` and ``band_index`` are emitted in the
    **physical** frame because they index the trial KI's lambda
    matrices in :func:`compute_alpha_from_dscf` (which were computed
    before any swap).

    See :func:`build_filled_iter_source` for the spin-channel emission
    rule (closed-shell emits a single :attr:`SpinChannel.NONE` channel;
    spin-polarised emits UP + DOWN).
    """
    if base.nelup is None or base.neldw is None:
        raise ValueError("nelup and neldw are required (kcp.x runs at nspin=2)")
    overlay_swap = _spin_swap_save_overlay(nspin=2)
    n_empty_up = max(0, nbnd - base.nelup)
    max_n_filled = max(base.nelup, base.neldw)
    out: dict[str, EmptyIterItem] = {}
    for o in orbitals:
        if not o["representative"] or o["filled"]:
            continue
        spin = SpinChannel(o["spin"])
        orb_index = o["index"]
        n_filled_this_spin = base.neldw if spin == SpinChannel.DOWN else base.nelup

        # ``fixed_band`` is clamped to the per-spin LUMO position
        # (``min(band.index, n_filled+1)``). kcp.x interprets
        # ``fixed_band`` as the slot where the constrained orbital
        # lands after re-ordering, not as the band's pre-screening
        # index — and that slot is always the LUMO of the relevant
        # spin manifold. The actual orbital we're constraining is
        # selected by ``index_empty_to_save`` (which pz_print writes
        # into ``evcfixed_empty.dat``). Passing the un-clamped per-spin
        # position corrupts kcp.x's internal ordering on higher empties
        # of an open-shell DOWN channel.
        fixed_band_per_spin = n_filled_this_spin + 1
        fixed_band = (
            fixed_band_per_spin + base.nelup if spin == SpinChannel.DOWN else fixed_band_per_spin
        )

        # ``index_empty_to_save`` is a *global* 1-indexed counter
        # across the full empty manifold (UP empties first, then
        # DOWN): DOWN's per-spin index is shifted by ``n_empty_up``.
        # ``i`` here is the 0-indexed position within this spin's empty
        # manifold (= ``orb_index - n_filled_this_spin - 1``).
        i = orb_index - n_filled_this_spin - 1
        if spin == SpinChannel.DOWN:
            index_empty_to_save = i + 1 + n_empty_up
        else:
            index_empty_to_save = i + 1

        # The extra electron goes in the channel of the orbital being
        # screened. ``SpinChannel.NONE`` (closed-shell input) defaults to UP.
        if spin == SpinChannel.DOWN:
            nelup_np1 = base.nelup
            neldw_np1 = base.neldw + 1
        else:
            nelup_np1 = base.nelup + 1
            neldw_np1 = base.neldw

        # kcp.x requires nupdwn(1) >= nupdwn(2). Swap when violated.
        # Ferromag with room (nelup=12, neldw=8 + DOWN: post-add
        # (12, 9)) satisfies the constraint without a swap.
        base_n_plus_1 = replace(base, nelec=base.nelec + 1, nelup=nelup_np1, neldw=neldw_np1)
        base_n = base
        fb = fixed_band
        overlay: dict[str, str] = {}
        if nelup_np1 is not None and neldw_np1 is not None and nelup_np1 < neldw_np1:
            base_n_plus_1, fb = _swap_kcp_frame(base_n_plus_1, fixed_band=fb, nbup=nbnd, nbdw=nbnd)
            base_n, _ = _swap_kcp_frame(base_n, fixed_band=fixed_band, nbup=nbnd, nbdw=nbnd)
            overlay = overlay_swap

        dummy_p = _build_dft_n_plus_1_dummy_parameters(
            base_n_plus_1, fixed_band=fb, index_empty_to_save=index_empty_to_save
        )
        # ``correction`` routes the print step (PZ flavour for KI, NKIPZ
        # for KIPZ) and the N+1 step (plain DFT for KI, orbdep-on for KIPZ).
        pz_p = _build_print_parameters(
            base_n,
            nbnd=nbnd,
            fixed_band=fb,
            index_empty_to_save=index_empty_to_save,
            correction=correction,
        )
        n_plus_1_p = _build_n_plus_1_parameters(
            base_n_plus_1,
            fixed_band=fb,
            index_empty_to_save=index_empty_to_save,
            correction=correction,
        )

        # ``band_index`` indexes into the trial-KI lambda matrix
        # (shape ``(nspin, n, n)``). The parser
        # (``parsers/kcp.py:_parse_lambdas``) block-diag stacks the
        # filled and empty Hamiltonians; kcp.x sizes both blocks to
        # the per-spin **max** (``max(nelup, neldw)`` for filled),
        # zero-padding the spin with fewer real bands. So the i-th
        # empty of *this* spin lives at row/col ``max_n_filled + i``,
        # not at the per-spin physical position. Closed-shell symmetric
        # systems happened to agree (``max == per-spin``) — bites
        # only open-shell.
        band_index = max_n_filled + i
        alphas_for_spin = empty_alphas[spin.value]
        out[map_key_for(o)] = {
            # Structured identity travels with the per-orbital data so
            # downstream consumers don't have to re-derive spin /
            # index / filled from the Map key string.
            "orbital": o,
            # Physical labels (used by ``compute_alpha_from_dscf`` to
            # index trial-KI lambda matrices computed pre-swap).
            "spin_channel": spin,
            "band_index": band_index,
            "alpha_guess": alphas_for_spin[i],
            # Fully-baked kcp.x parameter dicts in the (possibly
            # swapped) kcp frame; ``ComputeEmptyOrbitalScreeningParameter`` merges in
            # per-step overrides on top.
            "dummy_parameters": dummy_p,
            "pz_parameters": pz_p,
            "n_plus_1_parameters": n_plus_1_p,
            # Save-file overlay for ``pz_print`` (whose parent is the
            # trial KI's physical-frame save). Empty when no swap is
            # needed.
            "overlay": overlay,
        }
    return out


# ----------------------------------------------------------------------
# Public graphs
# ----------------------------------------------------------------------


@task.graph
def InitializeOrbitals(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    nelec: int,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int = 2,
    nelup: int | None = None,
    neldw: int | None = None,
    tot_magnetization: int | None = None,
    restart_mode: str = "from_scratch",
    outerloop: bool = True,
    parent_folder: orm.RemoteData | None = None,
    name: str = "dft_init",
    overrides: KcpNamelistOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> DFTCPOutputs:
    """Run a kcp.x DFT SCF (``do_orbdep=False``).

    Default behaviour is the standard from-scratch nspin=2 DFT init; the
    extra keyword-only flags exist to support the closed-shell
    spin-symmetric 3-step init chain: an nspin=1 from-scratch run with
    outer loop, then an nspin=2 from-scratch *dummy* with the outer loop
    disabled, then the final nspin=2 restarted run.

    Args:
        restart_mode: forwarded to ``&CONTROL.restart_mode``. Pair with
            ``parent_folder`` when restarting.
        outerloop: when ``False``, disables the outer / empty-manifold
            outer loops (used by the nspin=1/2 dummy steps).
        parent_folder: ``RemoteData`` whose save tree is symlinked onto
            the working directory; required for ``restart_mode='restart'``.
        name: ``call_link_label`` for the underlying kcp.x calc. Defaults
            to ``"dft_init"`` so the standard single-step path keeps its
            existing label; chained init callers should override this to
            disambiguate the three sub-steps.

    ``pseudos`` and the electron counts arrive as sockets resolved
    upstream by :func:`resolve_pseudo_family_task` and
    :func:`count_electrons_task` respectively, keeping all
    QueryBuilder / structure-walking work inside dedicated process
    nodes (avoids the ``TaggedValue`` proxy hitting SQLAlchemy).
    """
    base = _kcp_base_inputs(
        structure,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
    )
    parameters = _build_dft_parameters(
        base, nbnd=nbnd, restart_mode=restart_mode, outerloop=outerloop
    )
    if overrides:
        parameters = recursive_merge(parameters, overrides)

    inputs = _build_kcp_inputs(
        code,
        structure,
        parameters,
        pseudos,
        options=options,
        parent_folder=parent_folder,
        name=name,
    )
    outputs = KcpStep(**inputs)

    return DFTCPOutputs(
        parameters=outputs["output_parameters"],
        eigenvalues=outputs["output_eigenvalues"],
        remote_folder=outputs["remote_folder"],
    )


@task.graph
def KoopmansDSCFWorkflow(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudo_family: str,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int = 2,
    tot_magnetization: int | None = None,
    correction: Correction = Correction.KI,
    init_orbitals: VariationalOrbitalType = VariationalOrbitalType.KOHN_SHAM,
    alpha_numsteps: int = 1,
    fix_spin_contamination: bool = False,
    initial_alpha: float = 0.6,
    spin_polarized: bool = False,
    orbital_groups_self_hartree_tol: float | None = None,
    codes: Codes | None = None,
    blocks: list | None = None,
    kgrid: list[int] | None = None,
    kpoints: orm.KpointsData | None = None,
    gamma_only: bool = False,
    wannier_protocol: str | None = None,
    wannier_overrides: dict[str, Any] | None = None,
    mp_correction: bool | None = None,
    eps_inf: float | None = None,
    overrides: KoopmansDSCFOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> KoopmansDSCFOutputs:
    """Koopmans DSCF workflow — DFT init → trial KI → per-orbital DSCF refinement → final KI.

    Runs up to ``alpha_numsteps`` iterations of alpha refinement: each
    iteration's trial KI computes lambda matrices for the alpha formula
    and the per-orbital Delta-SCF fan-out (one ``dft_n-1`` for each
    filled orbital and one ``dft_n+1_dummy → pz_print → dft_n+1``
    triplet for each empty orbital) refines every alpha; a final KI
    then re-runs with the converged alphas.

    Two initialisation routes select on ``init_orbitals``:

    * ``'kohn-sham'`` (molecular): the DFT init runs on ``structure``
      directly and the trial KI receives the KS-as-variational overlay.
    * ``'mlwfs'`` / ``'projwfs'`` (periodic): wannierise → fold to the
      ``diag(kgrid)`` supercell → Wannier-seeded ``dft_init``
      (:func:`~aiida_koopmans.workgraphs.mlwf_init.MlwfInitialization`).
      Every kcp.x step then runs on the Γ-point supercell, with the
      extensive inputs (``nbnd``, ``tot_magnetization``, and — via the
      supercell structure — the electron counts) scaled by
      ``prod(kgrid)`` (legacy ``convert_kcp_to_supercell``). This route
      additionally requires ``codes`` (pw / wannier90 / pw2wannier90 /
      wann2kcp / merge_evc), ``blocks`` (projection blocks with
      *primitive* band indices; ``nbnd`` stays the primitive per-cell
      count too), ``kgrid``, and the matching explicit ``kpoints`` mesh.

    Spin-symmetrisation (``fix_spin_contamination=True``) is still
    deferred; ``_validate_scope`` rejects that path.
    """
    from aiida_koopmans.workgraphs.mlwf_init import MlwfInitialization
    from aiida_koopmans.workgraphs.supercell import (
        primitive_to_supercell,
        scale_extensive,
        supercell_size,
    )

    _validate_scope(
        correction=correction,
        init_orbitals=init_orbitals,
        fix_spin_contamination=fix_spin_contamination,
        structure=structure,
        blocks=blocks,
        kgrid=kgrid,
        kpoints=kpoints,
        codes=codes,
    )

    dft_overrides = overrides.get("dft") if overrides else None
    wannier_init = init_orbitals in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    )

    # Image-charge correction defaults (legacy ``_workflow.py:582-593``):
    # the Makov-Payne correction to the Delta-SCF energies is on for
    # periodic supercell runs and off (indeed forbidden) for molecules;
    # ``eps_inf`` falls back to 1.0 (vacuum — legacy warns that this is a
    # crude default for real materials).
    if mp_correction is None:
        mp_correction = wannier_init
    if eps_inf is None:
        eps_inf = 1.0

    # For the periodic Wannier route every kcp.x step runs on the Γ-point
    # supercell; the extensive inputs scale by the primitive-cell count.
    # ``_validate_scope`` guarantees ``kgrid`` and ``codes`` are set on
    # this route.
    if wannier_init:
        ncells = supercell_size(cast("list[int]", kgrid))
        run_structure = primitive_to_supercell(
            structure=structure,
            kgrid=kgrid,
            metadata={"call_link_label": "make_supercell"},
        ).result
        # The supercell bands are exactly the folded Wannier manifolds:
        # (occ + emp WFs per primitive cell) x images. Scaling the primitive
        # ``nbnd`` instead demands more empty states than the folded
        # ``evc0_empty`` provides — kcp.x then rejects the file
        # ("wavefunctions dimensions changed") and silently random-initialises
        # the empties. Legacy reference: Si 2x2x2 runs nbnd=64 (4 occ + 4 emp
        # WFs) even though the primitive wannierization used nbnd=20.
        run_nbnd = sum(block["num_wann"] for block in blocks) * ncells
        run_tot_magnetization = scale_extensive(tot_magnetization, ncells)
    else:
        run_structure = structure
        run_nbnd = nbnd
        run_tot_magnetization = tot_magnetization

    # Resolve pseudo family + electron counts once, at runtime, so the
    # results flow downstream as plain AiiDA-typed sockets instead of
    # ``TaggedValue`` proxies (the failure mode of the inline plain-Python
    # call inside a nested ``@task.graph`` body). Evaluated on the *run*
    # structure, so the supercell's electron counts come out pre-scaled.
    pseudos = resolve_pseudo_family_task(
        family_label=pseudo_family,
        structure=run_structure,
    )
    counts = count_electrons_task(
        structure=run_structure,
        pseudos=pseudos,
        nspin=nspin,
        tot_magnetization=run_tot_magnetization,
    )
    nelec = counts["nelec"]
    nelup = counts["nelup"]
    neldw = counts["neldw"]

    initial_evc_occupied1 = None
    initial_evc_occupied2 = None
    if wannier_init:
        init = MlwfInitialization(
            codes={**cast("dict", codes), "kcp": code},
            structure=structure,
            supercell=run_structure,
            pseudos=pseudos,
            blocks=blocks,
            kpoints=kpoints,
            kgrid=kgrid,
            nelec=nelec,
            nelup=nelup,
            neldw=neldw,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            nbnd=run_nbnd,
            nspin=nspin,
            tot_magnetization=run_tot_magnetization,
            spin_polarized=spin_polarized,
            gamma_only=gamma_only,
            pseudo_family=pseudo_family,
            wannier_protocol=wannier_protocol,
            wannier_overrides=wannier_overrides,
            options=options,
            metadata={"call_link_label": "wannier_initialization"},
        )
        dft_remote = init["remote_folder"]
        initial_evc_occupied1 = init["evc_occupied1"]
        initial_evc_occupied2 = init["evc_occupied2"]
    elif spin_polarized:
        # Spin-polarised systems are seeded directly from a single
        # nspin=2 from-scratch run: the up/down channels are independent,
        # with no pre-symmetrisation.
        dft = InitializeOrbitals(
            code=code,
            structure=structure,
            pseudos=pseudos,
            nelec=nelec,
            nelup=nelup,
            neldw=neldw,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            nbnd=nbnd,
            nspin=nspin,
            tot_magnetization=tot_magnetization,
            overrides=dft_overrides,
            options=options,
            metadata={"call_link_label": "dft_init"},
        )
        dft_remote = dft["remote_folder"]
    else:
        # Closed-shell spin-symmetric init chain:
        #
        # 1. nspin=1 from scratch — converges the single-channel solution.
        # 2. nspin=2 from-scratch dummy — lays out the nspin=2 save tree
        #    skeleton, outer loop disabled so the wavefunction content is
        #    irrelevant (only the layout matters).
        # 3. ConvertSpin1ToSpin2 — splices step-1 wavefunctions into the
        #    step-2 save layout, producing a spin-symmetric save.
        # 4. nspin=2 restart — final init starting from the symmetrised
        #    save; this is the ``remote_folder`` consumed by the
        #    downstream ComputeScreeningParameters.
        dft_nspin1 = InitializeOrbitals(
            code=code,
            structure=structure,
            pseudos=pseudos,
            nelec=nelec,
            nelup=None,
            neldw=None,
            tot_magnetization=None,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            nbnd=nbnd,
            nspin=1,
            outerloop=True,
            restart_mode="from_scratch",
            overrides=dft_overrides,
            options=options,
            metadata={"call_link_label": "dft_init_nspin1"},
        )

        dft_nspin2_dummy = InitializeOrbitals(
            code=code,
            structure=structure,
            pseudos=pseudos,
            nelec=nelec,
            nelup=nelup,
            neldw=neldw,
            tot_magnetization=tot_magnetization,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            nbnd=nbnd,
            nspin=2,
            outerloop=False,
            restart_mode="from_scratch",
            overrides=dft_overrides,
            options=options,
            metadata={"call_link_label": "dft_init_nspin2_dummy"},
        )

        # ``convert_spin1_to_spin2`` is a ``@task.calcfunction`` — pure
        # local file substitution, no external binary, no ``code`` /
        # ``computer`` metadata needed. Provenance is captured via a
        # ``CalcFunctionNode``.
        converted = convert_spin1_to_spin2(
            spin1_parent_folder=dft_nspin1["remote_folder"],
            spin2_dummy_parent_folder=dft_nspin2_dummy["remote_folder"],
            metadata={"call_link_label": "convert_spin1_to_spin2"},
        )

        dft = InitializeOrbitals(
            code=code,
            structure=structure,
            pseudos=pseudos,
            nelec=nelec,
            nelup=nelup,
            neldw=neldw,
            tot_magnetization=tot_magnetization,
            ecutwfc=ecutwfc,
            ecutrho=ecutrho,
            nbnd=nbnd,
            nspin=2,
            outerloop=True,
            restart_mode="restart",
            parent_folder=converted["remote_folder"],
            overrides=dft_overrides,
            options=options,
            metadata={"call_link_label": "dft_init_nspin2"},
        )
        dft_remote = dft["remote_folder"]

    screening = ComputeScreeningParameters(
        code=code,
        structure=run_structure,
        pseudos=pseudos,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=run_nbnd,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=run_tot_magnetization,
        initial_alpha=initial_alpha,
        correction=correction,
        init_orbitals=init_orbitals,
        spin_polarized=spin_polarized,
        alpha_numsteps=alpha_numsteps,
        self_hartree_tol=orbital_groups_self_hartree_tol,
        dft_remote=dft_remote,
        initial_evc_occupied1=initial_evc_occupied1,
        initial_evc_occupied2=initial_evc_occupied2,
        mp_correction=mp_correction,
        eps_inf=eps_inf,
        overrides=overrides,
        options=options,
    )

    # ------------------------------------------------------------------
    # Final KI: applies the converged screening parameters to produce
    # the Koopmans-corrected spectrum. Restarts from the *last
    # iteration's* trial KI save (``screening["trial_remote"]``) so it
    # inherits the converged variational orbital basis (not the bare DFT
    # save).
    #
    # Built via a ``RunFinalKI`` @task.graph wrapper rather than inline
    # ``KcpStep(...)`` because the parameter-builder arithmetic
    # (``conv_thr = 1e-9 * nelec``) needs ``nelec`` as a plain int.
    # Here at the workflow level ``nelec`` is a socket (output of
    # ``count_electrons_task``); the @task.graph boundary unwraps it.
    #
    # ``alphas`` and ``parent_folder`` are wired explicitly from the
    # ``screening`` outputs at the call site so the provenance graph
    # shows that the final KI consumes the converged DSCF screening
    # parameters.
    # ------------------------------------------------------------------
    ki_final = RunFinalKI(
        code=code,
        structure=run_structure,
        pseudos=pseudos,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=run_nbnd,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=run_tot_magnetization,
        correction=correction,
        alphas=screening["alphas"],
        parent_folder=screening["trial_remote"],
        overrides=overrides.get("ki") if overrides else None,
        options=options,
    )
    return KoopmansDSCFOutputs(
        parameters=ki_final["parameters"],
        eigenvalues=ki_final["eigenvalues"],
        lambdas=ki_final["lambdas"],
        bare_lambdas=ki_final["bare_lambdas"],
        remote_folder=ki_final["remote_folder"],
        alphas=screening["alphas"],
    )


@task.graph
def RunFinalKI(
    *,
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int,
    nelec: int,
    correction: Correction,
    alphas: AlphaScreening,
    parent_folder: orm.RemoteData,
    nelup: int | None = None,
    neldw: int | None = None,
    tot_magnetization: int | None = None,
    overrides: KcpNamelistOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> KIFinalOutputs:
    """Apply the converged screening parameters via a final KI run.

    Thin wrapper around a single ``KcpCalculation``. Exists as its own
    ``@task.graph`` so the parameter-builder arithmetic (``conv_thr =
    1e-9 * nelec`` etc.) runs in a scope where ``nelec`` is a plain
    int — at the outer ``KoopmansDSCFWorkflow`` level ``nelec`` is a
    socket (output of ``count_electrons_task``) and the resulting
    socket-valued parameters dict can't be serialised into the
    ``KcpCalculation``'s attributes.

    The wrapper's ``call_link_label`` and the inner CalcJob's
    ``ki_final`` are both prettified to ``"KI Final"`` in the progress
    table — the suppression rule in ``add_process_rows`` then collapses
    the wrapper-and-child pair into a single row.
    """
    base = _kcp_base_inputs(
        structure,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
    )
    ki_parameters = _build_orbdep_parameters(base, nbnd=nbnd, correction=correction)
    if overrides:
        ki_parameters = recursive_merge(ki_parameters, overrides)
    final_inputs = _build_kcp_inputs(
        code,
        structure,
        ki_parameters,
        pseudos,
        options=options,
        alphas=alphas,
        parent_folder=parent_folder,
        name="kipz_final" if correction == Correction.KIPZ else "ki_final",
    )
    final = KcpStep(**final_inputs)
    return KIFinalOutputs(
        parameters=final["output_parameters"],
        eigenvalues=final["output_eigenvalues"],
        lambdas=final["output_lambdas"],
        bare_lambdas=final["output_bare_lambdas"],
        remote_folder=final["remote_folder"],
    )


@task.graph
def ComputeFilledOrbitalScreeningParameter(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    ecutwfc: float,
    ecutrho: float,
    nspin: int,
    nelec: int,
    fixed_band: int,
    spin_channel: SpinChannel,
    band_index: int,
    alpha_guess: float,
    trial_remote: orm.RemoteData,
    trial_output_parameters: dict,
    trial_lambdas: np.ndarray,
    trial_bare_lambdas: np.ndarray,
    nelup: int | None = None,
    neldw: int | None = None,
    tot_magnetization: int | None = None,
    mp_correction: bool = False,
    eps_inf: float = 1.0,
    overrides: KcpNamelistOverrides | None = None,
    options: dict[str, Any] | None = None,
    correction: Correction = Correction.KI,
) -> OrbitalDeltaSCFOutputs:
    """Compute the new alpha for one **filled** orbital via Delta-SCF.

    Submits a single ``dft_n-1`` kcp.x run (with ``fixed_band=fixed_band``,
    one electron pulled out of that orbital via ``f_cutoff=1e-5``), then
    evaluates the alpha formula at ``(spin_channel, band_index)``
    against the trial KI's lambda and bare-lambda matrices.

    Args:
        fixed_band: 1-indexed kcp.x band index of the orbital to refine.
            Goes into ``&SYSTEM.fixed_band``.
        spin_channel: which array key on the trial KI's ArrayData outputs
            to query for the lambda diagonal.
        band_index: 0-indexed numpy index into the lambda matrix
            (``= fixed_band - 1``).
        alpha_guess: the alpha currently assigned to this orbital
            (initial guess on iteration 1; previous-iteration alpha
            otherwise).
        trial_remote: ``RemoteData`` of the trial KI calc — sets
            ``parent_folder`` so kcp.x finds the orbital-dependent save.
        trial_output_parameters / trial_lambdas / trial_bare_lambdas:
            outputs of the same trial KI; passed in here rather than
            re-loaded so the orbital task is provenance-pure.
    """
    base = _kcp_base_inputs(
        structure,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
    )
    parameters = _build_n_minus_1_parameters(base, fixed_band=fixed_band, correction=correction)
    if overrides:
        parameters = recursive_merge(parameters, overrides)

    inputs = _build_kcp_inputs(
        code,
        structure,
        parameters,
        pseudos,
        options=options,
        parent_folder=trial_remote,
        # ``call_link_label`` reflects the calc-type name: plain DFT for
        # KI, KIPZ-flavoured for KIPZ. Shows up in the live progress
        # display via ``progress.prettify_label``.
        name="kipz_n_minus_1" if correction == Correction.KIPZ else "dft_n_minus_1",
    )
    dft_outputs = KcpStep(**inputs)

    result = compute_alpha_from_dscf(
        trial_output_parameters=trial_output_parameters,
        perturbed_output_parameters=dft_outputs["output_parameters"],
        trial_lambdas=trial_lambdas,
        trial_bare_lambdas=trial_bare_lambdas,
        spin_channel=spin_channel,
        band_index=band_index,
        alpha_guess=alpha_guess,
        filled=True,
        mp_correction=mp_correction,
        eps_inf=eps_inf,
    )

    return OrbitalDeltaSCFOutputs(
        alpha=result["alpha"],
        error=result["error"],
    )


@task.graph
def ComputeEmptyOrbitalScreeningParameter(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    dummy_parameters: dict,
    pz_parameters: dict,
    n_plus_1_parameters: dict,
    overlay: dict,
    spin_channel: SpinChannel,
    band_index: int,
    alpha_guess: float,
    pz_alphas: AlphaScreening,
    trial_remote: orm.RemoteData,
    trial_output_parameters: dict,
    trial_lambdas: np.ndarray,
    trial_bare_lambdas: np.ndarray,
    mp_correction: bool = False,
    eps_inf: float = 1.0,
    overrides: dict[str, KcpNamelistOverrides | None] | None = None,
    options: dict[str, Any] | None = None,
    correction: Correction = Correction.KI,
) -> OrbitalDeltaSCFOutputs:
    """Compute the new alpha for one **empty** orbital via Delta-SCF.

    Three-step kcp.x sub-pipeline:

    1. ``dft_n+1_dummy`` — scratch DFT with the empty orbital populated
       (``fixed_band`` + ``nelec=N+1``); writes the save layout the
       subsequent steps consume. Rerun every iteration.
    2. ``pz_print`` — PZ run on the fixed empty orbital (parent =
       ``trial_remote``) that writes ``evcfixed_empty.dat``.
    3. ``dft_n+1`` — SCF DFT (parent = the dummy + ``pz_print``'s
       ``evcfixed_empty.dat`` via ``parent_folder_evcfixed``).

    Then the alpha calcfunction extracts the (spin_channel, band_index)
    diagonal of the trial KI's lambda matrices, computes ``dE``, and
    returns the new alpha + convergence error.

    Every per-orbital quantity (electron counts, ``fixed_band``,
    ``index_empty_to_save``, ``nbnd``, cutoffs, the spin-swap decision
    and overlay payload) is pre-baked into the three parameter dicts
    and ``overlay`` by :func:`build_empty_iter_source` — a real
    ``@task`` that sees resolved values for ``spin_channel`` /
    ``nelup`` / ``neldw`` and can therefore apply the kcp-frame swap
    correctly. The graph body is branch-free and never compares a
    deferred socket to a :class:`SpinChannel` enum.

    ``spin_channel`` / ``band_index`` arrive in the **physical** frame
    (they index the trial KI's lambda matrices, computed pre-swap);
    the three parameter dicts are in the kcp.x frame (which differs
    only when the swap fires).

    ``pz_alphas`` is built upstream by :func:`generate_alphas` (a
    uniform :class:`AlphaScreening` of ``alpha_guess`` values) — no
    socket arithmetic happens inside this graph body.
    """
    dummy_overrides = overrides.get("dft_dummy") if overrides else None  # type: ignore[union-attr]
    pz_overrides = overrides.get("pz_print") if overrides else None  # type: ignore[union-attr]
    n_plus_1_overrides = overrides.get("dft_n_plus_1") if overrides else None  # type: ignore[union-attr]

    if dummy_overrides:
        dummy_parameters = recursive_merge(dummy_parameters, dummy_overrides)
    dummy_inputs = _build_kcp_inputs(
        code,
        structure,
        dummy_parameters,
        pseudos,
        options=options,
        name="dft_n_plus_1_dummy",  # always plain DFT in both KI and KIPZ
    )
    dummy_outputs = KcpStep(**dummy_inputs)

    if pz_overrides:
        pz_parameters = recursive_merge(pz_parameters, pz_overrides)
    # ``pz_print`` reads orbitals from the trial KI (it operates at the
    # original electron count) and writes ``evcfixed_empty.dat`` for
    # the next step. ``pz_alphas`` is the uniform-``alpha_guess`` payload
    # already built by ``generate_alphas`` upstream. ``overlay`` is the
    # spin-swap save-file map (``{}`` when no swap is needed);
    # ``_build_kcp_inputs`` skips the overlay socket when the dict is
    # empty.
    pz_inputs = _build_kcp_inputs(
        code,
        structure,
        pz_parameters,
        pseudos,
        options=options,
        alphas=pz_alphas,
        parent_folder=trial_remote,
        variational_orbital_overlays=overlay,
        name="kipz_print" if correction == Correction.KIPZ else "pz_print",
    )
    pz_outputs = KcpStep(**pz_inputs)

    if n_plus_1_overrides:
        n_plus_1_parameters = recursive_merge(n_plus_1_parameters, n_plus_1_overrides)
    n_plus_1_inputs = _build_kcp_inputs(
        code,
        structure,
        n_plus_1_parameters,
        pseudos,
        options=options,
        parent_folder=dummy_outputs["remote_folder"],
        parent_folder_evcfixed=pz_outputs["remote_folder"],
        name="kipz_n_plus_1" if correction == Correction.KIPZ else "dft_n_plus_1",
    )
    n_plus_1_outputs = KcpStep(**n_plus_1_inputs)

    result = compute_alpha_from_dscf(
        trial_output_parameters=trial_output_parameters,
        perturbed_output_parameters=n_plus_1_outputs["output_parameters"],
        trial_lambdas=trial_lambdas,
        trial_bare_lambdas=trial_bare_lambdas,
        spin_channel=spin_channel,
        band_index=band_index,
        alpha_guess=alpha_guess,
        filled=False,
        mp_correction=mp_correction,
        eps_inf=eps_inf,
    )

    return OrbitalDeltaSCFOutputs(
        alpha=result["alpha"],
        error=result["error"],
    )


# ----------------------------------------------------------------------
# Per-orbital DSCF fan-out. A separate ``@task.graph`` (rather than inline in
# ``ScreeningIteration``) because the fan-out cardinality depends on
# ``orbitals`` — a *runtime* output of ``assign_orbital_groups`` (it reads the
# trial KI's self-Hartree metric). Inside this deferred body ``orbitals`` is
# concrete, so the scatter is a native ``for`` loop and the gather a plain
# dict of per-orbital output sockets (the documented dynamic scatter-gather
# pattern; see ``block_wannierize.py`` for the same shape).
# ----------------------------------------------------------------------


@task.graph
def ComputeOrbitalScreeningParameters(
    *,
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    base: KcpBaseInputs,
    nbnd: int,
    correction: Correction,
    orbitals: list[VariationalOrbital],
    current_alphas: AlphaScreening,
    trial_remote: orm.RemoteData,
    trial_output_parameters: dict,
    trial_lambdas: np.ndarray,
    trial_bare_lambdas: np.ndarray,
    mp_correction: bool = False,
    eps_inf: float = 1.0,
    filled_overrides: KcpNamelistOverrides | None = None,
    empty_overrides_dict: dict[str, KcpNamelistOverrides | None] | None = None,
    options: dict[str, Any] | None = None,
) -> _PerOrbitalAlphaOutputs:
    """Refine every representative orbital's alpha via per-orbital Delta-SCF.

    Scatters one :func:`ComputeFilledOrbitalScreeningParameter` per representative filled
    orbital and one :func:`ComputeEmptyOrbitalScreeningParameter` per representative empty
    orbital (native ``for`` loops over the item dicts built by
    :func:`build_filled_iter_source` / :func:`build_empty_iter_source`,
    which run inline here on the concrete ``orbitals`` list). The
    per-orbital sub-graphs share only the read-only trial-KI scratch, so
    they run in parallel. Each sub-graph's ``call_link_label`` is
    ``compute_alpha_<map_key>`` (e.g. ``compute_alpha_orb_3`` / ``compute_alpha_up_orb_10``).

    The gathered ``{map_key: alpha/error}`` socket dicts feed
    :func:`expand_alphas_by_group` (broadcast representative results onto
    group members) and :func:`assemble_alpha_screening` (pack into the
    per-spin :class:`AlphaScreening` shape kcp.x consumes).
    """
    filled_items = build_filled_iter_source(
        nelup=base.nelup,
        neldw=base.neldw,
        orbitals=orbitals,
        filled_alphas=current_alphas["filled"],
    )
    filled_alphas: dict[str, Any] = {}
    filled_errors: dict[str, Any] = {}
    for key, item in filled_items.items():
        filled_out = ComputeFilledOrbitalScreeningParameter(
            code=code,
            structure=structure,
            pseudos=pseudos,
            ecutwfc=base.ecutwfc,
            ecutrho=base.ecutrho,
            nspin=base.nspin,
            nelec=base.nelec,
            nelup=base.nelup,
            neldw=base.neldw,
            tot_magnetization=base.tot_magnetization,
            fixed_band=item["fixed_band"],
            spin_channel=item["spin_channel"],
            band_index=item["band_index"],
            alpha_guess=item["alpha_guess"],
            trial_remote=trial_remote,
            trial_output_parameters=trial_output_parameters,
            trial_lambdas=trial_lambdas,
            trial_bare_lambdas=trial_bare_lambdas,
            mp_correction=mp_correction,
            eps_inf=eps_inf,
            overrides=filled_overrides,
            options=options,
            correction=correction,
            metadata={"call_link_label": f"compute_alpha_{key}"},
        )
        filled_alphas[key] = filled_out["alpha"]
        filled_errors[key] = filled_out["error"]

    empty_items = build_empty_iter_source(
        base=base,
        nbnd=nbnd,
        orbitals=orbitals,
        empty_alphas=current_alphas["empty"],
        correction=correction,
    )
    empty_alphas: dict[str, Any] = {}
    empty_errors: dict[str, Any] = {}
    for key, empty_item in empty_items.items():
        empty_out = ComputeEmptyOrbitalScreeningParameter(
            code=code,
            structure=structure,
            pseudos=pseudos,
            dummy_parameters=empty_item["dummy_parameters"],
            pz_parameters=empty_item["pz_parameters"],
            n_plus_1_parameters=empty_item["n_plus_1_parameters"],
            overlay=empty_item["overlay"],
            spin_channel=empty_item["spin_channel"],
            band_index=empty_item["band_index"],
            alpha_guess=empty_item["alpha_guess"],
            pz_alphas=current_alphas,
            trial_remote=trial_remote,
            trial_output_parameters=trial_output_parameters,
            trial_lambdas=trial_lambdas,
            trial_bare_lambdas=trial_bare_lambdas,
            mp_correction=mp_correction,
            eps_inf=eps_inf,
            overrides=empty_overrides_dict,
            options=options,
            correction=correction,
            metadata={"call_link_label": f"compute_alpha_{key}"},
        )
        empty_alphas[key] = empty_out["alpha"]
        empty_errors[key] = empty_out["error"]

    # Broadcast each representative's alpha onto every member of its
    # group, then pack the per-orbital flat dicts into the per-spin
    # ``AlphaScreening`` shape kcp.x consumes. When grouping was
    # disabled (``self_hartree_tol=None``), ``expand_alphas_by_group``
    # is the identity modulo the filled/empty split.
    expanded = expand_alphas_by_group(
        filled_rep_alphas=filled_alphas or None,
        filled_rep_errors=filled_errors or None,
        empty_rep_alphas=empty_alphas or None,
        empty_rep_errors=empty_errors or None,
        orbitals=orbitals,
    )
    gathered = assemble_alpha_screening(
        orbitals=orbitals,
        filled_alphas=expanded["filled_alphas"],
        filled_errors=expanded["filled_errors"],
        empty_alphas=expanded["empty_alphas"],
        empty_errors=expanded["empty_errors"],
    )
    return _PerOrbitalAlphaOutputs(
        alphas=gathered["alphas"],
        errors=gathered["errors"],
    )


# ----------------------------------------------------------------------
# One DSCF iteration body: trial KI → per-orbital DSCF → assemble alphas.
# Extracted so the recursive ``RefineScreeningParameters`` in
# ``ComputeScreeningParameters`` can call it once per pass.
# ----------------------------------------------------------------------


@task.graph
def ScreeningIteration(
    *,
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    base: KcpBaseInputs,
    nbnd: int,
    correction: Correction,
    spin_polarized: bool,
    current_alphas: AlphaScreening,
    parent_folder: orm.RemoteData,
    is_first_iteration: bool = False,
    self_hartree_tol: float | None = None,
    variational_orbital_overlays: dict | None = None,
    initial_evc_occupied1: orm.SinglefileData | None = None,
    initial_evc_occupied2: orm.SinglefileData | None = None,
    mp_correction: bool = False,
    eps_inf: float = 1.0,
    ki_overrides: KcpNamelistOverrides | None = None,
    filled_overrides: KcpNamelistOverrides | None = None,
    empty_overrides_dict: dict[str, KcpNamelistOverrides | None] | None = None,
    options: dict[str, Any] | None = None,
) -> ScreeningIterationOutputs:
    """One iteration of the alpha-refinement loop.

    Runs a trial KI / KIPZ starting from ``current_alphas`` +
    ``parent_folder``, then the per-orbital Delta-SCF fan-out
    (:func:`ComputeOrbitalScreeningParameters` — a nested ``@task.graph`` because the
    fan-out cardinality depends on ``assign_orbital_groups``' runtime
    output), which packs the gathered per-orbital alphas back into an
    :class:`AlphaScreening`. Reports ``max_error`` so the recursive
    :func:`RefineScreeningParameters` can stop on convergence.

    ``is_first_iteration`` is forwarded to the trial-step builder so
    KIPZ's molecular first trial can run its inner-loop CG once;
    subsequent iterations restart from the previous trial's
    already-converged variational basis, so the inner loop is unnecessary.

    The trial step is named ``ki_trial`` / ``kipz_trial`` per
    ``correction``. ``variational_orbital_overlays`` is supplied on
    the first iteration only (the KS-as-variational overlay);
    subsequent iterations inherit the converged ``evc0N.dat`` from
    the previous iteration's trial save via the primary parent walk.

    ``initial_evc_occupied{1,2}`` are the Wannier-init counterpart of
    the KS overlay, likewise first-iteration-only: the folded
    ``evc_occupied{n}.dat`` files re-staged into the trial's read
    ``K00001`` with ``restart_from_wannier_pwscf`` switched on (legacy
    ``DeltaSCFIterationWorkflow``, ``_koopmans_dscf.py:505+521-522``;
    the empty-manifold ``evc0_empty{n}.dat`` flow through the
    ``dft_init`` parent save automatically).

    ``base`` is a frozen ``KcpBaseInputs`` dataclass and crosses this
    ``@task.graph`` boundary intact.
    """
    ki_parameters = _build_orbdep_parameters(
        base,
        nbnd=nbnd,
        correction=correction,
        is_first_iteration=is_first_iteration,
    )
    read_wavefunctions: dict[str, Any] | None = None
    if initial_evc_occupied1 is not None and initial_evc_occupied2 is not None:
        ki_parameters["SYSTEM"]["restart_from_wannier_pwscf"] = True
        read_wavefunctions = {
            "evc_occupied1": initial_evc_occupied1,
            "evc_occupied2": initial_evc_occupied2,
        }
    if ki_overrides:
        ki_parameters = recursive_merge(ki_parameters, ki_overrides)

    trial_inputs = _build_kcp_inputs(
        code,
        structure,
        ki_parameters,
        pseudos,
        options=options,
        alphas=current_alphas,
        parent_folder=parent_folder,
        variational_orbital_overlays=variational_orbital_overlays,
        read_wavefunctions=read_wavefunctions,
        name="kipz_trial" if correction == Correction.KIPZ else "ki_trial",
    )
    trial = KcpStep(**trial_inputs)

    # Cluster variational orbitals by trial-KI self-Hartree so each
    # group only screens one representative; non-representative members
    # inherit the alpha after the gather (see
    # :func:`expand_alphas_by_group`). With ``self_hartree_tol=None``
    # (the default) the task short-circuits to one-orbital-per-group
    # — every orbital is its own representative, so the fan-out is
    # unchanged.
    metric = extract_self_hartree_from_kcp(output_parameters=trial["output_parameters"])
    orbitals = assign_orbital_groups(
        metric=metric.result,
        nelup=base.nelup,
        neldw=base.neldw,
        nbnd=nbnd,
        spin_polarized=spin_polarized,
        tol=self_hartree_tol,
    )

    # Per-orbital Delta-SCF fan-out. Nested ``@task.graph`` so its body
    # runs once ``orbitals`` (a runtime output of ``assign_orbital_groups``)
    # is concrete — the scatter is then a native ``for`` loop, the gather
    # a plain dict of per-orbital sockets.
    per_orbital = ComputeOrbitalScreeningParameters(
        code=code,
        structure=structure,
        pseudos=pseudos,
        base=base,
        nbnd=nbnd,
        correction=correction,
        orbitals=orbitals.result,
        current_alphas=current_alphas,
        trial_remote=trial["remote_folder"],
        trial_output_parameters=trial["output_parameters"],
        trial_lambdas=trial["output_lambdas"],
        trial_bare_lambdas=trial["output_bare_lambdas"],
        mp_correction=mp_correction,
        eps_inf=eps_inf,
        filled_overrides=filled_overrides,
        empty_overrides_dict=empty_overrides_dict,
        options=options,
        metadata={"call_link_label": "compute_orbital_screening_parameters"},
    )

    max_err = max_alpha_error(
        filled_errors=per_orbital["errors"]["filled"],
        empty_errors=per_orbital["errors"]["empty"],
    )

    return {
        "alphas": per_orbital["alphas"],
        "errors": per_orbital["errors"],
        "trial_remote": trial["remote_folder"],
        "max_error": max_err.result,
    }


# ----------------------------------------------------------------------
# Alpha-refinement recursion: each call receives the *previous* iteration's
# outputs as concrete inputs, so the stop decision (converged, or iteration
# budget exhausted) is a plain Python branch in the deferred body.
# Non-terminal calls run one more ``ScreeningIteration`` and recurse on its
# outputs.
# ----------------------------------------------------------------------


@task.graph
def RefineScreeningParameters(
    *,
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    base: KcpBaseInputs,
    nbnd: int,
    correction: Correction,
    spin_polarized: bool,
    prev_alphas: AlphaScreening,
    prev_trial_remote: orm.RemoteData,
    prev_max_error: float,
    remaining_steps: int,
    alpha_conv_thr: float,
    self_hartree_tol: float | None = None,
    mp_correction: bool = False,
    eps_inf: float = 1.0,
    ki_overrides: KcpNamelistOverrides | None = None,
    filled_overrides: KcpNamelistOverrides | None = None,
    empty_overrides_dict: dict[str, KcpNamelistOverrides | None] | None = None,
    options: dict[str, Any] | None = None,
) -> ScreeningParametersOutputs:
    """Recursive alpha-refinement: iterate until converged or out of budget.

    Terminates (returning the previous iteration's ``alphas`` +
    ``trial_remote`` unchanged) when either the previous iteration's
    ``max |dE - lambda|`` fell below ``alpha_conv_thr`` (``1e-3 eV``) or
    ``remaining_steps`` hit zero (the ``alpha_numsteps`` cap; the first
    iteration is unrolled in :func:`ComputeScreeningParameters`, so the
    initial budget is ``alpha_numsteps - 1``).

    Otherwise runs one more :func:`ScreeningIteration` — parented on the
    previous trial KI's save, with no variational-orbital overlay (the
    converged ``evc0N.dat`` is already in place) and
    ``is_first_iteration=False`` (KIPZ's inner-loop CG ran on iteration
    1 only) — and recurses on its outputs with a decremented budget.
    """
    if remaining_steps <= 0 or prev_max_error < alpha_conv_thr:
        return ScreeningParametersOutputs(
            alphas=prev_alphas,
            trial_remote=prev_trial_remote,
        )

    iteration = ScreeningIteration(
        code=code,
        structure=structure,
        pseudos=pseudos,
        base=base,
        nbnd=nbnd,
        correction=correction,
        spin_polarized=spin_polarized,
        current_alphas=prev_alphas,
        parent_folder=prev_trial_remote,
        is_first_iteration=False,
        self_hartree_tol=self_hartree_tol,
        variational_orbital_overlays=None,
        mp_correction=mp_correction,
        eps_inf=eps_inf,
        ki_overrides=ki_overrides,
        filled_overrides=filled_overrides,
        empty_overrides_dict=empty_overrides_dict,
        options=options,
        metadata={"call_link_label": "screening_iteration"},
    )

    remainder = RefineScreeningParameters(
        code=code,
        structure=structure,
        pseudos=pseudos,
        base=base,
        nbnd=nbnd,
        correction=correction,
        spin_polarized=spin_polarized,
        prev_alphas=iteration["alphas"],
        prev_trial_remote=iteration["trial_remote"],
        prev_max_error=iteration["max_error"],
        remaining_steps=remaining_steps - 1,
        alpha_conv_thr=alpha_conv_thr,
        self_hartree_tol=self_hartree_tol,
        mp_correction=mp_correction,
        eps_inf=eps_inf,
        ki_overrides=ki_overrides,
        filled_overrides=filled_overrides,
        empty_overrides_dict=empty_overrides_dict,
        options=options,
        metadata={"call_link_label": "refine_screening_parameters"},
    )
    return ScreeningParametersOutputs(
        alphas=remainder["alphas"],
        trial_remote=remainder["trial_remote"],
    )


# ----------------------------------------------------------------------
# Alpha-refinement driver: unrolled first iteration + recursive refinement.
# ----------------------------------------------------------------------


@task.graph
def ComputeScreeningParameters(
    *,
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int,
    nelec: int,
    initial_alpha: float,
    correction: Correction,
    init_orbitals: VariationalOrbitalType,
    dft_remote: orm.RemoteData,
    nelup: int | None = None,
    neldw: int | None = None,
    tot_magnetization: int | None = None,
    spin_polarized: bool = False,
    alpha_numsteps: int = 1,
    alpha_conv_thr: float = 1.0e-3,
    self_hartree_tol: float | None = None,
    initial_evc_occupied1: orm.SinglefileData | None = None,
    initial_evc_occupied2: orm.SinglefileData | None = None,
    mp_correction: bool = False,
    eps_inf: float = 1.0,
    overrides: KoopmansDSCFOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> ScreeningParametersOutputs:
    """Multi-iteration alpha refinement: trial KI → per-orbital DSCF.

    Computes the per-orbital screening parameters by running one or
    more DSCF iterations. The *final* KI calculation (which applies
    those parameters to produce a single converged Koopmans-corrected
    spectrum) lives in :func:`KoopmansDSCFWorkflow`, one level up —
    that's the application of the screening parameters, not part of
    computing them.

    The first iteration's trial KI restarts from ``dft_remote`` (the DFT
    init's ``remote_folder``) with the uniform ``initial_alpha`` guess and
    receives the KS-as-variational overlay (for ``init_orbitals='kohn-sham'``)
    or, for the periodic Wannier init, the folded ``evc_occupied{n}.dat``
    staging via ``initial_evc_occupied{1,2}``. Subsequent iterations restart
    from the previous iteration's trial KI save and consume the previous
    iteration's gathered alphas — no overlay / staging needed because the
    converged ``evc0N.dat`` is already in place.

    The iteration count is bounded by ``alpha_numsteps`` and the loop also
    short-circuits when ``max |dE - lambda| < alpha_conv_thr`` (1e-3 eV).

    The final KI restarts from the *last* iteration's trial KI
    ``remote_folder`` and uses the final gathered alphas; only the alphas
    differ from the trial pass.

    All per-orbital fan-out happens inside ``ScreeningIteration`` (via
    the nested ``ComputeOrbitalScreeningParameters`` graph); iterations 2..N run
    through the recursive :func:`RefineScreeningParameters`.
    """
    ki_overrides = overrides.get("ki") if overrides else None
    filled_overrides = overrides.get("dft_n_minus_1") if overrides else None
    empty_overrides_dict: dict[str, KcpNamelistOverrides | None] | None
    if overrides:
        empty_overrides_dict = {
            "dft_dummy": overrides.get("dft_n_plus_1_dummy"),
            "pz_print": overrides.get("pz_print"),
            "dft_n_plus_1": overrides.get("dft_n_plus_1"),
        }
    else:
        empty_overrides_dict = None

    # Uniform-``initial_alpha`` payload feeds the first iteration's trial
    # KI (and its empty-orbital ``pz_print``). Subsequent iterations
    # consume the previous iteration's gathered alphas via the recursive
    # ``RefineScreeningParameters`` below.
    initial_alphas = generate_alphas(
        alpha_guess=initial_alpha,
        nbnd=nbnd,
        nelup=nelup,
        neldw=neldw,
        spin_polarized=spin_polarized,
    )

    base = _kcp_base_inputs(
        structure,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
    )

    # KS overlay applies only on the iteration that consumes ``dft_remote``
    # directly (the parent DFT save still has the raw ``evc0N.dat`` from
    # the DFT init, not the trial KI's inner-loop minimum) — i.e. the
    # unrolled first iteration below. All subsequent iterations parent on
    # the previous trial KI and inherit its converged ``evc0N.dat`` via
    # the primary parent walk.
    ks_overlay: dict[str, str] | None = None
    if init_orbitals == VariationalOrbitalType.KOHN_SHAM:
        nspin_overlay_iter = (1, 2) if nspin == 2 else (1,)
        # Stems only — the CalcJob appends ``.dat`` at submission time
        # (AiiDA's attribute store rejects Dict keys containing ``.``).
        ks_overlay = {
            **{f"evc{i}": f"evc0{i}" for i in nspin_overlay_iter},
            **{f"evc_empty{i}": f"evc0_empty{i}" for i in nspin_overlay_iter},
        }

    # ------------------------------------------------------------------
    # Alpha-refinement. The first iteration is unrolled here (it differs
    # from the rest: parent is ``dft_remote``, it carries the KS overlay,
    # and ``is_first_iteration=True`` drives KIPZ's one-off inner-loop CG
    # pass). Iterations 2..N run through the recursive
    # ``RefineScreeningParameters``, which receives the previous iteration's
    # outputs as concrete inputs and stops on convergence
    # (``max_error < alpha_conv_thr``) or budget exhaustion.
    #
    # TRIPWIRE -- KIPZ caching: under ``Correction.KIPZ`` the n+/-1 DFT
    # alpha-step calculations are alpha-dependent: they consume the current
    # ``file_alpharef.txt`` via ``do_orbdep=True``. ``ScreeningIteration``
    # re-runs every iteration so this is implicitly correct, but a
    # future refactor that caches alpha-step calcs keyed off "DFT inputs
    # are alpha-independent" must opt out for KIPZ. Stay in sync with
    # ``_add_kipz_orbdep``.
    # ------------------------------------------------------------------
    iter_1 = ScreeningIteration(
        code=code,
        structure=structure,
        pseudos=pseudos,
        base=base,
        nbnd=nbnd,
        correction=correction,
        spin_polarized=spin_polarized,
        current_alphas=initial_alphas,
        parent_folder=dft_remote,
        # First trial: KIPZ on a molecular system needs an inner-loop CG
        # pass to converge the variational orbitals starting from KS-init.
        # Subsequent iterations restart from the previous trial's already-
        # converged variational basis; see ``is_first_iteration=False``
        # inside ``RefineScreeningParameters`` and ``_build_orbdep_parameters``'s
        # ``do_innerloop`` decision.
        is_first_iteration=True,
        self_hartree_tol=self_hartree_tol,
        variational_orbital_overlays=ks_overlay,
        initial_evc_occupied1=initial_evc_occupied1,
        initial_evc_occupied2=initial_evc_occupied2,
        mp_correction=mp_correction,
        eps_inf=eps_inf,
        ki_overrides=ki_overrides,
        filled_overrides=filled_overrides,
        empty_overrides_dict=empty_overrides_dict,
        options=options,
    )

    # ``alpha_numsteps == 1`` needs no refinement beyond iter_1 — skip the
    # recursion node entirely so the graph matches the single-iteration
    # shape exactly.
    if alpha_numsteps <= 1:
        return {
            "alphas": iter_1["alphas"],
            "trial_remote": iter_1["trial_remote"],
        }

    refinement = RefineScreeningParameters(
        code=code,
        structure=structure,
        pseudos=pseudos,
        base=base,
        nbnd=nbnd,
        correction=correction,
        spin_polarized=spin_polarized,
        prev_alphas=iter_1["alphas"],
        prev_trial_remote=iter_1["trial_remote"],
        prev_max_error=iter_1["max_error"],
        # iter_1 already ran, so the recursion budget is one less.
        remaining_steps=alpha_numsteps - 1,
        alpha_conv_thr=alpha_conv_thr,
        self_hartree_tol=self_hartree_tol,
        mp_correction=mp_correction,
        eps_inf=eps_inf,
        ki_overrides=ki_overrides,
        filled_overrides=filled_overrides,
        empty_overrides_dict=empty_overrides_dict,
        options=options,
        metadata={"call_link_label": "refine_screening_parameters"},
    )

    # The final KI (application of the converged screening parameters)
    # lives in ``KoopmansDSCFWorkflow``; here we just return the screening
    # parameters and the parent save the final KI should restart from.
    return {
        "alphas": refinement["alphas"],
        "trial_remote": refinement["trial_remote"],
    }


# ----------------------------------------------------------------------
# Scope enforcement
# ----------------------------------------------------------------------


def _validate_scope(
    *,
    correction: Correction,
    init_orbitals: VariationalOrbitalType,
    fix_spin_contamination: bool,
    structure: orm.StructureData,
    blocks: list | None = None,
    kgrid: list[int] | None = None,
    kpoints: orm.KpointsData | None = None,
    codes: Codes | None = None,
) -> None:
    """Fail fast on inputs the workflow cannot honour yet.

    Two initialisation routes are supported: molecular Kohn-Sham
    (``init_orbitals='kohn-sham'``, non-periodic) and periodic Wannier
    (``init_orbitals in ('mlwfs', 'projwfs')``, which additionally needs
    the wannierisation inputs ``blocks`` / ``kgrid`` / ``kpoints`` /
    ``codes``). Everything else raises.
    """
    supported = {Correction.KI, Correction.KIPZ}
    if correction not in supported:
        raise NotImplementedError(
            f"correction={correction!r} not yet supported. "
            f"Supported corrections: {sorted(c.value for c in supported)}. "
            "PKIPZ requires a perturbative post-processing pass on top of a "
            "KI trial; NONE / ALL are workflow-control flags not consumed here."
        )
    if fix_spin_contamination:
        raise NotImplementedError(
            "fix_spin_contamination=True is not yet supported: it requires a "
            "spin-symmetrisation pre-pass (a dedicated SpinSymmetrizeTask) "
            "that has not been written yet."
        )

    wannier_init = init_orbitals in (
        VariationalOrbitalType.MLWFS,
        VariationalOrbitalType.PROJWFS,
    )
    periodic = any(structure.pbc)
    if wannier_init:
        if not periodic:
            raise ValueError(
                f"init_orbitals={init_orbitals!r} requires a periodic structure — "
                "Wannierisation is only defined for extended systems."
            )
        required = {"blocks": blocks, "kgrid": kgrid, "kpoints": kpoints, "codes": codes}
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                f"init_orbitals={init_orbitals!r} needs the wannierisation inputs "
                f"{missing} (projection blocks, the Monkhorst-Pack grid, the "
                "explicit k-mesh, and the pw/wannier90/pw2wannier90/wann2kcp/"
                "merge_evc codes)."
            )
    elif init_orbitals != VariationalOrbitalType.KOHN_SHAM:
        raise NotImplementedError(
            f"init_orbitals={init_orbitals!r} not yet supported. Supported: "
            f"'kohn-sham' (molecular), 'mlwfs' / 'projwfs' (periodic). "
            "PZ initialisation requires a pz_innerloop_init step."
        )
    elif periodic:
        raise NotImplementedError(
            "init_orbitals='kohn-sham' on a periodic structure is not yet "
            "supported — it needs a pw.x-only wannierize pass plus a "
            "ks2kcp folding mode. Use init_orbitals='mlwfs' / 'projwfs' "
            "for periodic systems."
        )


# ----------------------------------------------------------------------
# Parameter builders
# ----------------------------------------------------------------------


def _swap_kcp_frame(
    base: KcpBaseInputs,
    *,
    fixed_band: int,
    nbup: int,
    nbdw: int,
) -> tuple[KcpBaseInputs, int]:
    """Apply the spin-channel swap to a kcp.x input frame.

    The mutations are:

    * swap ``nelup <-> neldw``;
    * negate ``tot_magnetization`` (if not None);
    * shift ``fixed_band``: if it points into the up block
      (``fixed_band <= nbup``) it becomes ``fixed_band + nbdw``,
      otherwise it becomes ``fixed_band - nbup``.

    ``nbup`` / ``nbdw`` are the number of bands per spin block (``nelup`` /
    ``neldw`` when ``nbnd is None``, else ``nbnd`` for both). Callers
    should pass the values that match the calc they're building (this
    helper does not second-guess that choice).

    Returns ``(swapped_base, swapped_fixed_band)``. The original ``base``
    is not mutated (KcpBaseInputs is a frozen dataclass).
    """
    swapped = replace(
        base,
        nelup=base.neldw,
        neldw=base.nelup,
        tot_magnetization=(-base.tot_magnetization if base.tot_magnetization is not None else None),
    )
    if fixed_band > nbup:
        new_fixed = fixed_band - nbup
    else:
        new_fixed = fixed_band + nbdw
    return swapped, new_fixed


def _spin_swap_save_overlay(*, nspin: int) -> dict[str, str]:
    """``variational_orbital_overlays`` payload that flips spin-tagged save files.

    Returns ``{"evc02": "evc01", "evc01": "evc02", "evc_empty2": "evc_empty1",
    "evc_empty1": "evc_empty2", ...}`` -- stem-only (no ``.dat`` suffix; the
    CalcJob appends it). Used by ``ComputeEmptyOrbitalScreeningParameter`` when the N+1
    sub-runs run in the swapped frame but their parent (the trial KI's
    ``RemoteData``) is in the physical frame: each spin-tagged save file
    needs to be presented to kcp.x with its spin index flipped.

    Returns an empty dict for ``nspin == 1`` (no swap meaningful).
    """
    if nspin != 2:
        return {}
    pairs = [
        ("evc01", "evc02"),
        ("evc02", "evc01"),
        ("evc_empty1", "evc_empty2"),
        ("evc_empty2", "evc_empty1"),
        ("evc0_empty1", "evc0_empty2"),
        ("evc0_empty2", "evc0_empty1"),
    ]
    return dict(pairs)


def _kcp_base_inputs(
    structure: orm.StructureData,
    *,
    nspin: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    tot_magnetization: int | None,
    ecutwfc: float,
    ecutrho: float,
) -> KcpBaseInputs:
    """Assemble the shared :class:`KcpBaseInputs` payload from a structure.

    ``mt_correction`` and ``ntyp`` are derived from the structure
    (``not any(structure.pbc)`` and ``len(structure.kinds)``); the rest
    pass through unchanged. Callers in the workgraph layer build this
    once per step group and forward it to each ``_build_*`` invocation.
    """
    return KcpBaseInputs(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=not any(structure.pbc),
        ntyp=len(structure.kinds),
    )


def _build_dft_parameters(
    base: KcpBaseInputs,
    *,
    nbnd: int,
    restart_mode: str = "from_scratch",
    outerloop: bool = True,
) -> dict[str, Any]:
    """Parameter dict for the DFT initialization step.

    Args:
        base: shared cell-/basis-/electron-count inputs.
        nbnd: total number of bands (filled + empty). Pass ``0`` for
            alpha-step builders which strip it after composition.
        restart_mode: value of ``&CONTROL.restart_mode``. ``"from_scratch"``
            for the standard init; ``"restart"`` when chaining off a
            previous save (e.g. the final nspin=2 step in the closed-shell
            spin-symmetric 3-step init).
        outerloop: when ``False``, set ``do_outerloop=False`` and
            ``do_outerloop_empty=False`` and drop ``empty_states_maxstep``
            (the nspin=1/2 dummy init steps).
    """
    conv_thr = 1.0e-9 * base.nelec
    system: dict[str, Any] = {
        "ecutwfc": base.ecutwfc,
        "ecutrho": base.ecutrho,
        "nbnd": nbnd,
        "nspin": base.nspin,
        "do_ee": base.mt_correction,
        "do_orbdep": False,
        "fixed_state": False,
        "do_wf_cmplx": True,
        "nelec": base.nelec,
    }
    if base.nspin == 2:
        if base.nelup is not None:
            system["nelup"] = base.nelup
        if base.neldw is not None:
            system["neldw"] = base.neldw
        if base.tot_magnetization is not None:
            system["tot_magnetization"] = base.tot_magnetization
    # ``ndr`` and ``ndw`` are owned by the CalcJob (see
    # ``KcpCalculation._inject_owned_keys``) — the builders deliberately
    # leave them unset so there's only one source of truth.
    electrons: dict[str, Any] = {
        "electron_dynamics": "cg",
        "passop": 2.0,
        "ortho_para": 1,
        "maxiter": 300,
        "do_outerloop": outerloop,
        "do_outerloop_empty": outerloop,
        "conv_thr": conv_thr,
    }
    # ``empty_states_maxstep`` is only meaningful when the empty-manifold
    # outer loop runs; drop it when the loop is disabled.
    if outerloop:
        electrons["empty_states_maxstep"] = 300
    params: dict[str, Any] = {
        "CONTROL": {
            "calculation": "cp",
            "verbosity": "low",
            "iprint": 1,
            "disk_io": "high",
            "write_hr": False,
            "restart_mode": restart_mode,
        },
        "SYSTEM": system,
        "ELECTRONS": electrons,
        "IONS": {
            "ion_dynamics": "none",
            "ion_nstepe": 5,
            **{f"ion_radius({i + 1})": 1.0 for i in range(base.ntyp)},
        },
    }
    if base.mt_correction:
        params["EE"] = {"which_compensation": "tcc"}
    return params


_CORRECTION_TO_WHICH_ORBDEP = {
    Correction.KI: "nki",
    Correction.PZ: "pz",
    Correction.KIPZ: "nkipz",
}


def _build_orbdep_parameters(
    base: KcpBaseInputs,
    *,
    nbnd: int,
    correction: Correction,
    is_first_iteration: bool = False,
) -> dict[str, Any]:
    """Parameter dict for the trial / final / print step of an ODD-functional run.

    Routes KI / KIPZ / PZ through the same orbital-dependent
    screening machinery. ``correction`` selects ``&NKSIC.which_orbdep``
    via :data:`_CORRECTION_TO_WHICH_ORBDEP`
    (``KI`` → ``nki``, ``KIPZ`` → ``nkipz``, ``PZ`` → ``pz``);
    other :class:`Correction` members (``PKIPZ`` / ``NONE`` / ``ALL``)
    are workflow-level controls and raise here.

    ``is_first_iteration`` toggles ``do_innerloop`` on for KIPZ
    *molecular* trial calcs on the **first** alpha-loop iteration.
    KIPZ's variational orbitals shift with alpha, so the first trial
    (starting from KS-init
    orbitals) needs one CG inner-loop pass to converge them; later
    iterations restart from the previous trial's already-converged
    variational basis, so the inner loop is unnecessary.
    """
    params = _build_dft_parameters(base, nbnd=nbnd)
    # ``restart_mode`` is the only ``&CONTROL`` key the builder owns; ndr/ndw
    # are forced by the CalcJob (see ``_build_dft_parameters`` for context).
    params["CONTROL"]["restart_mode"] = "restart"

    # Orbital-dependent screening.
    params["SYSTEM"]["do_orbdep"] = True

    # The orbital-dependent SCF runs no outer loop and no inner loop
    # except for PZ.
    params["ELECTRONS"]["do_outerloop"] = False
    params["ELECTRONS"]["do_outerloop_empty"] = False
    # ``empty_states_maxstep`` is only meaningful when ``do_outerloop_empty``
    # is true; drop it when the empty-manifold loop is disabled.
    params["ELECTRONS"].pop("empty_states_maxstep", None)

    if correction not in _CORRECTION_TO_WHICH_ORBDEP:
        raise ValueError(
            f"Unsupported correction {correction!r} for ODD parameter build; "
            f"expected one of {set(_CORRECTION_TO_WHICH_ORBDEP)}"
        )
    which_orbdep = _CORRECTION_TO_WHICH_ORBDEP[correction]

    # PZ always wants the inner loop on. KIPZ wants it on for the first
    # trial on molecular systems — orbitals start from KS / PZ-init and
    # need one CG inner-loop pass to converge to the KIPZ-consistent
    # variational basis; subsequent iterations restart from the previous
    # trial's already-converged orbitals. ``base.mt_correction`` is True
    # iff non-PBC (``not any(structure.pbc)``).
    do_innerloop = correction == Correction.PZ or (
        correction == Correction.KIPZ and is_first_iteration and base.mt_correction
    )

    params["NKSIC"] = {
        "which_orbdep": which_orbdep,
        "odd_nkscalfact": True,
        "odd_nkscalfact_empty": True,
        "nkscalfact": 1.0,
        "do_innerloop": do_innerloop,
        "do_innerloop_empty": False,
        "do_innerloop_cg": True,
        "innerloop_cg_nreset": 20,
        "innerloop_cg_nsd": 2,
        "innerloop_init_n": 3,
        "innerloop_nmax": 100,
        "hartree_only_sic": False,
        "esic_conv_thr": 1.0e-9 * base.nelec,
        "do_bare_eigs": True,
    }
    return params


# ----------------------------------------------------------------------
# Delta-SCF alpha-refinement sub-step builders
# ----------------------------------------------------------------------
#
# These render the kcp.x inputs for the per-orbital sub-runs that compute
# alpha screening parameters via Delta SCF. Step list:
#
#   filled orbital → ``dft_n-1``
#   empty  orbital → ``dft_n+1_dummy`` (iter 1 only) → ``pz_print`` → ``dft_n+1``
#
# Common deltas vs ``_build_dft_parameters``:
# - ``nbnd`` removed.
# - ``conv_thr`` and ``esic_conv_thr`` 100x looser.
# - ``empty_states_maxstep`` / ``do_outerloop_empty`` removed (no empty
#   manifold treatment in these single-orbital runs).
# - ``&NKSIC`` always present (even when ``do_orbdep=False``) carrying
#   the shared inner-loop convergence knobs.

_LOOSE_CONV_FACTOR = 100.0  # ``conv_thr *= 100`` for alpha-loop sub-runs


def _alpha_step_lite_nksic(
    *, conv_thr: float, index_empty_to_save: int | None = None
) -> dict[str, Any]:
    """Minimal ``&NKSIC`` block emitted on every alpha-loop step.

    All keys here appear in every alpha-loop ``.cpi`` regardless of
    ``do_orbdep``; ``index_empty_to_save`` is set only on the
    empty-orbital sub-runs.
    """
    nksic: dict[str, Any] = {
        "do_innerloop": False,
        "do_innerloop_cg": True,
        "innerloop_cg_nreset": 20,
        "innerloop_cg_nsd": 2,
        "innerloop_init_n": 3,
        "innerloop_nmax": 100,
        "hartree_only_sic": False,
        "esic_conv_thr": conv_thr,
    }
    if index_empty_to_save is not None:
        nksic["index_empty_to_save"] = index_empty_to_save
    return nksic


def _alpha_step_dft_base(base: KcpBaseInputs) -> dict[str, Any]:
    """``&CONTROL/SYSTEM/ELECTRONS`` skeleton shared by every DFT-like alpha step.

    Built from ``_build_dft_parameters`` then trimmed: ``nbnd`` dropped,
    ``conv_thr`` loosened, empty-manifold knobs removed.
    """
    params = _build_dft_parameters(base, nbnd=0)  # nbnd stripped below
    params["SYSTEM"].pop("nbnd", None)
    params["ELECTRONS"].pop("empty_states_maxstep", None)
    params["ELECTRONS"].pop("do_outerloop_empty", None)
    params["ELECTRONS"]["conv_thr"] *= _LOOSE_CONV_FACTOR
    return params


def _add_kipz_orbdep(params: dict) -> None:
    """Turn a DFT-skeleton alpha step into a KIPZ one (in place).

    For KIPZ the n-1 / n+1 alpha-loop steps run with orbital-dependent
    screening on (``do_orbdep=True``, ``which_orbdep='nkipz'``), unlike
    KI's plain DFT n-1 / n+1. The shared "lite" NKSIC block is extended
    with the screening knobs the trial KIPZ carries.

    This implies KIPZ's DFT-step results are alpha-*dependent* and
    **must** be re-run on every alpha iteration (the current port already
    does, with no caching layer): any future alpha-independent calc reuse
    must be gated on ``functional != 'kipz'``.
    """
    params["SYSTEM"]["do_orbdep"] = True
    params["NKSIC"].update(
        {
            "which_orbdep": "nkipz",
            "odd_nkscalfact": True,
            "odd_nkscalfact_empty": True,
            "nkscalfact": 1.0,
            "do_bare_eigs": True,
        }
    )


def _build_n_minus_1_parameters(
    base: KcpBaseInputs,
    *,
    fixed_band: int,
    correction: Correction = Correction.KI,
) -> dict[str, Any]:
    """Filled-orbital N-1 step (``dft_n-1`` / ``kipz_n-1``).

    Plain DFT for ``Correction.KI``; gains orbital-dependent
    screening (``do_orbdep=True``, ``nkipz``) for ``Correction.KIPZ``.
    Restarts from the trial save.
    """
    params = _alpha_step_dft_base(base)
    params["CONTROL"]["restart_mode"] = "restart"
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["SYSTEM"]["f_cutoff"] = 1.0e-5
    params["SYSTEM"]["fixed_state"] = True
    # ``do_outerloop`` already True from the DFT base.
    params["NKSIC"] = _alpha_step_lite_nksic(conv_thr=params["ELECTRONS"]["conv_thr"])
    if correction == Correction.KIPZ:
        _add_kipz_orbdep(params)
    return params


def _build_dft_n_plus_1_dummy_parameters(
    base: KcpBaseInputs,
    *,
    fixed_band: int,
    index_empty_to_save: int = 1,
) -> dict[str, Any]:
    """``dft_n+1_dummy`` step: scratch DFT with one electron *added*.

    Run only on the first iteration of the alpha loop, once per *empty*
    orbital. Sets up the save-directory layout that ``pz_print`` and
    ``dft_n+1`` consume on subsequent steps.

    Caller must pass a ``base`` already incremented for the N+1 charge
    state (spin-up gets the extra electron) — i.e. ``nelec += 1`` and
    ``nelup += 1`` relative to the trial-KI base.
    """
    params = _alpha_step_dft_base(base)
    params["CONTROL"]["restart_mode"] = "from_scratch"
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["SYSTEM"]["fixed_state"] = False
    params["ELECTRONS"]["do_outerloop"] = False
    params["NKSIC"] = _alpha_step_lite_nksic(
        conv_thr=params["ELECTRONS"]["conv_thr"], index_empty_to_save=index_empty_to_save
    )
    return params


def _build_n_plus_1_parameters(
    base: KcpBaseInputs,
    *,
    fixed_band: int,
    index_empty_to_save: int = 1,
    correction: Correction = Correction.KI,
) -> dict[str, Any]:
    """Empty-orbital N+1 step (``dft_n+1`` / ``kipz_n+1``).

    Restarts from ``dft_n+1_dummy`` plus the print step's
    ``evcfixed_empty.dat``; caller stages both files into the work dir.
    """
    params = _alpha_step_dft_base(base)
    params["CONTROL"]["restart_mode"] = "restart"
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["SYSTEM"]["f_cutoff"] = 1.0
    params["SYSTEM"]["restart_from_wannier_pwscf"] = True
    params["SYSTEM"]["fixed_state"] = True
    params["NKSIC"] = _alpha_step_lite_nksic(
        conv_thr=params["ELECTRONS"]["conv_thr"], index_empty_to_save=index_empty_to_save
    )
    if correction == Correction.KIPZ:
        _add_kipz_orbdep(params)
    return params


def _build_print_parameters(
    base: KcpBaseInputs,
    *,
    nbnd: int,
    fixed_band: int,
    index_empty_to_save: int = 1,
    correction: Correction = Correction.KI,
) -> dict[str, Any]:
    """Empty-orbital print step (``pz_print`` / ``kipz_print``).

    Writes ``evcfixed_empty{ispin}.dat`` for the n+1 step. Runs at
    the *original* electron count (same nelec / nelup / neldw as the
    trial KI); only ``fixed_band`` differs.
    """
    # Build the PZ-flavour ODD skeleton (sets ``do_innerloop=True``
    # which we override below); the actual ``which_orbdep`` is
    # rewritten on the next line so the print step matches the
    # caller's KI / KIPZ workflow.
    params = _build_orbdep_parameters(base, nbnd=nbnd, correction=Correction.PZ)
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["NKSIC"]["which_orbdep"] = "nkipz" if correction == Correction.KIPZ else "pz"
    params["NKSIC"]["print_wfc_anion"] = True
    params["NKSIC"]["index_empty_to_save"] = index_empty_to_save
    # The print step's only job is to write ``evcfixed_empty{ispin}.dat``;
    # it operates on already-converged orbitals from the trial-KI save and
    # must NOT run the inner-loop SCF (which the PZ skeleton turns on by
    # default). Leaving it on makes kcp.x run a full PZ inner-CG cycle,
    # ballooning the print step from ~1 second to ~20 minutes.
    params["NKSIC"]["do_innerloop"] = False
    return params


# ----------------------------------------------------------------------
# Intentionally deferred:
# ----------------------------------------------------------------------
#
# 1. **pKIPZ.** Perturbative post-processing pass on top of a KI trial;
#    ``_validate_scope`` rejects it.
#
# 2. **eps_inf='auto'** for the Makov-Payne correction. The correction
#    itself is implemented (``compute_alpha_from_dscf``; on by default
#    for the periodic Wannier-init route) but ``eps_inf`` must be given
#    numerically — the automatic ph.x-based calculation of the dielectric
#    constant is not yet wired into this route.
#
# 3. **Mixing across iterations** (``alpha_mixing``).
#
# 4. **alpha-independent calc reuse** across iterations. The ``dft_n-1``
#    results for filled orbitals don't depend on alpha, but every
#    iteration currently re-runs them. NOTE: any future caching must opt
#    out for KIPZ, whose alpha steps are alpha-*dependent* — see the
#    TRIPWIRE comment in ``ComputeScreeningParameters``.
#
# 5. **ML predict shortcut**: short-circuiting the loop using a
#    pre-trained ML model.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Shared CalcJob-input assembly
# ----------------------------------------------------------------------


def _fft_dimension_allowed(nr: int) -> bool:
    """QE's FFT-dimension rule: factors of 2/3/5 only (no 7s or 11s)."""
    if nr < 1:
        return False
    remainder = nr
    powers = {2: 0, 3: 0, 5: 0, 7: 0, 11: 0}
    for factor in powers:
        while remainder > 1 and remainder % factor == 0:
            remainder //= factor
            powers[factor] += 1
    return remainder == 1 and powers[7] == 0 and powers[11] == 0


def _good_fft(nr: int) -> int:
    """Bump ``nr`` up to the next FFT-friendly dimension (legacy ``good_fft``)."""
    while not _fft_dimension_allowed(nr) and nr <= 2049:
        nr += 1
    return nr


def _autogenerate_nrb(
    structure: orm.StructureData,
    pseudos: dict[str, UpfData],
    parameters: dict[str, Any],
) -> None:
    """Fill ``SYSTEM.nr{1,2,3}b`` when any pseudo carries core corrections.

    Port of legacy ``_koopmans_cp.py:_autogenerate_nr``: kcp.x aborts with
    "nr1b, nr2b, nr3b must be given for ultrasoft and core corrected pp"
    when a pseudo has non-linear core corrections and the small-box grid is
    unset (bites e.g. PseudoDojo; SG15 has no NLCC). Same conservative
    guess as legacy: the full density-grid dimensions scaled by
    ``2 * rc_safe / L_i`` with ``rc_safe = 3`` Bohr (every PseudoDojo
    cutoff radius is <= 2.6 Bohr). User-supplied values always win.
    """
    from qe_tools import CONSTANTS
    from upf_to_json import upf_to_json

    system = parameters.setdefault("SYSTEM", {})
    if all(system.get(key) is not None for key in ("nr1b", "nr2b", "nr3b")):
        return

    def _core_corrected(pseudo: UpfData) -> bool:
        try:
            header = upf_to_json(pseudo.get_content(), pseudo.filename)["pseudo_potential"][
                "header"
            ]
        except Exception:
            # Unparseable UPF (e.g. minimal test fixtures): treat as no-NLCC.
            # Not a silent-corruption risk — a real core-corrected pseudo that
            # slips through makes kcp.x abort loudly with its own
            # "nr1b, nr2b, nr3b must be given" error.
            return False
        return bool(header["core_correction"])

    if not any(_core_corrected(pseudo) for pseudo in pseudos.values()):
        return

    angstrom_to_bohr = 1.0 / CONSTANTS.bohr_to_ang
    cell = np.array(structure.cell, dtype=float)
    alat_bohr = float(np.linalg.norm(cell[0])) * angstrom_to_bohr
    # Reduced lattice vectors ("at" in QE), dimensionless in units of alat.
    at = cell * angstrom_to_bohr / alat_bohr

    ecutrho = float(system.get("ecutrho") or 4.0 * system["ecutwfc"])
    # Density-grid dimensions, as QE derives them:
    # nr_i = 2 * int( sqrt(ecutrho) / (2 pi / alat) * |at_i| ) + 1
    nr = [
        _good_fft(2 * int(np.sqrt(ecutrho) / (2.0 * np.pi / alat_bohr) * np.linalg.norm(vec)) + 1)
        for vec in at
    ]
    rc_safe = 3.0
    for key, vec, nr_i in zip(("nr1b", "nr2b", "nr3b"), at, nr):
        system[key] = _good_fft(int(nr_i * 2.0 * rc_safe / (np.linalg.norm(vec) * alat_bohr)))


def _build_kcp_inputs(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    parameters: dict[str, Any],
    pseudos: dict[str, UpfData],
    *,
    options: dict[str, Any] | None = None,
    alphas: AlphaScreening | None = None,
    parent_folder: orm.RemoteData | None = None,
    parent_folder_evcfixed: orm.RemoteData | None = None,
    variational_orbital_overlays: dict[str, str] | None = None,
    read_wavefunctions: dict[str, Any] | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Assemble a kwargs dict for ``KcpStep(**inputs)``.

    Plain Python data (the ``parameters`` dict, the ``alphas``
    TypedDict) is handed straight through; aiida-workgraph's
    serialization adapter wraps each value into the matching AiiDA
    Node when the underlying CalcJob socket is set.

    ``name`` becomes ``metadata.call_link_label`` on the resulting CalcJob —
    that's what shows up in ``verdi process list`` and the koopmans progress
    table (e.g. ``kcp-dft_init`` instead of ``kcp-KcpCalculation``).

    Inside the per-orbital screening sub-graphs, ``name`` is set statically
    (e.g. ``"dft_n_minus_1"``, ``"pz_print"``, ``"dft_n_plus_1_dummy"``,
    ``"dft_n_plus_1"``); the band/spin identity lives on the *wrapping*
    sub-graph's ``call_link_label`` instead (``compute_alpha_<map_key>``, set by
    the ``ComputeOrbitalScreeningParameters`` fan-out loop), so provenance reads as e.g.
    ``compute_alpha_up_orb_2 -> dft_n_minus_1``.

    ``parent_folder_evcfixed`` is the ``RemoteData`` of a ``pz_print``
    run; only the ``dft_n+1`` step of the empty-orbital Delta-SCF branch
    needs this. The CalcJob symlinks the file
    ``out/<prefix>_<NDW>.save/K00001/evcfixed_empty.dat`` from that
    folder onto its read save (see
    ``KcpCalculation._build_remote_symlink_list``).

    ``read_wavefunctions`` maps destination stems to the
    ``SinglefileData`` (or socket) holding the wavefunction; the CalcJob
    copies each into its read ``K00001`` as ``<stem>.dat`` (the MLWF-init
    staging of the folded ``evc_occupied{n}.dat`` / ``evc0_empty{n}.dat``
    merge outputs).
    """
    _autogenerate_nrb(structure, pseudos, parameters)
    inputs: dict[str, Any] = {
        "code": code,
        "structure": structure,
        "parameters": parameters,
        "pseudos": pseudos,
    }
    if alphas is not None:
        inputs["alphas"] = alphas
    if parent_folder is not None:
        inputs["parent_folder"] = parent_folder
    if parent_folder_evcfixed is not None:
        inputs["parent_folder_evcfixed"] = parent_folder_evcfixed
    if variational_orbital_overlays:
        inputs["variational_orbital_overlays"] = orm.Dict(dict=variational_orbital_overlays)
    if read_wavefunctions:
        inputs["read_wavefunctions"] = read_wavefunctions
    metadata: dict[str, Any] = {}
    if options:
        metadata["options"] = options
    if name:
        metadata["call_link_label"] = name
    if metadata:
        inputs["metadata"] = metadata
    return inputs
