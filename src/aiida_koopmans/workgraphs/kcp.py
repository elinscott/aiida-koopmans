"""aiida-workgraph builders for kcp.x.

Wraps :class:`~aiida_koopmans.calculations.kcp.KcpCalculation` as a task and
composes it into higher-level workflows.

**Current scope (MVP).** Only the minimum path needed for tutorial_1 (ozone,
KI + DSCF + kohn-sham init, molecular / non-periodic, alpha_numsteps=1) is
implemented. All other branches of the legacy ``KoopmansDSCFWorkflow`` raise
``NotImplementedError`` at build time with a clear message.

The MVP ``KoopmansDSCFTask`` executes **two** kcp.x calls:

1. DFT initialization (``do_orbdep=False``, nspin=2, from scratch)
2. KI final (``do_orbdep=True``, ``which_orbdep='nki'``, restart from step 1,
   initial alphas = ``initial_alpha`` for every orbital)

The legacy implementation for the same inputs executes 20 kcp.x calls
including spin-symmetrization (7), a trial KI pass (1), a Delta SCF loop to
compute the alphas (12), and a final KI (1). Porting the Delta SCF alpha loop
and spin-symmetrization is deferred to later phases — the code below is
structured so those extensions can slot in as additional ``@task.graph``
helpers without reshaping the public ``KoopmansDSCFTask`` signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import numpy as np
from aiida import orm
from aiida.plugins import DataFactory
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_workgraph import Map, dynamic, task

from aiida_koopmans.calculations.kcp import KcpCalculation
from aiida_koopmans.types import AlphaScreening, SpinChannel
from aiida_koopmans.utils import (
    count_electrons_task,
    filled_and_empty_counts,
    filled_and_empty_counts_task,
    resolve_pseudo_family_task,
)
from aiida_koopmans.workgraphs.convert_spin import convert_spin1_to_spin2

UpfData = DataFactory("pseudo.upf")


# ----------------------------------------------------------------------
# Output / override typing
# ----------------------------------------------------------------------
#
# Annotations declare what consumer pyfunctions see *after* aiida-pythonjob's
# auto-deserialization: ``orm.Dict → dict``, single-key ``orm.ArrayData →
# np.ndarray``. Lambdas/bare-lambdas come from the parser as a single
# stacked ``(nspin, n, n)`` matrix (see ``KcpParser._parse_lambdas``); index
# axis-0 by ``SpinChannel.index``. ``remote_folder`` stays as
# ``orm.RemoteData`` because downstream ``parent_folder`` sockets take the
# node, not its payload.


class DFTCPOutputs(TypedDict):
    """Outputs of a single kcp.x DFT-only run."""

    parameters: dict
    eigenvalues: np.ndarray
    remote_folder: orm.RemoteData


class KoopmansDSCFOutputs(TypedDict):
    """Outputs of the KI correction step (the final result of a KI-DSCF workflow)."""

    parameters: dict
    eigenvalues: np.ndarray
    lambdas: np.ndarray
    bare_lambdas: np.ndarray
    remote_folder: orm.RemoteData


@dataclass(frozen=True)
class KcpBaseInputs:
    """Cell-, basis-, and electron-count inputs shared by every kcp.x step.

    Each parameter builder takes one ``KcpBaseInputs`` (built once per
    workgraph from ``structure`` + electron-count outputs) plus its
    step-specific kwargs. Collapsing this set into a single object
    removes the 9-kwarg forwarding boilerplate that otherwise repeats
    at every builder call site. ``nbnd`` is intentionally *not* here —
    DFT/KI/PZ steps need it but alpha-step (dft_n±1) builders strip it,
    so it stays a step-level kwarg.

    A frozen ``dataclass`` rather than a ``TypedDict``: aiida-workgraph
    routes dataclass-typed sockets through ``structured_to_dict`` →
    ``dataclasses.asdict`` (which preserves ``None`` fields) and
    reconstructs via ``cls(**value)`` on the receiving side. Plain
    ``dict`` / ``TypedDict`` sockets, by contrast, silently strip
    ``None``-valued entries in transit — closed-shell tutorial_1
    (``tot_magnetization=None``) hit exactly that failure mode before
    this migration.
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
    the MVP defaults via ``aiida_quantumespresso.workflows.protocols.utils.recursive_merge``.
    """

    CONTROL: dict[str, Any]
    SYSTEM: dict[str, Any]
    ELECTRONS: dict[str, Any]
    IONS: dict[str, Any]
    CELL: dict[str, Any]
    EE: dict[str, Any]
    NKSIC: dict[str, Any]


class KoopmansDSCFOverrides(TypedDict, total=False):
    """Per-step overrides for ``KoopmansDSCFTask``.

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
    """Gathered outputs of the per-orbital scatter, packed into per-spin lists.

    ``alphas`` and ``errors`` share the same per-spin / filled-vs-empty
    layout (see :class:`AlphaScreening`), making the ``alphas`` field
    drop-in for the kcp.x ``alphas`` socket on the final KI step.
    """

    alphas: AlphaScreening
    errors: AlphaScreening


class OneDSCFIterationOutputs(TypedDict):
    """Outputs of one alpha-refinement iteration (trial KI + per-orbital DSCF).

    Used to thread the next iteration's inputs through the ``While`` loop
    (alpha refinement) without going through ``wg.ctx``:

    * ``alphas`` — gathered per-orbital screening parameters; becomes the
      next iteration's trial-KI ``alphas`` input.
    * ``errors`` — gathered ``|dE - lambda|`` per orbital; retained for
      diagnostics / convergence reporting.
    * ``trial_remote`` — the trial KI's ``remote_folder``; becomes the
      next iteration's ``parent_folder`` (and, after the loop, the final
      KI's parent).
    * ``max_error`` — convergence indicator; the loop terminates when
      this falls below the legacy ``1e-3 eV`` threshold (legacy
      ``_koopmans_dscf.py:633``).
    """

    alphas: AlphaScreening
    errors: AlphaScreening
    trial_remote: orm.RemoteData
    max_error: float


# ----------------------------------------------------------------------
# Raw CalcJob as a workgraph task
# ----------------------------------------------------------------------

KcpBaseTask = task(KcpCalculation)


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
) -> dict:
    """Compute the new alpha for one orbital from its Delta-SCF perturbed run.

    Implements equation 10 of Nguyen et al. (2018) 10.1103/PhysRevX.8.021051,
    matching legacy ``_koopmans_dscf.py:944``::

        alpha_new = alpha_guess * (dE - lambda_0) / (lambda_a - lambda_0)

    where:

    - ``dE = E_trial - E_dft_n-1`` for filled orbitals,
      ``dE = E_dft_n+1 - E_trial`` for empty orbitals;
    - ``lambda_a`` is the diagonal element of the trial KI's
      orbital-dependent Hamiltonian at ``(band_index, band_index)``;
    - ``lambda_0`` is the same diagonal element of the **bare** Hamiltonian.

    Both energies and lambdas are in eV (the parser converts from Hartree),
    so the units cancel on division. ``error = |dE - lambda_a|`` is the
    convergence indicator the legacy loop monitors. The lambda arrays are
    stacked ``(nspin, n, n)``; ``spin_channel.index`` selects the spin axis.
    """
    trial_e = trial_output_parameters["energy"]
    perturbed_e = perturbed_output_parameters["energy"]
    spin = spin_channel.index
    lambda_a = float(trial_lambdas[spin, band_index, band_index].real)
    lambda_0 = float(trial_bare_lambdas[spin, band_index, band_index].real)
    dE = trial_e - perturbed_e if filled else perturbed_e - trial_e  # noqa: N806
    alpha_new = alpha_guess * (dE - lambda_0) / (lambda_a - lambda_0)
    error = abs(dE - lambda_a)
    return {"alpha": alpha_new, "error": error}


@task
def assemble_alpha_screening(
    *,
    filled_alphas: Annotated[dict | None, dynamic(float)] = None,
    filled_errors: Annotated[dict | None, dynamic(float)] = None,
    empty_alphas: Annotated[dict | None, dynamic(float)] = None,
    empty_errors: Annotated[dict | None, dynamic(float)] = None,
) -> _PerOrbitalAlphaOutputs:
    """Pack per-orbital scatter outputs into the :class:`AlphaScreening` shape.

    Inputs are flat dicts produced by the scatter helpers, keyed
    ``f"{spin_tag}_orb_{i}"`` where ``spin_tag`` is the
    :class:`SpinChannel` value (``"none"`` for closed-shell, or ``"up"``
    / ``"down"`` for the spin-polarised case) and ``i`` is the 0-indexed
    variational orbital within the per-channel manifold.

    The packer is purely structural — it pulls the channel set and the
    per-channel count from the input dict alone, with no knowledge of
    ``nspin`` or ``spin_polarized``. The closed-shell up→down mirror
    (when kcp.x needs both spin slots in ``file_alpharef``) lives in
    :meth:`KcpCalculation._write_alpha_files`, not here.

    Returns a ``{"alphas": AlphaScreening, "errors": AlphaScreening}``
    pair: the ``alphas`` field plugs straight into the kcp.x ``alphas``
    socket of the final KI step.
    """
    # Map zones with zero iterations may surface their gathered output
    # as ``None``; treat that as the empty dict.
    filled_alphas = filled_alphas or {}
    filled_errors = filled_errors or {}
    empty_alphas = empty_alphas or {}
    empty_errors = empty_errors or {}

    def _pack(flat: dict) -> dict[SpinChannel, list[float]]:
        if not flat:
            return {}
        # Discover the channel set from the input keys alone. Two shapes:
        # * ``"orb_<n>"`` — closed-shell representative channel (NONE).
        # * ``"<up|down>_orb_<n>"`` — spin-polarised.
        # Indices are 1-indexed and may be non-contiguous (filled and
        # empty manifolds share the same numbering — see
        # :func:`build_filled_iter_source` / :func:`build_empty_iter_source`).
        # Sort by index so output lists carry orbitals in band order.
        by_spin: dict[SpinChannel, list[tuple[int, float]]] = {}
        for key, value in flat.items():
            tag, _, idx_str = key.rpartition("_")
            if not tag.endswith("orb"):
                raise ValueError(f"Unexpected key {key!r}")
            spin_tag = tag[: -len("orb")].rstrip("_")
            spin = SpinChannel(spin_tag) if spin_tag else SpinChannel.NONE
            by_spin.setdefault(spin, []).append((int(idx_str), float(value)))
        out: dict[SpinChannel, list[float]] = {}
        for spin, items in by_spin.items():
            items.sort(key=lambda t: t[0])
            out[spin] = [v for _, v in items]
        return out

    filled_packed = _pack(filled_alphas)
    empty_packed = _pack(empty_alphas)
    filled_err_packed = _pack(filled_errors)
    empty_err_packed = _pack(empty_errors)

    alphas: AlphaScreening = {"filled": filled_packed, "empty": empty_packed}
    errors: AlphaScreening = {"filled": filled_err_packed, "empty": empty_err_packed}
    return {"alphas": alphas, "errors": errors}


@task
def max_alpha_error(filled_errors: dict, empty_errors: dict):
    """Convergence indicator for one DSCF iteration.

    Returns ``max |dE - lambda|`` across every per-orbital error in both
    branches. The alpha-refinement loop in ``KIDscfRefinementTask`` stops
    when this falls below ``1e-3 eV`` (legacy threshold from
    ``_koopmans_dscf.py:633``). ``filled_errors`` / ``empty_errors`` are
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
    n_filled: int,
    n_empty: int,
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
    * ``spin_polarized=True`` → emit both UP and DOWN with identical
      uniform values.

    ``n_filled`` / ``n_empty`` are totals across both spin channels;
    halve them once to get the per-channel count.
    """
    n_filled_per_channel = n_filled // 2
    n_empty_per_channel = n_empty // 2
    if not spin_polarized:
        return {
            "filled": {SpinChannel.NONE: [alpha_guess] * n_filled_per_channel},
            "empty": {SpinChannel.NONE: [alpha_guess] * n_empty_per_channel},
        }
    return {
        "filled": {
            SpinChannel.UP: [alpha_guess] * n_filled_per_channel,
            SpinChannel.DOWN: [alpha_guess] * n_filled_per_channel,
        },
        "empty": {
            SpinChannel.UP: [alpha_guess] * n_empty_per_channel,
            SpinChannel.DOWN: [alpha_guess] * n_empty_per_channel,
        },
    }


@task
def build_filled_iter_source(
    nbnd: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    filled_alphas: dict,
    spin_polarized: bool = False,
) -> Annotated[dict, dynamic(dict)]:
    """Materialise the per-orbital iterator for the *filled* Map zone.

    Each emitted item is a ``dict`` carrying the per-orbital parameters
    the consumer (``OrbitalDeltaSCFFilledTask``) needs: ``fixed_band``
    (1-indexed kcp.x), ``spin_channel`` (kept as the :class:`SpinChannel`
    enum), ``band_index`` (0-indexed numpy index into the stacked lambda
    matrices), and ``alpha_guess`` (per-orbital alpha already in use for
    that band on the current refinement iteration).

    ``filled_alphas`` is keyed by spin tag ("none" / "up" / "down" —
    matching :class:`SpinChannel`'s string values, which is what survives
    the AiiDA serializer round-trip) and maps to per-channel alpha lists.
    On iteration 1 the caller passes :func:`generate_alphas`'s uniform
    output; on subsequent iterations the previous iteration's gathered
    alphas (an :class:`AlphaScreening`'s ``filled`` half).

    Spin handling:

    * ``spin_polarized=False`` (closed shell, the common case): emit a
      single representative channel keyed by :attr:`SpinChannel.NONE`.
      Legacy convention is to solve the up-spin orbitals only and copy
      their alphas onto the down-spin twins (see
      ``koopmans/bands.py:298-301`` + ``to_solve`` at line 318-320). The
      mirror onto kcp.x's per-spin ``file_alpharef`` happens later in
      :meth:`KcpCalculation._write_alpha_files`.
    * ``spin_polarized=True``: emit both UP and DOWN channels.

    kcp.x always runs with ``nspin=2`` in our KI flow (the dispatcher
    hardcodes this); the closed-shell halving is a function of
    ``spin_polarized``, not ``nspin``.
    """
    n_filled, _ = filled_and_empty_counts(nspin=2, nbnd=nbnd, nelec=nelec, nelup=nelup, neldw=neldw)
    n_filled_per_channel = n_filled // 2
    spin_list = [SpinChannel.UP, SpinChannel.DOWN] if spin_polarized else [SpinChannel.NONE]
    out: dict[str, dict] = {}
    for spin in spin_list:
        alphas_for_spin = filled_alphas[spin.value]
        for i in range(n_filled_per_channel):
            # Orbital indices are **1-indexed** and shared with the empty
            # manifold (see :func:`build_empty_iter_source`) so they line up
            # with kcp.x's own band numbering. Closed-shell: bare
            # ``orb_<n>`` key (no spin tag — only one representative
            # channel). Spin-polarised: ``<up|down>_orb_<n>``. ``_pack`` in
            # :func:`assemble_alpha_screening` parses both shapes back to
            # :class:`SpinChannel`.
            orb_index = i + 1
            tag_prefix = "" if spin is SpinChannel.NONE else f"{spin.value}_"
            out[f"{tag_prefix}orb_{orb_index}"] = {
                "fixed_band": orb_index,
                "spin_channel": spin,
                "band_index": i,
                "alpha_guess": alphas_for_spin[i],
            }
    return out


@task
def build_empty_iter_source(
    nbnd: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    empty_alphas: dict,
    spin_polarized: bool = False,
) -> Annotated[dict, dynamic(dict)]:
    """Materialise the per-orbital iterator for the *empty* Map zone.

    Empty orbitals come *after* the filled manifold within each per-spin
    block, so ``fixed_band`` and ``band_index`` are offset by
    ``n_filled_per_channel``. ``index_empty_to_save`` is 1-based within
    the per-spin empty manifold (kcp.x convention).

    ``empty_alphas`` follows the same shape as ``filled_alphas`` in
    :func:`build_filled_iter_source`: keyed by spin tag, mapping to
    per-channel alpha lists indexed within the empty manifold (so index
    ``0`` is the first empty orbital, *not* the global band index).

    See :func:`build_filled_iter_source` for the spin-channel emission
    rule (closed-shell emits a single :attr:`SpinChannel.NONE` channel;
    spin-polarised emits UP + DOWN).
    """
    n_filled, n_empty = filled_and_empty_counts(
        nspin=2, nbnd=nbnd, nelec=nelec, nelup=nelup, neldw=neldw
    )
    n_filled_per_channel = n_filled // 2
    n_empty_per_channel = n_empty // 2
    spin_list = [SpinChannel.UP, SpinChannel.DOWN] if spin_polarized else [SpinChannel.NONE]
    out: dict[str, dict] = {}
    for spin in spin_list:
        alphas_for_spin = empty_alphas[spin.value]
        for i in range(n_empty_per_channel):
            # Empty orbital indices continue the filled-manifold numbering
            # (1-indexed, no restart at 0) — matches kcp.x's band ordering.
            # See :func:`build_filled_iter_source` for the spin-tag rule.
            orb_index = n_filled_per_channel + i + 1
            tag_prefix = "" if spin is SpinChannel.NONE else f"{spin.value}_"
            out[f"{tag_prefix}orb_{orb_index}"] = {
                "fixed_band": orb_index,
                "spin_channel": spin,
                "band_index": orb_index - 1,
                "index_empty_to_save": i + 1,
                "alpha_guess": alphas_for_spin[i],
            }
    return out


@task
def _get_value(data: dict, key: str):
    """Extract a single field from a Map item dict.

    Map items are dict-valued; AiiDA forbids direct subscripting of
    sockets, so each field accessed inside a Map zone goes through this
    one-shot task. See aiida-workgraph
    ``docs/gallery/advanced/autogen/context_manager.py`` for the
    canonical pattern.
    """
    return data[key]


# ----------------------------------------------------------------------
# Public graphs
# ----------------------------------------------------------------------


@task.graph
def DFTCPTask(
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
    spin-symmetric 3-step init chain (legacy
    ``restart_with_higher_precision``): an nspin=1 from-scratch run with
    outer loop, then an nspin=2 from-scratch *dummy* with the outer loop
    disabled, then the final nspin=2 restarted run.

    Args:
        restart_mode: forwarded to ``&CONTROL.restart_mode``. Pair with
            ``parent_folder`` when restarting.
        outerloop: when ``False``, disables the outer / empty-manifold
            outer loops (matches legacy nspin{1,2}_dummy_calculator).
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
    outputs = KcpBaseTask(**inputs)

    return DFTCPOutputs(
        parameters=outputs["output_parameters"],
        eigenvalues=outputs["output_eigenvalues"],
        remote_folder=outputs["remote_folder"],
    )


@task.graph
def KoopmansDSCFTask(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudo_family: str,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int = 2,
    tot_magnetization: int | None = None,
    functional: str = "ki",
    init_orbitals: str = "kohn-sham",
    alpha_numsteps: int = 1,
    fix_spin_contamination: bool = False,
    initial_alpha: float = 0.6,
    spin_polarized: bool = False,
    overrides: KoopmansDSCFOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> KoopmansDSCFOutputs:
    """Koopmans DSCF workflow — DFT init → trial KI → per-orbital DSCF refinement → final KI.

    Runs one iteration of alpha refinement: a trial KI computes lambda
    matrices for the alpha formula, the per-orbital Delta-SCF scatter
    (one ``dft_n-1`` for each filled orbital and one
    ``dft_n+1_dummy → pz_print → dft_n+1`` triplet for each empty
    orbital) refines every alpha, and a final KI re-runs with those
    refined alphas (restarting from the DFT save, not the trial KI).

    Multi-iteration refinement (``alpha_numsteps > 1``) and
    spin-symmetrisation (``fix_spin_contamination=True``) are deferred
    to Phase B; ``_validate_scope`` rejects those paths.
    """
    _validate_scope(
        functional=functional,
        init_orbitals=init_orbitals,
        alpha_numsteps=alpha_numsteps,
        fix_spin_contamination=fix_spin_contamination,
        structure=structure,
    )

    dft_overrides = overrides.get("dft") if overrides else None

    # Resolve pseudo family + electron counts once, at runtime, so the
    # results flow downstream as plain AiiDA-typed sockets instead of
    # ``TaggedValue`` proxies (the failure mode of the inline plain-Python
    # call inside a nested ``@task.graph`` body).
    pseudos = resolve_pseudo_family_task(
        family_label=pseudo_family,
        structure=structure,
    )
    counts = count_electrons_task(
        structure=structure,
        pseudos=pseudos,
        nspin=nspin,
        tot_magnetization=tot_magnetization,
    )
    nelec = counts["nelec"]
    nelup = counts["nelup"]
    neldw = counts["neldw"]

    if spin_polarized:
        # Spin-polarised systems are seeded directly from a single
        # nspin=2 from-scratch run; legacy treats the up/down channels as
        # independent and does not pre-symmetrise.
        dft = DFTCPTask(
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
    else:
        # Closed-shell spin-symmetric init chain (legacy
        # ``restart_with_higher_precision`` in
        # ``koopmans/workflows/_workflow.py:1602-1670``):
        #
        # 1. nspin=1 from scratch — converges the single-channel solution.
        # 2. nspin=2 from-scratch dummy — lays out the nspin=2 save tree
        #    skeleton, outer loop disabled so the wavefunction content is
        #    irrelevant (only the layout matters).
        # 3. ConvertSpin1ToSpin2 — splices step-1 wavefunctions into the
        #    step-2 save layout, producing a spin-symmetric save.
        # 4. nspin=2 restart — final init starting from the symmetrised
        #    save; this is the ``remote_folder`` consumed by the
        #    downstream KIDscfRefinementTask.
        dft_nspin1 = DFTCPTask(
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

        dft_nspin2_dummy = DFTCPTask(
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

        dft = DFTCPTask(
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

    return KIDscfRefinementTask(
        code=code,
        structure=structure,
        pseudos=pseudos,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        initial_alpha=initial_alpha,
        functional=functional,
        init_orbitals=init_orbitals,
        spin_polarized=spin_polarized,
        dft_remote=dft["remote_folder"],
        overrides=overrides,
        options=options,
    )


@task.graph
def OrbitalDeltaSCFFilledTask(
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
    overrides: KcpNamelistOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> OrbitalDeltaSCFOutputs:
    """Compute the new alpha for one **filled** orbital via Delta-SCF.

    Submits a single ``dft_n-1`` kcp.x run (with ``fixed_band=fixed_band``,
    one electron pulled out of that orbital via ``f_cutoff=1e-5``), then
    evaluates the legacy alpha formula at ``(spin_channel, band_index)``
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
    parameters = _build_dft_n_minus_1_parameters(base, fixed_band=fixed_band)
    if overrides:
        parameters = recursive_merge(parameters, overrides)

    inputs = _build_kcp_inputs(
        code,
        structure,
        parameters,
        pseudos,
        options=options,
        parent_folder=trial_remote,
        name="dft_n_minus_1",
    )
    dft_outputs = KcpBaseTask(**inputs)

    result = compute_alpha_from_dscf(
        trial_output_parameters=trial_output_parameters,
        perturbed_output_parameters=dft_outputs["output_parameters"],
        trial_lambdas=trial_lambdas,
        trial_bare_lambdas=trial_bare_lambdas,
        spin_channel=spin_channel,
        band_index=band_index,
        alpha_guess=alpha_guess,
        filled=True,
    )

    return OrbitalDeltaSCFOutputs(
        alpha=result["alpha"],
        error=result["error"],
    )


@task.graph
def OrbitalDeltaSCFEmptyTask(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int,
    nelec: int,
    fixed_band: int,
    spin_channel: SpinChannel,
    band_index: int,
    index_empty_to_save: int,
    alpha_guess: float,
    pz_alphas: AlphaScreening,
    trial_remote: orm.RemoteData,
    trial_output_parameters: dict,
    trial_lambdas: np.ndarray,
    trial_bare_lambdas: np.ndarray,
    nelup: int | None = None,
    neldw: int | None = None,
    tot_magnetization: int | None = None,
    overrides: KcpNamelistOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> OrbitalDeltaSCFOutputs:
    """Compute the new alpha for one **empty** orbital via Delta-SCF.

    Three-step kcp.x sub-pipeline (legacy
    ``02-calculate-screening-via-dscf/01-iteration-1/<orbital>/``):

    1. ``dft_n+1_dummy`` — scratch DFT with the empty orbital populated
       (``fixed_band`` + ``nelec=N+1``); writes the save layout the
       subsequent steps consume. Phase A always runs this every
       iteration; legacy reuses it from iteration 1 onwards (deferred).
    2. ``pz_print`` — PZ run on the fixed empty orbital (parent =
       ``trial_remote``) that writes ``evcfixed_empty.dat``.
    3. ``dft_n+1`` — SCF DFT (parent = the dummy + ``pz_print``'s
       ``evcfixed_empty.dat`` via ``parent_folder_evcfixed``).

    Then the alpha calcfunction extracts the (spin_channel, band_index)
    diagonal of the trial KI's lambda matrices, computes ``dE``, and
    returns the new alpha + convergence error.

    ``index_empty_to_save`` is the kcp.x index (1-based, within the
    empty manifold) used to identify which empty's wavefunction
    ``dft_n+1_dummy`` and ``pz_print`` save and ``dft_n+1`` reads.
    For systems with a single empty orbital (e.g. tutorial_1 ozone)
    this is always ``1``.

    ``pz_alphas`` is built upstream by :func:`generate_alphas` (a
    uniform :class:`AlphaScreening` of ``alpha_guess`` values) — no
    socket arithmetic happens inside this graph body.
    """
    dummy_overrides = overrides.get("dft_dummy") if overrides else None  # type: ignore[union-attr]
    pz_overrides = overrides.get("pz_print") if overrides else None  # type: ignore[union-attr]
    n_plus_1_overrides = overrides.get("dft_n_plus_1") if overrides else None  # type: ignore[union-attr]

    # Two ``base`` payloads: the N+1 charge state for dummy / n+1 (one
    # extra electron, conventionally placed on the up channel) and the
    # original N-charge for pz_print.
    base_n_plus_1 = _kcp_base_inputs(
        structure,
        nspin=nspin,
        nelec=nelec + 1,
        nelup=(nelup + 1) if nelup is not None else None,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
    )
    base_n = _kcp_base_inputs(
        structure,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
    )

    dummy_parameters = _build_dft_n_plus_1_dummy_parameters(
        base_n_plus_1, fixed_band=fixed_band, index_empty_to_save=index_empty_to_save
    )
    if dummy_overrides:
        dummy_parameters = recursive_merge(dummy_parameters, dummy_overrides)
    dummy_inputs = _build_kcp_inputs(
        code,
        structure,
        dummy_parameters,
        pseudos,
        options=options,
        name="dft_n_plus_1_dummy",
    )
    dummy_outputs = KcpBaseTask(**dummy_inputs)

    pz_parameters = _build_pz_print_parameters(
        base_n, nbnd=nbnd, fixed_band=fixed_band, index_empty_to_save=index_empty_to_save
    )
    if pz_overrides:
        pz_parameters = recursive_merge(pz_parameters, pz_overrides)
    # ``pz_print`` reads orbitals from the trial KI (it operates at the
    # original electron count) and writes ``evcfixed_empty.dat`` for
    # the next step. ``pz_alphas`` is the uniform-``alpha_guess`` payload
    # already built by ``generate_alphas`` upstream.
    pz_inputs = _build_kcp_inputs(
        code,
        structure,
        pz_parameters,
        pseudos,
        options=options,
        alphas=pz_alphas,
        parent_folder=trial_remote,
        name="pz_print",
    )
    pz_outputs = KcpBaseTask(**pz_inputs)

    n_plus_1_parameters = _build_dft_n_plus_1_parameters(
        base_n_plus_1, fixed_band=fixed_band, index_empty_to_save=index_empty_to_save
    )
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
        name="dft_n_plus_1",
    )
    n_plus_1_outputs = KcpBaseTask(**n_plus_1_inputs)

    result = compute_alpha_from_dscf(
        trial_output_parameters=trial_output_parameters,
        perturbed_output_parameters=n_plus_1_outputs["output_parameters"],
        trial_lambdas=trial_lambdas,
        trial_bare_lambdas=trial_bare_lambdas,
        spin_channel=spin_channel,
        band_index=band_index,
        alpha_guess=alpha_guess,
        filled=False,
    )

    return OrbitalDeltaSCFOutputs(
        alpha=result["alpha"],
        error=result["error"],
    )


# ----------------------------------------------------------------------
# One DSCF iteration body: trial KI → per-orbital DSCF → assemble alphas.
# Extracted so the multi-iteration loop in ``KIDscfRefinementTask`` can
# call it once per pass (a ``While`` zone gates re-entry in the procedural
# refinement builder).
# ----------------------------------------------------------------------


@task.graph
def OneDSCFIteration(
    *,
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudos: Annotated[dict, dynamic(UpfData)],
    base: KcpBaseInputs,
    nbnd: int,
    functional: str,
    spin_polarized: bool,
    current_alphas: AlphaScreening,
    parent_folder: orm.RemoteData,
    variational_orbital_overlays: dict | None = None,
    ki_overrides: KcpNamelistOverrides | None = None,
    filled_overrides: KcpNamelistOverrides | None = None,
    empty_overrides_dict: dict[str, KcpNamelistOverrides | None] | None = None,
    options: dict[str, Any] | None = None,
) -> OneDSCFIterationOutputs:
    """One iteration of the alpha-refinement loop.

    Runs a trial KI starting from ``current_alphas`` + ``parent_folder``,
    then a per-orbital Delta-SCF scatter (Map zones over filled / empty
    branches), then ``assemble_alpha_screening`` to pack the gathered
    per-orbital alphas back into an :class:`AlphaScreening`. Reports
    ``max_error`` so the outer ``While`` loop can stop on convergence.

    The trial KI is named ``ki_trial`` (call_link_label) — the ``While``
    loop appends an iteration suffix at the workgraph layer via the task
    builder's own auto-disambiguation. ``variational_orbital_overlays``
    is supplied on the first iteration only (the KS-as-variational
    overlay); subsequent iterations inherit the converged ``evc0N.dat``
    from the previous iteration's trial save via the primary parent walk.

    ``base`` is a frozen ``KcpBaseInputs`` dataclass and crosses this
    ``@task.graph`` boundary intact (aiida-workgraph routes
    dataclass-typed sockets through ``dataclasses.asdict`` + ``cls(**)``
    reconstruction, preserving ``None``-valued fields — unlike plain
    ``dict`` sockets, which silently strip ``None`` entries in transit).
    """
    ki_parameters = _build_ki_parameters(base, nbnd=nbnd, functional=functional)
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
        name="ki_trial",
    )
    trial = KcpBaseTask(**trial_inputs)

    filled_source = build_filled_iter_source(
        nbnd=nbnd,
        nelec=base.nelec,
        nelup=base.nelup,
        neldw=base.neldw,
        filled_alphas=current_alphas["filled"],
        spin_polarized=spin_polarized,
    )
    with Map(filled_source) as filled_zone:
        filled_item = filled_zone.item.value
        filled_out = OrbitalDeltaSCFFilledTask(
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
            fixed_band=_get_value(data=filled_item, key="fixed_band").result,
            spin_channel=_get_value(data=filled_item, key="spin_channel").result,
            band_index=_get_value(data=filled_item, key="band_index").result,
            alpha_guess=_get_value(data=filled_item, key="alpha_guess").result,
            trial_remote=trial["remote_folder"],
            trial_output_parameters=trial["output_parameters"],
            trial_lambdas=trial["output_lambdas"],
            trial_bare_lambdas=trial["output_bare_lambdas"],
            overrides=filled_overrides,
            options=options,
        )
        filled_zone.gather({"alpha": filled_out["alpha"], "error": filled_out["error"]})

    empty_source = build_empty_iter_source(
        nbnd=nbnd,
        nelec=base.nelec,
        nelup=base.nelup,
        neldw=base.neldw,
        empty_alphas=current_alphas["empty"],
        spin_polarized=spin_polarized,
    )
    with Map(empty_source) as empty_zone:
        empty_item = empty_zone.item.value
        empty_out = OrbitalDeltaSCFEmptyTask(
            code=code,
            structure=structure,
            pseudos=pseudos,
            ecutwfc=base.ecutwfc,
            ecutrho=base.ecutrho,
            nbnd=nbnd,
            nspin=base.nspin,
            nelec=base.nelec,
            nelup=base.nelup,
            neldw=base.neldw,
            tot_magnetization=base.tot_magnetization,
            fixed_band=_get_value(data=empty_item, key="fixed_band").result,
            spin_channel=_get_value(data=empty_item, key="spin_channel").result,
            band_index=_get_value(data=empty_item, key="band_index").result,
            index_empty_to_save=_get_value(data=empty_item, key="index_empty_to_save").result,
            alpha_guess=_get_value(data=empty_item, key="alpha_guess").result,
            pz_alphas=current_alphas,
            trial_remote=trial["remote_folder"],
            trial_output_parameters=trial["output_parameters"],
            trial_lambdas=trial["output_lambdas"],
            trial_bare_lambdas=trial["output_bare_lambdas"],
            overrides=empty_overrides_dict,
            options=options,
        )
        empty_zone.gather({"alpha": empty_out["alpha"], "error": empty_out["error"]})

    gathered = assemble_alpha_screening(
        filled_alphas=filled_zone.outputs.alpha,
        filled_errors=filled_zone.outputs.error,
        empty_alphas=empty_zone.outputs.alpha,
        empty_errors=empty_zone.outputs.error,
    )

    max_err = max_alpha_error(
        filled_errors=gathered["errors"]["filled"],
        empty_errors=gathered["errors"]["empty"],
    )

    return {
        "alphas": gathered["alphas"],
        "errors": gathered["errors"],
        "trial_remote": trial["remote_folder"],
        "max_error": max_err.result,
    }


# ----------------------------------------------------------------------
# Single-iteration alpha-refinement: trial KI → per-orbital DSCF → final KI.
# ----------------------------------------------------------------------


@task.graph
def KIDscfRefinementTask(
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
    functional: str,
    init_orbitals: str,
    dft_remote: orm.RemoteData,
    nelup: int | None = None,
    neldw: int | None = None,
    tot_magnetization: int | None = None,
    spin_polarized: bool = False,
    overrides: KoopmansDSCFOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> KoopmansDSCFOutputs:
    """One iteration of alpha refinement: trial KI → per-orbital DSCF → final KI.

    The trial KI restarts from ``dft_remote`` (the DFT init's
    ``remote_folder``) with a uniform ``initial_alpha`` guess and converges
    the variational orbital basis. The final KI then restarts from the
    *trial KI*'s ``remote_folder`` (legacy ``_koopmans_dscf.py:276+333``)
    so it inherits those converged ``evc0N.dat`` variational orbitals; only
    the alphas differ between the two passes. The per-orbital DSCF sub-runs
    are independently parented on the trial KI (they read its lambdas to
    compute the alpha update).

    All per-orbital fan-out happens through ``Map`` zones over a
    runtime-generated source dict (see ``build_filled_iter_source`` /
    ``build_empty_iter_source``); no socket arithmetic is performed
    inside this body.
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

    # ------------------------------------------------------------------
    # Filled / empty counts as sockets — never derived by socket arithmetic.
    # ------------------------------------------------------------------
    counts = filled_and_empty_counts_task(
        nspin=nspin, nbnd=nbnd, nelec=nelec, nelup=nelup, neldw=neldw
    )
    n_filled = counts["n_filled"]
    n_empty = counts["n_empty"]

    # Uniform-``initial_alpha`` payload feeds the first iteration's trial
    # KI (and its empty-orbital ``pz_print``). Subsequent iterations
    # consume the previous iteration's gathered alphas; that wiring lives
    # in the ``While`` loop that B.3 will add.
    initial_alphas = generate_alphas(
        alpha_guess=initial_alpha,
        n_filled=n_filled,
        n_empty=n_empty,
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
    # the DFT init, not the trial KI's inner-loop minimum). Once B.3
    # wraps the loop, only the *first* iteration receives this overlay;
    # all subsequent iterations parent on the previous trial KI and
    # inherit its converged ``evc0N.dat`` via the primary parent walk.
    ks_overlay: dict[str, str] | None = None
    if init_orbitals == "kohn-sham":
        nspin_overlay_iter = (1, 2) if nspin == 2 else (1,)
        # Stems only — the CalcJob appends ``.dat`` at submission time
        # (AiiDA's attribute store rejects Dict keys containing ``.``).
        ks_overlay = {
            **{f"evc{i}": f"evc0{i}" for i in nspin_overlay_iter},
            **{f"evc_empty{i}": f"evc0_empty{i}" for i in nspin_overlay_iter},
        }

    # ------------------------------------------------------------------
    # Single iteration. B.3 will wrap this call (or rather the underlying
    # ``OneDSCFIteration`` graph task) inside a ``While`` zone whose
    # condition reads ``iteration["max_error"] < 1e-3``, with a cap at
    # ``alpha_numsteps``.
    # ------------------------------------------------------------------
    iteration = OneDSCFIteration(
        code=code,
        structure=structure,
        pseudos=pseudos,
        base=base,
        nbnd=nbnd,
        functional=functional,
        spin_polarized=spin_polarized,
        current_alphas=initial_alphas,
        parent_folder=dft_remote,
        variational_orbital_overlays=ks_overlay,
        ki_overrides=ki_overrides,
        filled_overrides=filled_overrides,
        empty_overrides_dict=empty_overrides_dict,
        options=options,
    )

    # ------------------------------------------------------------------
    # Final KI: refined alphas, restart from the *trial KI* save. The
    # trial pass already converged the variational orbital basis (its
    # ``evc0N.dat`` files are the inner-loop minimum), and the final KI
    # picks up from there — legacy ``_koopmans_dscf.py:276+333``
    # parents the final KI on the trial KI's ``n_electron_restart_dir``
    # (its ``_60.save``), not on the bare DFT save. Restarting from DFT
    # would discard the trial's variational rotations and silently
    # rotate the canonical KI eigenvalues / lambda matrices.
    # ------------------------------------------------------------------
    ki_parameters = _build_ki_parameters(base, nbnd=nbnd, functional=functional)
    if ki_overrides:
        ki_parameters = recursive_merge(ki_parameters, ki_overrides)
    final_inputs = _build_kcp_inputs(
        code,
        structure,
        ki_parameters,
        pseudos,
        options=options,
        alphas=iteration["alphas"],
        parent_folder=iteration["trial_remote"],
        name="ki_final",
    )
    final = KcpBaseTask(**final_inputs)

    return KoopmansDSCFOutputs(
        parameters=final["output_parameters"],
        eigenvalues=final["output_eigenvalues"],
        lambdas=final["output_lambdas"],
        bare_lambdas=final["output_bare_lambdas"],
        remote_folder=final["remote_folder"],
    )


# ----------------------------------------------------------------------
# MVP scope enforcement
# ----------------------------------------------------------------------


def _validate_scope(
    *,
    functional: str,
    init_orbitals: str,
    alpha_numsteps: int,
    fix_spin_contamination: bool,
    structure: orm.StructureData,
) -> None:
    """Fail fast on inputs the MVP workflow cannot honour yet."""
    if functional != "ki":
        raise NotImplementedError(
            f"functional={functional!r} not yet ported. Only 'ki' is implemented. "
            "KIPZ / pKIPZ need the full DSCF alpha refinement loop and the "
            "additional trial-pass logic from the legacy KoopmansDSCFWorkflow."
        )
    if init_orbitals != "kohn-sham":
        raise NotImplementedError(
            f"init_orbitals={init_orbitals!r} not yet ported. Only 'kohn-sham' is "
            "implemented. MLWF / projected-WF initialisation requires a separate "
            "wannierize + fold-to-supercell pipeline."
        )
    if alpha_numsteps != 1:
        raise NotImplementedError(
            f"alpha_numsteps={alpha_numsteps} not yet supported. "
            "Multi-iteration alpha refinement is Phase B; only alpha_numsteps=1 supported."
        )
    if fix_spin_contamination:
        raise NotImplementedError(
            "fix_spin_contamination=True is not yet ported. The legacy workflow "
            "runs a 7-call spin-symmetrisation pre-pass; its AiiDA equivalent "
            "is a separate SpinSymmetrizeTask that hasn't been written yet."
        )
    if any(structure.pbc):
        raise NotImplementedError(
            "Periodic systems are not yet supported. The MVP targets the "
            "molecular (non-periodic) case used in tutorial_1. Periodic "
            "workflows require supercell folding and Wannier orbitals."
        )


# ----------------------------------------------------------------------
# Parameter builders
# ----------------------------------------------------------------------


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
            ``do_outerloop_empty=False`` and drop ``empty_states_maxstep``.
            Matches the legacy ``nspin1_dummy_calculator`` /
            ``nspin2_dummy_calculator`` settings used by
            ``restart_with_higher_precision``.
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
    # outer loop runs; legacy dummy calculators strip it when the loop is
    # disabled (see ``nspin1_dummy_calculator`` in the legacy code).
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


def _build_ki_parameters(
    base: KcpBaseInputs,
    *,
    nbnd: int,
    functional: str,
) -> dict[str, Any]:
    """Parameter dict for the KI correction step. Restarts from the DFT save file."""
    params = _build_dft_parameters(base, nbnd=nbnd)
    # ``restart_mode`` is the only ``&CONTROL`` key the KI builder owns; ndr/ndw
    # are forced by the CalcJob (see ``_build_dft_parameters`` for context).
    params["CONTROL"]["restart_mode"] = "restart"

    # Orbital-dependent screening.
    params["SYSTEM"]["do_orbdep"] = True

    # The orbital-dependent SCF runs no outer loop and no inner loop except
    # for PZ — see the legacy decision tree in
    # ``koopmans/src/koopmans/workflows/_koopmans_dscf.py:1129-1138``.
    params["ELECTRONS"]["do_outerloop"] = False
    params["ELECTRONS"]["do_outerloop_empty"] = False
    # ``empty_states_maxstep`` is only meaningful when ``do_outerloop_empty``
    # is true; legacy strips it when the empty-manifold loop is disabled.
    params["ELECTRONS"].pop("empty_states_maxstep", None)

    params["NKSIC"] = {
        "which_orbdep": "nki",
        "odd_nkscalfact": True,
        "odd_nkscalfact_empty": True,
        "nkscalfact": 1.0,
        "do_innerloop": functional == "pz",
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
# Delta -SCF alpha-refinement sub-step builders
# ----------------------------------------------------------------------
#
# These render the kcp.x inputs for the per-orbital sub-runs that compute
# alpha screening parameters via Delta SCF. Step list (legacy ``tutorial_1`` /
# ``02-calculate-screening-via-dscf/01-iteration-1/``):
#
#   filled orbital → ``dft_n-1``
#   empty  orbital → ``dft_n+1_dummy`` (iter 1 only) → ``pz_print`` → ``dft_n+1``
#
# Common deltas vs ``_build_dft_parameters`` (legacy
# ``_koopmans_dscf.py:1087-1126``):
# - ``nbnd`` removed.
# - ``conv_thr`` and ``esic_conv_thr`` 100x looser.
# - ``empty_states_maxstep`` / ``do_outerloop_empty`` removed (no empty
#   manifold treatment in these single-orbital runs).
# - ``&NKSIC`` always present (even when ``do_orbdep=False``) carrying
#   the shared inner-loop convergence knobs.
#
# Phase A scope: KI only (functional='ki'), single iteration, non-spin-
# polarised, no orbital grouping, no Makov-Payne correction, no early
# exit. See the deferred-items block at the end of this file.

_LOOSE_CONV_FACTOR = 100.0  # legacy: ``conv_thr *= 100`` for alpha-loop sub-runs


def _alpha_step_lite_nksic(
    *, conv_thr: float, index_empty_to_save: int | None = None
) -> dict[str, Any]:
    """Minimal ``&NKSIC`` block emitted on every alpha-loop step.

    All keys here appear in every legacy alpha-loop ``.cpi`` regardless of
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


def _build_dft_n_minus_1_parameters(
    base: KcpBaseInputs,
    *,
    fixed_band: int,
) -> dict[str, Any]:
    """``dft_n-1`` step: DFT with one electron removed from ``fixed_band``.

    Run once per *filled* orbital being screened. Restarts from the
    trial-KI save (provided by the caller via ``parent_folder``).
    """
    params = _alpha_step_dft_base(base)
    params["CONTROL"]["restart_mode"] = "restart"
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["SYSTEM"]["f_cutoff"] = 1.0e-5
    params["SYSTEM"]["fixed_state"] = True
    # ``do_outerloop`` already True from the DFT base.
    params["NKSIC"] = _alpha_step_lite_nksic(conv_thr=params["ELECTRONS"]["conv_thr"])
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
    state (legacy convention: spin-up gets the extra electron) — i.e.
    ``nelec += 1`` and ``nelup += 1`` relative to the trial-KI base.
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


def _build_dft_n_plus_1_parameters(
    base: KcpBaseInputs,
    *,
    fixed_band: int,
    index_empty_to_save: int = 1,
) -> dict[str, Any]:
    """``dft_n+1`` step: SCF DFT with one electron in ``fixed_band``.

    Restarts from ``dft_n+1_dummy`` plus ``pz_print``'s
    ``evcfixed_empty.dat`` (``restart_from_wannier_pwscf=True``). The
    caller is responsible for staging both files into the working dir.
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
    return params


def _build_pz_print_parameters(
    base: KcpBaseInputs,
    *,
    nbnd: int,
    fixed_band: int,
    index_empty_to_save: int = 1,
) -> dict[str, Any]:
    """``pz_print`` step: PZ run on the fixed empty orbital, prints anion wfc.

    Sandwiched between ``dft_n+1_dummy`` and ``dft_n+1`` for empty
    orbitals. Writes ``evcfixed_empty.dat`` (via
    ``print_wfc_anion=True``) so ``dft_n+1`` can use it as a starting
    wavefunction.

    Runs at the *original* electron count (not N+1) — same nelec /
    nelup / neldw as trial KI; only ``fixed_band`` differs.
    """
    params = _build_ki_parameters(base, nbnd=nbnd, functional="pz")
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["NKSIC"]["which_orbdep"] = "pz"
    params["NKSIC"]["print_wfc_anion"] = True
    params["NKSIC"]["index_empty_to_save"] = index_empty_to_save
    # The pz_print step's only job is to write ``evcfixed_empty{ispin}.dat``;
    # it operates on already-converged orbitals from the trial-KI save and
    # must NOT run the inner-loop SCF. ``_build_ki_parameters(functional="pz")``
    # turns it on by default — override here. Legacy reference:
    # tutorial_1's pz_print.cpi has ``do_innerloop=.false.``. Without this,
    # kcp.x runs a full PZ inner-CG cycle and pz_print balloons from
    # ~1 second to ~20 minutes.
    params["NKSIC"]["do_innerloop"] = False
    return params


# ----------------------------------------------------------------------
# Phase A scope notes — what's intentionally deferred:
# ----------------------------------------------------------------------
#
# This first slice of the alpha-refinement loop targets the simplest valid
# input: ``functional='ki'``, single iteration, non-spin-polarised,
# kohn-sham init orbitals, no orbital grouping. Specifically deferred:
#
# 1. **Multi-iteration with early exit.** Legacy runs up to
#    ``alpha_numsteps`` iterations and exits early when every band's
#    ``|Delta E - λ|`` falls below ``alpha_conv_thr``. Phase A runs exactly
#    one iteration. Phase B will wrap the iteration body in an
#    aiida-workgraph iteration primitive with a convergence predicate.
#
# 2. **Orbital grouping.** Legacy auto-groups orbitals by self-Hartree
#    and spread tolerances and only refines one representative per
#    group (``self.bands.assign_groups`` in ``_koopmans_dscf.py``).
#    Phase A refines every band individually — correct but wasteful
#    for systems with degeneracies (e.g. p-orbitals on cubic
#    substrates).
#
# 3. **Spin-polarised systems** (``spin_polarized=True``). Legacy
#    treats every (spin, index) pair as a unique group. Phase A
#    assumes ``nspin=2`` closed-shell — both channels share one set
#    of alpha values. Adding the spin-polarised branch needs per-spin
#    iteration and per-(spin, band) ``fixed_band`` indexing.
#
# 4. **KIPZ / pKIPZ.** The PZ-style sub-prefixes (``kipz_n-1``,
#    ``kipz_print``, ``kipz_n+1``) replace the DFT sub-prefixes for
#    the KIPZ functional. Mostly mechanical — the existing scope
#    guard in ``_validate_scope`` still rejects anything other than
#    KI.
#
# 5. **Makov-Payne correction** to Delta E (``mp_correction``,
#    ``eps_inf``). Legacy applies a per-orbital correction term when
#    the system is charged-periodic. Phase A omits it — the
#    structure scope guard already rejects periodic systems.
#
# 6. **Mixing across iterations** (``alpha_mixing``). Without a
#    loop there's nothing to mix; relevant only once Phase B lands.
#
# 7. **alpha-independent calc reuse** across iterations. Legacy caches
#    the ``dft_n-1`` results for filled orbitals because they don't
#    depend on alpha (``_koopmans_dscf.py:806-815``). One-iteration
#    Phase A doesn't need this.
#
# 8. **ML predict shortcut** (``self.ml.predict``). Legacy can
#    short-circuit the loop using a pre-trained ML model. Out of
#    scope.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Shared CalcJob-input assembly
# ----------------------------------------------------------------------


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
    name: str | None = None,
) -> dict[str, Any]:
    """Assemble a kwargs dict for ``KcpBaseTask(**inputs)``.

    Plain Python data (the ``parameters`` dict, the ``alphas``
    TypedDict) is handed straight through; aiida-workgraph's
    serialization adapter wraps each value into the matching AiiDA
    Node when the underlying CalcJob socket is set.

    ``name`` becomes ``metadata.call_link_label`` on the resulting CalcJob —
    that's what shows up in ``verdi process list`` and the koopmans progress
    table (e.g. ``kcp-dft_init`` instead of ``kcp-KcpCalculation``).

    Inside the per-orbital Map zones, ``name`` is set statically (e.g.
    ``"dft_n-1"``, ``"pz_print"``, ``"dft_n+1_dummy"``, ``"dft_n+1"``) — the
    band/spin identity is not interpolated at build time because
    ``fixed_band`` etc. arrive as sockets. ``aiida-workgraph`` reattaches it
    at Map-expansion time: ``aiida_workgraph.engine.task_manager.copy_task``
    prefixes every cloned descendant with the source-dict key, so the
    runtime label lands as e.g. ``kcp-up_band_2_dft_n-1`` (spin/band from the
    Map source dict + the static sub-step name).

    ``parent_folder_evcfixed`` is the ``RemoteData`` of a ``pz_print``
    run; only the ``dft_n+1`` step of the empty-orbital Delta-SCF branch
    needs this. The CalcJob symlinks the file
    ``out/<prefix>_<NDW>.save/K00001/evcfixed_empty.dat`` from that
    folder onto its read save (see
    ``KcpCalculation._build_remote_symlink_list``).
    """
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
    metadata: dict[str, Any] = {}
    if options:
        metadata["options"] = options
    if name:
        metadata["call_link_label"] = name
    if metadata:
        inputs["metadata"] = metadata
    return inputs
