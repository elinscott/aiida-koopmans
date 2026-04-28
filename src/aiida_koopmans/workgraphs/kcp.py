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

from typing import Any, TypedDict

from aiida import orm
from aiida.plugins import DataFactory
from aiida_quantumespresso.workflows.protocols.utils import recursive_merge
from aiida_workgraph import task

from aiida_koopmans.calculations.kcp import KcpCalculation
from aiida_koopmans.utils import (
    count_electrons,
    filled_and_empty_counts,
    resolve_pseudo_family,
)

UpfData = DataFactory("pseudo.upf")


# ----------------------------------------------------------------------
# Output / override typing
# ----------------------------------------------------------------------


class DFTCPOutputs(TypedDict):
    """Outputs of a single kcp.x DFT-only run."""

    parameters: orm.Dict
    eigenvalues: orm.ArrayData
    remote_folder: orm.RemoteData


class KoopmansDSCFOutputs(TypedDict):
    """Outputs of the KI correction step (the final result of a KI-DSCF workflow)."""

    parameters: orm.Dict
    eigenvalues: orm.ArrayData
    lambdas: orm.ArrayData
    remote_folder: orm.RemoteData


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
    """Per-step overrides for ``KoopmansDSCFTask``."""

    dft: KcpNamelistOverrides
    ki: KcpNamelistOverrides


# ----------------------------------------------------------------------
# Raw CalcJob as a workgraph task
# ----------------------------------------------------------------------

KcpBaseTask = task(KcpCalculation)


# ----------------------------------------------------------------------
# Public graphs
# ----------------------------------------------------------------------


@task.graph
def DFTCPTask(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    pseudo_family: str,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int = 2,
    tot_magnetization: int | None = None,
    overrides: KcpNamelistOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> DFTCPOutputs:
    """Run a kcp.x DFT SCF (``do_orbdep=False``) from scratch.

    No spin symmetrization (``fix_spin_contamination=False``). If the caller
    needs spin symmetrization, they should wrap this task in a higher-level
    graph that runs the symmetrization process — that graph is not yet
    ported.
    """
    pseudos = resolve_pseudo_family(pseudo_family, structure)
    nelec, nelup, neldw = count_electrons(
        structure, pseudos, nspin=nspin, tot_magnetization=tot_magnetization
    )

    parameters = _build_dft_parameters(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=not any(structure.pbc),
    )
    if overrides:
        parameters = recursive_merge(parameters, dict(overrides))

    inputs = _build_kcp_inputs(
        code, structure, parameters, pseudos, options=options, name="dft_init"
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
    overrides: KoopmansDSCFOverrides | None = None,
    options: dict[str, Any] | None = None,
) -> KoopmansDSCFOutputs:
    """Koopmans DSCF workflow — DFT init followed by a KI correction.

    **MVP scope.** Only the two-step DFT → KI pipeline is run. The Delta SCF
    alpha-refinement loop is not executed yet; every orbital is assigned
    ``initial_alpha`` (legacy default 0.6). When the loop is added, this
    task's signature should not change — only its body.
    """
    _validate_scope(
        functional=functional,
        init_orbitals=init_orbitals,
        alpha_numsteps=alpha_numsteps,
        fix_spin_contamination=fix_spin_contamination,
        structure=structure,
    )

    mt_correction = not any(structure.pbc)

    dft_overrides = overrides.get("dft") if overrides else None
    ki_overrides = overrides.get("ki") if overrides else None

    dft = DFTCPTask(
        code=code,
        structure=structure,
        pseudo_family=pseudo_family,
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        tot_magnetization=tot_magnetization,
        overrides=dft_overrides,
        options=options,
        metadata={"call_link_label": "dft_init"},
    )

    pseudos = resolve_pseudo_family(pseudo_family, structure)
    nelec, nelup, neldw = count_electrons(
        structure, pseudos, nspin=nspin, tot_magnetization=tot_magnetization
    )

    ki_parameters = _build_ki_parameters(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=mt_correction,
        functional=functional,
    )
    if ki_overrides:
        ki_parameters = recursive_merge(ki_parameters, dict(ki_overrides))

    n_filled, n_empty = filled_and_empty_counts(
        nspin=nspin, nbnd=nbnd, nelec=nelec, nelup=nelup, neldw=neldw
    )
    alphas = orm.Dict(
        dict={
            "filled": [initial_alpha] * n_filled,
            "empty": [initial_alpha] * n_empty,
        }
    )

    ki_inputs = _build_kcp_inputs(
        code,
        structure,
        ki_parameters,
        pseudos,
        options=options,
        alphas=alphas,
        parent_folder=dft["remote_folder"],
        name="ki_final",
    )
    ki_outputs = KcpBaseTask(**ki_inputs)

    return KoopmansDSCFOutputs(
        parameters=ki_outputs["output_parameters"],
        eigenvalues=ki_outputs["output_eigenvalues"],
        lambdas=ki_outputs["output_lambdas"],
        remote_folder=ki_outputs["remote_folder"],
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
            f"alpha_numsteps={alpha_numsteps} not yet supported. The MVP uses "
            "the per-orbital initial_alpha and does not refine it via the Delta SCF "
            "loop. Pass alpha_numsteps=1 for now."
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


def _build_dft_parameters(
    *,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    tot_magnetization: int | None,
    mt_correction: bool,
) -> dict[str, Any]:
    """Parameter dict for the DFT initialization step."""
    conv_thr = 1.0e-9 * nelec
    system: dict[str, Any] = {
        "ecutwfc": ecutwfc,
        "ecutrho": ecutrho,
        "nbnd": nbnd,
        "nspin": nspin,
        "do_ee": mt_correction,
        "do_orbdep": False,
        "fixed_state": False,
        "do_wf_cmplx": True,
        "nelec": nelec,
    }
    if nspin == 2:
        if nelup is not None:
            system["nelup"] = nelup
        if neldw is not None:
            system["neldw"] = neldw
        if tot_magnetization is not None:
            system["tot_magnetization"] = tot_magnetization
    # ``ndr`` and ``ndw`` are owned by the CalcJob (see
    # ``KcpCalculation._inject_owned_keys``) — the builders deliberately
    # leave them unset so there's only one source of truth.
    params: dict[str, Any] = {
        "CONTROL": {
            "calculation": "cp",
            "verbosity": "low",
            "iprint": 1,
            "disk_io": "high",
            "write_hr": False,
            "restart_mode": "from_scratch",
        },
        "SYSTEM": system,
        "ELECTRONS": {
            "electron_dynamics": "cg",
            "passop": 2.0,
            "ortho_para": 1,
            "maxiter": 300,
            "empty_states_maxstep": 300,
            "do_outerloop": True,
            "do_outerloop_empty": True,
            "conv_thr": conv_thr,
        },
        "IONS": {
            "ion_dynamics": "none",
            "ion_nstepe": 5,
        },
    }
    # kcp.x reads ``&EE`` iff ``do_ee=.true.`` — keep the two consistent.
    if mt_correction:
        params["EE"] = {"which_compensation": "tcc"}
    return params


def _build_ki_parameters(
    *,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    tot_magnetization: int | None,
    mt_correction: bool,
    functional: str,
) -> dict[str, Any]:
    """Parameter dict for the KI correction step. Restarts from the DFT save file."""
    params = _build_dft_parameters(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=mt_correction,
    )
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
        "esic_conv_thr": 1.0e-9 * nelec,
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


def _alpha_step_dft_base(
    *,
    ecutwfc: float,
    ecutrho: float,
    nspin: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    tot_magnetization: int | None,
    mt_correction: bool,
) -> dict[str, Any]:
    """``&CONTROL/SYSTEM/ELECTRONS`` skeleton shared by every DFT-like alpha step.

    Built from ``_build_dft_parameters`` then trimmed: ``nbnd`` dropped,
    ``conv_thr`` loosened, empty-manifold knobs removed.
    """
    params = _build_dft_parameters(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=0,  # required by helper signature but stripped below.
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=mt_correction,
    )
    params["SYSTEM"].pop("nbnd", None)
    params["ELECTRONS"].pop("empty_states_maxstep", None)
    params["ELECTRONS"].pop("do_outerloop_empty", None)
    params["ELECTRONS"]["conv_thr"] *= _LOOSE_CONV_FACTOR
    return params


def _build_dft_n_minus_1_parameters(
    *,
    ecutwfc: float,
    ecutrho: float,
    nspin: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    tot_magnetization: int | None,
    mt_correction: bool,
    fixed_band: int,
) -> dict[str, Any]:
    """``dft_n-1`` step: DFT with one electron removed from ``fixed_band``.

    Run once per *filled* orbital being screened. Restarts from the
    trial-KI save (provided by the caller via ``parent_folder``).
    """
    params = _alpha_step_dft_base(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=mt_correction,
    )
    params["CONTROL"]["restart_mode"] = "restart"
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["SYSTEM"]["f_cutoff"] = 1.0e-5
    params["SYSTEM"]["fixed_state"] = True
    # ``do_outerloop`` already True from the DFT base.
    params["NKSIC"] = _alpha_step_lite_nksic(conv_thr=params["ELECTRONS"]["conv_thr"])
    return params


def _build_dft_n_plus_1_dummy_parameters(
    *,
    ecutwfc: float,
    ecutrho: float,
    nspin: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    tot_magnetization: int | None,
    mt_correction: bool,
    fixed_band: int,
    index_empty_to_save: int = 1,
) -> dict[str, Any]:
    """``dft_n+1_dummy`` step: scratch DFT with one electron *added*.

    Run only on the first iteration of the alpha loop, once per *empty*
    orbital. Sets up the save-directory layout that ``pz_print`` and
    ``dft_n+1`` consume on subsequent steps.

    Caller must pass ``nelec`` / ``nelup`` already incremented for the
    N+1 charge state (legacy convention: spin-up gets the extra
    electron).
    """
    params = _alpha_step_dft_base(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=mt_correction,
    )
    params["CONTROL"]["restart_mode"] = "from_scratch"
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["SYSTEM"]["fixed_state"] = False
    params["ELECTRONS"]["do_outerloop"] = False
    params["NKSIC"] = _alpha_step_lite_nksic(
        conv_thr=params["ELECTRONS"]["conv_thr"], index_empty_to_save=index_empty_to_save
    )
    return params


def _build_dft_n_plus_1_parameters(
    *,
    ecutwfc: float,
    ecutrho: float,
    nspin: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    tot_magnetization: int | None,
    mt_correction: bool,
    fixed_band: int,
    index_empty_to_save: int = 1,
) -> dict[str, Any]:
    """``dft_n+1`` step: SCF DFT with one electron in ``fixed_band``.

    Restarts from ``dft_n+1_dummy`` plus ``pz_print``'s
    ``evcfixed_empty.dat`` (``restart_from_wannier_pwscf=True``). The
    caller is responsible for staging both files into the working dir.
    """
    params = _alpha_step_dft_base(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=mt_correction,
    )
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
    *,
    ecutwfc: float,
    ecutrho: float,
    nbnd: int,
    nspin: int,
    nelec: int,
    nelup: int | None,
    neldw: int | None,
    tot_magnetization: int | None,
    mt_correction: bool,
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
    params = _build_ki_parameters(
        ecutwfc=ecutwfc,
        ecutrho=ecutrho,
        nbnd=nbnd,
        nspin=nspin,
        nelec=nelec,
        nelup=nelup,
        neldw=neldw,
        tot_magnetization=tot_magnetization,
        mt_correction=mt_correction,
        functional="pz",
    )
    params["SYSTEM"]["fixed_band"] = fixed_band
    params["NKSIC"]["which_orbdep"] = "pz"
    params["NKSIC"]["print_wfc_anion"] = True
    params["NKSIC"]["index_empty_to_save"] = index_empty_to_save
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
    alphas: orm.Dict | None = None,
    parent_folder: orm.RemoteData | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Assemble a kwargs dict for ``KcpBaseTask(**inputs)``.

    ``name`` becomes ``metadata.call_link_label`` on the resulting CalcJob —
    that's what shows up in ``verdi process list`` and the koopmans progress
    table (e.g. ``kcp-dft_init`` instead of ``kcp-KcpCalculation``).
    """
    inputs: dict[str, Any] = {
        "code": code,
        "structure": structure,
        "parameters": orm.Dict(dict=parameters),
        "pseudos": pseudos,
    }
    if alphas is not None:
        inputs["alphas"] = alphas
    if parent_folder is not None:
        inputs["parent_folder"] = parent_folder
    metadata: dict[str, Any] = {}
    if options:
        metadata["options"] = options
    if name:
        metadata["call_link_label"] = name
    if metadata:
        inputs["metadata"] = metadata
    return inputs
