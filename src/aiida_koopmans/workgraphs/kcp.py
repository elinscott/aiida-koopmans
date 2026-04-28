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
including spin-symmetrization (7), a trial KI pass (1), a ΔSCF loop to
compute the alphas (12), and a final KI (1). Porting the ΔSCF alpha loop
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

    **MVP scope.** Only the two-step DFT → KI pipeline is run. The ΔSCF
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
            "the per-orbital initial_alpha and does not refine it via the ΔSCF "
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
    params: dict[str, Any] = {
        "CONTROL": {
            "calculation": "cp",
            "verbosity": "low",
            "iprint": 1,
            "disk_io": "high",
            "write_hr": False,
            "ndr": 50,
            "ndw": 50,
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
    # Restart from the DFT save (ndw=50) and write to a new one (ndw=60).
    params["CONTROL"]["restart_mode"] = "restart"
    params["CONTROL"]["ndr"] = 50
    params["CONTROL"]["ndw"] = 60

    # Orbital-dependent screening.
    params["SYSTEM"]["do_orbdep"] = True

    # The orbital-dependent SCF runs no outer loop and no inner loop except
    # for PZ — see the legacy decision tree in
    # ``koopmans/src/koopmans/workflows/_koopmans_dscf.py:1129-1138``.
    params["ELECTRONS"]["do_outerloop"] = False
    params["ELECTRONS"]["do_outerloop_empty"] = False

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
