"""Input assembly for the kcp.x CalcJob.

Builds the kwargs dict a ``task(KcpCalculation)`` step takes, including
the ``SYSTEM.nr{1,2,3}b`` box-grid autogeneration kcp.x needs when a
pseudopotential carries core corrections.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aiida import orm
from aiida_pseudo.data.pseudo.upf import UpfData

from aiida_koopmans.parallelization import (
    ParallelizationDict,
    merge_parallelization_into_inputs,
)
from aiida_koopmans.screening import AlphaScreening


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
    """Bump ``nr`` up to the next FFT-friendly dimension."""
    while not _fft_dimension_allowed(nr) and nr <= 2049:
        nr += 1
    return nr


def _autogenerate_nrb(
    structure: orm.StructureData,
    pseudos: dict[str, UpfData],
    parameters: dict[str, Any],
) -> None:
    """Fill ``SYSTEM.nr{1,2,3}b`` when any pseudo carries core corrections.

    kcp.x aborts with
    "nr1b, nr2b, nr3b must be given for ultrasoft and core corrected pp"
    when a pseudo has non-linear core corrections and the small-box grid is
    unset (bites e.g. PseudoDojo; SG15 has no NLCC). The conservative
    guess is the full density-grid dimensions scaled by
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
    for key, vec, nr_i in zip(("nr1b", "nr2b", "nr3b"), at, nr, strict=True):
        system[key] = _good_fft(int(nr_i * 2.0 * rc_safe / (np.linalg.norm(vec) * alat_bohr)))


def build_kcp_inputs(
    code: orm.AbstractCode,
    structure: orm.StructureData,
    parameters: dict[str, Any],
    pseudos: dict[str, UpfData],
    *,
    parallelization: ParallelizationDict | None = None,
    alphas: AlphaScreening | None = None,
    parent_folder: orm.RemoteData | None = None,
    parent_folder_evcfixed: orm.RemoteData | None = None,
    variational_orbital_overlays: dict[str, str] | None = None,
    read_wavefunctions: dict[str, Any] | None = None,
    name: str | None = None,
    display: str | None = None,
) -> dict[str, Any]:
    """Assemble a kwargs dict for ``KcpStep(**inputs)``.

    Plain Python data (the ``parameters`` dict, the ``alphas``
    TypedDict) is handed straight through; aiida-workgraph's
    serialization adapter wraps each value into the matching AiiDA
    Node when the underlying CalcJob socket is set.

    ``name`` becomes ``metadata.call_link_label`` on the resulting CalcJob,
    which is what provenance reads as (``kcp-dft_init`` instead of
    ``kcp-KcpCalculation``); ``display`` becomes its ``metadata.label``,
    which is how a reader is shown it (``DFT initialization``).

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
    if name:
        inputs["metadata"] = {"call_link_label": name}
    if display:
        inputs.setdefault("metadata", {})["label"] = display
    merge_parallelization_into_inputs(inputs, parallelization, "kcp")
    return inputs
