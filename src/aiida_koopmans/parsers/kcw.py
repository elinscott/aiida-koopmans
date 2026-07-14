"""Parsers for the kcw.x CalcJobs (wann2kcw / screen / ham modes).

All three modes share the stdout header/footer conventions (``JOB DONE``,
``KCW          :  ...s CPU  ...s WALL``); the mode-specific content is the
``relaxed ... alpha ...`` screening table for ``screen`` and the interpolated
eigenvalue blocks for ``ham``.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
from aiida import orm
from qe_tools import CONSTANTS

from aiida_koopmans.parsers.base import KoopmansStdoutParser, time_string_to_seconds
from aiida_koopmans.parsers.kcp import safe_floats


class KcwScreenParameters(TypedDict, total=False):
    """``output_parameters`` payload of a ``screen``-mode kcw.x run.

    The ``job_done`` / ``walltime`` / ``walltime_units`` keys come from the
    shared base scalars; the rest are the per-orbital screening columns. Runtime
    value stays an ``orm.Dict``; this is annotation-level only.
    """

    job_done: bool
    walltime: float | None
    walltime_units: str
    relaxed: list[float]
    unrelaxed: list[float]
    self_hartree: list[float]
    self_hartree_units: str


class KcwHamParameters(TypedDict, total=False):
    """``output_parameters`` payload of a ``ham``-mode kcw.x run.

    The ``job_done`` / ``walltime`` / ``walltime_units`` keys come from the
    shared base scalars; the eigenvalue tables are per grid-point. Runtime
    value stays an ``orm.Dict``; this is annotation-level only.
    """

    job_done: bool
    walltime: float | None
    walltime_units: str
    ks_eigenvalues_on_grid: list[list[float]]
    ki_eigenvalues_on_grid: list[list[float]]
    eigenvalue_units: str
    ks_homo_energy: float
    ks_lumo_energy: float
    ki_homo_energy: float
    ki_lumo_energy: float


def _interpolated_block(lines: list[str], start: int) -> list[float]:
    """Collect one "KC interpolated eigenvalues" block (runs until a blank line)."""
    values: list[float] = []
    j = start
    while j < len(lines) and lines[j].strip():
        values += safe_floats(lines[j])
        j += 1
    return values


def _is_numeric_row(payload: str) -> bool:
    """Whether a KS/KI-prefixed line is an eigenvalue row (numbers only).

    Progress lines like "KS Hamiltonian calculation at k= ... DONE" share the
    prefix; anything containing letters is not grid data.
    """
    return not any(c.isalpha() for c in payload)


def _record_homo_lumo(stripped: str, results: dict[str, Any]) -> None:
    """Store a KS/KI homo-lumo summary line as scalar results."""
    homo, lumo = safe_floats(stripped)[-2:]
    prefix = "ks" if stripped.startswith("KS") else "ki"
    results[f"{prefix}_homo_energy"] = homo
    results[f"{prefix}_lumo_energy"] = lumo


class KcwBaseParser(KoopmansStdoutParser):
    """Shared stdout retrieval + common-scalar parsing for the kcw.x modes."""

    def parse(self, **kwargs: Any):
        """Read the stdout, delegate to ``_parse_mode``, and check completion."""
        stdout = self._read_stdout()
        if not isinstance(stdout, str):
            return stdout

        results = self._parse_common(stdout)
        mode_exit = self._parse_mode(stdout, results)

        self.out("output_parameters", orm.Dict(dict=results))

        if mode_exit is not None:
            return mode_exit
        if not results["job_done"]:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
        return None

    def _parse_mode(self, stdout: str, results: dict[str, Any]):
        """Mode-specific parsing hook; may mutate ``results`` and emit outputs.

        Returns an exit code to abort with, or ``None``.
        """
        return None

    def _parse_common(self, stdout: str) -> dict[str, Any]:
        """Extract ``job_done`` and the walltime from the kcw.x stdout."""
        results = self._base_scalars(stdout)
        for line in stdout.splitlines():
            if "KCW          :" in line:
                time_str = line.split("CPU")[-1].split("WALL")[0].strip()
                try:
                    results["walltime"] = time_string_to_seconds(time_str)
                except ValueError:
                    pass
        return results


class Wann2kcParser(KcwBaseParser):
    """Parse a ``wann2kcw``-mode run: only the common scalars."""


class KcwScreenParser(KcwBaseParser):
    """Parse a ``screen``-mode run: the per-orbital screening parameters.

    Each converged orbital prints a line of the form::

        iwann  =  1  relaxed = 0.0935  unrelaxed = 0.6516  alpha = 0.1436  self Hartree = 0.3554

    (a trailing ``*`` on ``iwann*`` marks orbitals whose alpha was copied
    from another member of the same spread group -- they are included). The
    ``alpha`` column becomes the ``alphas`` ``orm.List`` output; the other
    columns land in ``output_parameters``.
    """

    def _parse_mode(self, stdout: str, results: dict[str, Any]):
        alphas: list[float] = []
        relaxed: list[float] = []
        unrelaxed: list[float] = []
        self_hartrees: list[float] = []
        for line in stdout.splitlines():
            if "relaxed" not in line:
                continue
            tokens = line.split()
            try:
                # Fixed column positions from the end (the iwann token at the
                # start may or may not carry a ``*``): ... relaxed = <r>
                # unrelaxed = <u> alpha = <a> self Hartree = <sh>
                relaxed.append(float(tokens[-11]))
                unrelaxed.append(float(tokens[-8]))
                alphas.append(float(tokens[-5]))
                self_hartrees.append(float(tokens[-1]) * CONSTANTS.ry_to_ev)
            except (IndexError, ValueError):
                continue

        screen_fields: KcwScreenParameters = {
            "relaxed": relaxed,
            "unrelaxed": unrelaxed,
            "self_hartree": self_hartrees,
            "self_hartree_units": "eV",
        }
        results.update(screen_fields)

        if not alphas:
            return self.exit_codes.ERROR_OUTPUT_ALPHAS_MISSING
        self.out("alphas", orm.List(list=alphas))
        return None


class KcwHamParser(KcwBaseParser):
    """Parse a ``ham``-mode run: Koopmans eigenvalues on the grid and k-path.

    ``KC interpolated eigenvalues at k=`` blocks become the ``bands``
    ``BandsData`` output; the ``KS`` / ``KI`` per-grid-point eigenvalue
    tables land in ``output_parameters``. For a Gamma-only run kcw.x prints
    no interpolated blocks, so the KI grid eigenvalues stand in for them.
    """

    def _parse_mode(self, stdout: str, results: dict[str, Any]):
        lines = stdout.splitlines()
        kpts: list[list[float]] = []
        eigenvalues: list[list[float]] = []
        ks_on_grid: list[list[float]] = []
        ki_on_grid: list[list[float]] = []

        for i_line, line in enumerate(lines):
            if "KC interpolated eigenvalues at k=" in line:
                kpts.append([float(x) for x in line.split()[-3:]])
                eigenvalues.append(_interpolated_block(lines, i_line + 2))

            if "band energies (ev):" in line:
                ks_on_grid.append([])
                ki_on_grid.append([])

            stripped = line.strip()
            if "highest occupied, lowest unoccupied level" in stripped:
                # Summary line ("KS/KI highest occupied, lowest unoccupied
                # level (ev): X Y"), not a grid eigenvalue row.
                _record_homo_lumo(stripped, results)
            elif stripped.startswith("KS ") and ks_on_grid and _is_numeric_row(stripped[3:]):
                ks_on_grid[-1] += safe_floats(stripped[3:])
            elif stripped.startswith("KI ") and ki_on_grid and _is_numeric_row(stripped[3:]):
                ki_on_grid[-1] += safe_floats(stripped[3:])

        ham_fields: KcwHamParameters = {
            "ks_eigenvalues_on_grid": ks_on_grid,
            "ki_eigenvalues_on_grid": ki_on_grid,
            "eigenvalue_units": "eV",
        }
        results.update(ham_fields)

        do_bands = self._do_bands_requested()

        if not eigenvalues and len(ki_on_grid) == 1:
            # Gamma-only: the KI grid eigenvalues *are* the band energies.
            eigenvalues = ki_on_grid
            kpts = [[0.0, 0.0, 0.0]]

        if eigenvalues:
            bands = orm.BandsData()
            bands.set_kpoints(np.array(kpts))
            bands.set_bands(np.array(eigenvalues), units="eV")
            self.out("bands", bands)
        elif do_bands:
            return self.exit_codes.ERROR_OUTPUT_BANDS_MISSING
        return None

    def _do_bands_requested(self) -> bool:
        """Whether the input parameters asked for the band interpolation."""
        params = self.node.inputs.parameters.get_dict()
        ham = {k.upper(): v for k, v in params.items()}.get("HAM", {})
        return bool({k.lower(): v for k, v in ham.items()}.get("do_bands", False))
