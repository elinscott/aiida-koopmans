"""Parsers for the kcw.x CalcJobs (wann2kcw / screen / ham modes).

All three modes share the stdout header/footer conventions (``JOB DONE``,
``KCW          :  ...s CPU  ...s WALL``); the mode-specific content is the
``relaxed ... alpha ...`` screening table for ``screen`` and the interpolated
eigenvalue blocks for ``ham``. The line patterns mirror the legacy ASE
readers (``ase_koopmans.io.espresso._wann2kc`` / ``_koopmans_screen`` /
``_koopmans_ham``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from aiida import orm
from aiida.parsers import Parser
from qe_tools import CONSTANTS

from aiida_koopmans.parsers.kcp import _safe_floats, _time_string_to_seconds

_RY_TO_EV = CONSTANTS.ry_to_ev


class KcwBaseParser(Parser):
    """Shared stdout retrieval + common-scalar parsing for the kcw.x modes."""

    def parse(self, **kwargs: Any):
        """Read the stdout, delegate to ``_parse_mode``, and check completion."""
        try:
            retrieved = self.retrieved
        except Exception:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        stdout_filename = self.node.base.attributes.get("output_filename")
        if stdout_filename not in retrieved.base.repository.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        try:
            stdout = retrieved.base.repository.get_object_content(stdout_filename)
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

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

    @staticmethod
    def _parse_common(stdout: str) -> dict[str, Any]:
        """Extract ``job_done`` and the walltime from the kcw.x stdout."""
        results: dict[str, Any] = {
            "job_done": False,
            "walltime": None,
            "walltime_units": "s",
        }
        for line in stdout.splitlines():
            if "JOB DONE" in line:
                results["job_done"] = True
            if "KCW          :" in line:
                time_str = line.split("CPU")[-1].split("WALL")[0].strip()
                try:
                    results["walltime"] = _time_string_to_seconds(time_str)
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
    from another member of the same spread group -- they are included, same
    as the legacy reader). The ``alpha`` column becomes the ``alphas``
    ``orm.List`` output; the other columns land in ``output_parameters``.
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
                self_hartrees.append(float(tokens[-1]) * _RY_TO_EV)
            except (IndexError, ValueError):
                continue

        results["relaxed"] = relaxed
        results["unrelaxed"] = unrelaxed
        results["self_hartree"] = self_hartrees
        results["self_hartree_units"] = "eV"

        if not alphas:
            return self.exit_codes.ERROR_OUTPUT_ALPHAS_MISSING
        self.out("alphas", orm.List(list=alphas))
        return None


class KcwHamParser(KcwBaseParser):
    """Parse a ``ham``-mode run: Koopmans eigenvalues on the grid and k-path.

    ``KC interpolated eigenvalues at k=`` blocks become the ``bands``
    ``BandsData`` output; the ``KS`` / ``KI`` per-grid-point eigenvalue
    tables land in ``output_parameters``. For a Gamma-only run kcw.x prints
    no interpolated blocks, so the KI grid eigenvalues stand in for them
    (same fallback as the legacy ASE reader).
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
                eigenvalues.append([])
                j = i_line + 2
                while j < len(lines) and lines[j].strip():
                    eigenvalues[-1] += _safe_floats(lines[j])
                    j += 1

            if "band energies (ev):" in line:
                ks_on_grid.append([])
                ki_on_grid.append([])

            stripped = line.strip()
            if stripped.startswith("KS ") and ks_on_grid:
                ks_on_grid[-1] += _safe_floats(stripped[3:])
            if stripped.startswith("KI ") and ki_on_grid:
                ki_on_grid[-1] += _safe_floats(stripped[3:])

        results["ks_eigenvalues_on_grid"] = ks_on_grid
        results["ki_eigenvalues_on_grid"] = ki_on_grid
        results["eigenvalue_units"] = "eV"

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
