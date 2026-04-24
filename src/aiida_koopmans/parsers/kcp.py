"""Parser for Quantum ESPRESSO ``kcp.x`` output files.

Parses the stdout ``.cpo`` text output and, when orbital-dependent screening is
requested, the Hamiltonian XML files under
``<outdir>/<prefix>_<ndw>.save/K00001/``.

Unit conversions use ``qe_tools.CONSTANTS`` (the AiiDA ecosystem's source of
Hartree, Bohr, etc.).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
from aiida import orm
from aiida.parsers import Parser
from qe_tools import CONSTANTS

_HARTREE_TO_EV = CONSTANTS.hartree_to_ev
_BOHR_TO_ANG = CONSTANTS.bohr_to_ang


class KcpParser(Parser):
    """Parse the stdout and Hamiltonian XML outputs of a ``KcpCalculation``."""

    def parse(self, **kwargs: Any):  # noqa: ARG002  (AiiDA passes retrieved_temporary_folder)
        """Entry point called by AiiDA after the CalcJob finishes."""
        try:
            retrieved = self.retrieved
        except Exception:  # noqa: BLE001
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        stdout_filename = self.node.base.attributes.get("output_filename")
        if stdout_filename not in retrieved.base.repository.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        try:
            stdout = retrieved.base.repository.get_object_content(stdout_filename)
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

        parsed, eigenvalues = self._parse_stdout(stdout)
        self.out("output_parameters", orm.Dict(dict=parsed))

        if eigenvalues:
            eig_array = orm.ArrayData()
            # Pad rows to equal length so we can stack; missing entries become NaN.
            max_len = max(len(row) for row in eigenvalues)
            padded = np.full((len(eigenvalues), max_len), np.nan)
            for i, row in enumerate(eigenvalues):
                padded[i, : len(row)] = row
            eig_array.set_array("eigenvalues", padded)
            self.out("output_eigenvalues", eig_array)

        # Hamiltonian XMLs (only when do_orbdep is requested).
        params = self.node.inputs.parameters.get_dict()
        system = {k.lower(): v for k, v in params.get("SYSTEM", {}).items()}
        nksic = {k.lower(): v for k, v in params.get("NKSIC", {}).items()}
        do_orbdep = bool(system.get("do_orbdep", False))
        do_bare_eigs = bool(nksic.get("do_bare_eigs", False))
        nspin = int(system.get("nspin", 1))

        if do_orbdep:
            lambdas_status = self._parse_lambdas(
                retrieved, nspin=nspin, bare=False
            )
            if lambdas_status is self.exit_codes.ERROR_OUTPUT_HAM_MISSING:
                return lambdas_status
            self.out("output_lambdas", lambdas_status)

            if do_bare_eigs:
                bare_status = self._parse_lambdas(retrieved, nspin=nspin, bare=True)
                if bare_status is self.exit_codes.ERROR_OUTPUT_HAM_MISSING:
                    return bare_status
                self.out("output_bare_lambdas", bare_status)

        if not parsed.get("job_done", False):
            return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE

        return None

    # ------------------------------------------------------------------
    # stdout parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_stdout(stdout: str) -> tuple[dict[str, Any], list[list[float]]]:
        """Extract the scalar results and per-spin eigenvalue lists from the .cpo text."""
        results: dict[str, Any] = {
            "energy": None,
            "energy_units": "eV",
            "odd_energy": None,
            "homo_energy": None,
            "lumo_energy": None,
            "mp1_energy": None,
            "mp2_energy": None,
            "lambda_ii": None,
            "walltime": None,
            "walltime_units": "s",
            "convergence": {"filled": [], "empty": []},
            "job_done": False,
            "orbital_data": {
                "charge": [],
                "centres": [],
                "spreads": [],
                "self-Hartree": [],
            },
        }
        eigenvalues: list[list[float]] = []

        lines = stdout.splitlines()
        convergence_key = "filled"
        i_spin_orbital = None

        for idx, line in enumerate(lines):
            if "                total energy" in line:
                # Legacy pattern: total energy line has the value 3 tokens from the end.
                tokens = line.split()
                try:
                    results["energy"] = _fortran_float(tokens[-3]) * _HARTREE_TO_EV
                except (IndexError, ValueError):
                    pass

            if "odd energy" in line:
                tokens = line.split()
                try:
                    results["odd_energy"] = _fortran_float(tokens[3]) * _HARTREE_TO_EV
                except (IndexError, ValueError):
                    pass

            if "fixed_lambda" in line and results["lambda_ii"] is None:
                tokens = line.split()
                try:
                    results["lambda_ii"] = _fortran_float(tokens[-1]) * _HARTREE_TO_EV
                except (IndexError, ValueError):
                    pass

            if (
                "HOMO Eigenvalue (eV)" in line
                and idx + 2 < len(lines)
                and "*" not in lines[idx + 2]
            ):
                try:
                    results["homo_energy"] = _fortran_float(lines[idx + 2].strip())
                except ValueError:
                    pass

            if (
                "LUMO Eigenvalue (eV)" in line
                and idx + 2 < len(lines)
                and "*" not in lines[idx + 2]
            ):
                try:
                    results["lumo_energy"] = _fortran_float(lines[idx + 2].strip())
                except ValueError:
                    pass

            if "Makov-Payne 1-order energy" in line:
                tokens = line.split()
                try:
                    results["mp1_energy"] = _fortran_float(tokens[4]) * _HARTREE_TO_EV
                except (IndexError, ValueError):
                    pass

            if "Makov-Payne 2-order energy" in line:
                tokens = line.split()
                try:
                    results["mp2_energy"] = _fortran_float(tokens[4]) * _HARTREE_TO_EV
                except (IndexError, ValueError):
                    pass

            if "Eigenvalues (eV), kp" in line:
                if "Empty" not in line:
                    eigenvalues.append([])
                j = idx + 2
                while j < len(lines) and lines[j].strip():
                    eigenvalues[-1].extend(_safe_floats(lines[j]))
                    j += 1

            if "Orb -- Charge  ---" in line or "Orb -- Empty Charge" in line:
                tokens = line.split()
                try:
                    i_spin_orbital = int(tokens[-1]) - 1
                except ValueError:
                    i_spin_orbital = 0
                for key in results["orbital_data"]:
                    while len(results["orbital_data"][key]) < i_spin_orbital + 1:
                        results["orbital_data"][key].append([])

            if line.startswith(("OCC", "EMP")) and "NaN" not in line:
                cleaned = line.replace("********", "   0.000")
                values = [
                    _fortran_float(cleaned[i - 4 : i + 4])
                    for i, c in enumerate(cleaned)
                    if c == "."
                ]
                if len(values) >= 6 and i_spin_orbital is not None:
                    results["orbital_data"]["charge"][i_spin_orbital].append(values[0])
                    results["orbital_data"]["centres"][i_spin_orbital].append(
                        [v * _BOHR_TO_ANG for v in values[1:4]]
                    )
                    results["orbital_data"]["spreads"][i_spin_orbital].append(
                        values[4] * _BOHR_TO_ANG * _BOHR_TO_ANG
                    )
                    results["orbital_data"]["self-Hartree"][i_spin_orbital].append(values[5])

            if "PERFORMING CONJUGATE GRADIENT MINIMIZATION OF EMPTY STATES" in line:
                convergence_key = "empty"

            if "iteration = " in line and "eff iteration = " in line:
                entry = _parse_convergence_line(line)
                if entry is not None:
                    results["convergence"][convergence_key].append(entry)

            if "JOB DONE" in line:
                results["job_done"] = True

            if "wall time" in line:
                # Format: "<anything>, <X>m<Y>s wall time"
                try:
                    time_part = line.split(",")[1].strip()
                    # Drop trailing 'wall time'
                    time_part = time_part.rstrip()
                    if time_part.endswith("wall time"):
                        time_part = time_part[: -len("wall time")].strip()
                    results["walltime"] = _time_string_to_seconds(time_part)
                except (IndexError, ValueError):
                    pass

        return results, eigenvalues

    # ------------------------------------------------------------------
    # Hamiltonian XML parsing
    # ------------------------------------------------------------------

    def _parse_lambdas(self, retrieved, nspin: int, bare: bool):
        """Return an ArrayData of lambda matrices or an exit-code sentinel on failure."""
        prefix = self.node.process_class._PREFIX  # noqa: SLF001
        ndw = int(
            {k.lower(): v for k, v in self.node.inputs.parameters.get_dict().get("CONTROL", {}).items()}.get(
                "ndw", 50
            )
        )
        out_subfolder = self.node.process_class._OUTPUT_SUBFOLDER  # noqa: SLF001
        ham_dir = f"{out_subfolder}/{prefix}_{ndw}.save/K00001"

        nspin_idx = list(range(1, nspin + 1))
        array = orm.ArrayData()
        for ispin in nspin_idx:
            tag = str(ispin) if nspin > 1 else ""
            prefix_token = "hamiltonian0" if bare else "hamiltonian"
            filename_filled = f"{ham_dir}/{prefix_token}{tag}.xml"
            filename_empty = f"{ham_dir}/{prefix_token}_emp{tag}.xml"

            try:
                filled_content = retrieved.base.repository.get_object_content(filename_filled)
            except (OSError, FileNotFoundError):
                return self.exit_codes.ERROR_OUTPUT_HAM_MISSING

            filled = _read_hamiltonian_xml(filled_content)

            try:
                empty_content = retrieved.base.repository.get_object_content(filename_empty)
            except (OSError, FileNotFoundError):
                empty = None
            else:
                empty = _read_hamiltonian_xml(empty_content)

            combined = filled if empty is None else _block_diag(filled, empty)
            array.set_array(f"spin_{ispin}", combined)

        return array


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _fortran_float(token: str) -> float:
    """Parse a float that may use Fortran's 'd' exponent ('1.23d-4' -> 1.23e-4)."""
    return float(token.replace("d", "e").replace("D", "e"))


def _safe_floats(string: str) -> list[float]:
    """Parse whitespace-separated floats tolerantly, mapping ``*****`` to NaN."""
    out: list[float] = []
    for word in string.split():
        if "*" in word:
            # Reduce runs of stars, then pad with spaces and re-split.
            while "**" in word:
                word = word.replace("**", "*")
            for tok in word.replace("*", " * ").split():
                out.append(math.nan if tok == "*" else _fortran_float(tok))
        elif word.count(".") > 1:
            out.extend([math.nan] * word.count("."))
        else:
            try:
                out.append(_fortran_float(word))
            except ValueError:
                out.append(math.nan)
    return out


def _parse_convergence_line(line: str) -> dict[str, Any] | None:
    """Parse the periodic ``iteration = N   eff iteration = M   Etot(Ha) = ... delta_E = ...`` lines."""
    try:
        parts = [segment.split()[0] for segment in line.split("=")[1:]]
    except IndexError:
        return None
    if len(parts) < 3:
        return None
    try:
        entry = {
            "iteration": int(parts[0]),
            "eff_iteration": int(parts[1]),
            "Etot": _fortran_float(parts[2]) * _HARTREE_TO_EV,
        }
    except ValueError:
        return None
    if len(parts) >= 4:
        try:
            entry["delta_E"] = _fortran_float(parts[3]) * _HARTREE_TO_EV
        except ValueError:
            pass
    return entry


def _time_string_to_seconds(time_str: str) -> float:
    """Convert strings like ``1d 2h 3m 4s`` / ``3m 4s`` / ``4s`` to seconds."""
    days, hours, minutes = 0.0, 0.0, 0.0
    rem = time_str
    if "d" in rem:
        d_part, rem = rem.split("d", 1)
        days = float(d_part)
    if "h" in rem:
        h_part, rem = rem.split("h", 1)
        hours = float(h_part)
    if "m" in rem:
        m_part, rem = rem.split("m", 1)
        minutes = float(m_part)
    seconds = float(rem.rstrip("s").strip() or 0.0)
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _read_hamiltonian_xml(content: str) -> np.ndarray:
    """Parse a kcp.x hamiltonian XML into a complex square matrix (in eV)."""
    root = ET.fromstring(content)
    size = int(root.attrib["size"])
    side = int(math.isqrt(size))
    if side * side != size:
        raise ValueError(f"Hamiltonian XML size {size} is not a perfect square.")
    if root.text is None:
        raise ValueError("Hamiltonian XML has no payload.")
    entries: list[complex] = []
    for row in root.text.strip().split("\n"):
        re_part, im_part = (_fortran_float(x) for x in row.split(","))
        entries.append(complex(re_part, im_part))
    matrix = np.array(entries, dtype=np.complex128) * _HARTREE_TO_EV
    return matrix.reshape((side, side))


def _block_diag(filled: np.ndarray, empty: np.ndarray) -> np.ndarray:
    """Block-diagonal stacking without pulling in scipy (two blocks only)."""
    n, m = filled.shape[0], empty.shape[0]
    out = np.zeros((n + m, n + m), dtype=filled.dtype)
    out[:n, :n] = filled
    out[n:, n:] = empty
    return out
