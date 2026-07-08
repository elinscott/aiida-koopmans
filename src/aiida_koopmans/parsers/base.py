"""Shared scaffolding for the Koopmans QE-fork stdout parsers.

The kcp.x / kcw.x / wann2kcp.x parsers all open their stdout the same way --
guard the retrieved folder, look up the ``output_filename`` attribute, check
it landed, and read it -- returning the matching exit code at each step. They
also all seed their scalar-results dict with the same
``job_done`` / ``walltime`` skeleton. That boilerplate lives here.

``merge_evc.x`` reads no stdout (it only checks that the merged file was
retrieved), so its parser stays a plain :class:`~aiida.parsers.Parser`.
"""

from __future__ import annotations

from typing import Any

from aiida.engine import ExitCode
from aiida.parsers import Parser


class KoopmansStdoutParser(Parser):
    """Base for the Koopmans QE-fork parsers that read a Fortran stdout file."""

    def _read_stdout(self) -> str | ExitCode:
        """Return the stdout text, or an exit code if it is missing / unreadable.

        Callers check ``isinstance(result, str)``: a string is the stdout
        content, anything else is an :class:`~aiida.engine.ExitCode` to return
        from ``parse``. Relies on the associated CalcJob declaring the
        301 / 302 / 303 exit codes (see
        :class:`~aiida_koopmans.calculations.base.KoopmansStdoutCalculation`).
        """
        try:
            retrieved = self.retrieved
        except Exception:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        stdout_filename = self.node.base.attributes.get("output_filename")
        if stdout_filename not in retrieved.base.repository.list_object_names():
            return self.exit_codes.ERROR_OUTPUT_STDOUT_MISSING

        try:
            return retrieved.base.repository.get_object_content(stdout_filename)
        except OSError:
            return self.exit_codes.ERROR_OUTPUT_STDOUT_READ

    @staticmethod
    def _base_scalars(stdout: str) -> dict[str, Any]:
        """Seed the scalar-results dict with the shared ``job_done`` / ``walltime`` keys.

        ``JOB DONE`` never spans a line, so the whole-text substring test is
        equivalent to scanning line by line. Tool-specific parsers fill in the
        walltime (and their own extra keys) from here.
        """
        return {
            "job_done": "JOB DONE" in stdout,
            "walltime": None,
            "walltime_units": "s",
        }


def _time_string_to_seconds(time_str: str) -> float:
    """Convert strings like ``1d 2h 3m 4s`` / ``3m 4s`` / ``4.5s`` to seconds."""
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


__all__ = ("KoopmansStdoutParser", "_time_string_to_seconds")
