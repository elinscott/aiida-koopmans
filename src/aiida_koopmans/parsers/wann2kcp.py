"""Parser for Quantum ESPRESSO ``wann2kcp.x`` output files.

wann2kcp.x emits little structured output: the only signal worth capturing is
whether the run finished (``JOB DONE`` in stdout) and, when present, the
wall-time line. This is the same minimal "did it finish" check as the
``job_done`` extraction in :class:`~aiida_koopmans.parsers.kcp.KcpParser`.
"""

from __future__ import annotations

from typing import Any

from aiida import orm
from aiida.parsers import Parser


class Wann2kcpParser(Parser):
    """Parse the stdout of a ``Wann2kcpCalculation``.

    Emits a single ``output_parameters`` Dict with a ``job_done`` flag and an
    optional ``walltime`` (seconds). Returns ``ERROR_OUTPUT_STDOUT_INCOMPLETE``
    when the run did not reach ``JOB DONE``.
    """

    def parse(self, **kwargs: Any):
        """Entry point called by AiiDA after the CalcJob finishes."""
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

        parsed = self._parse_stdout(stdout)
        self.out("output_parameters", orm.Dict(dict=parsed))

        if not parsed.get("job_done", False):
            return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE

        return None

    @staticmethod
    def _parse_stdout(stdout: str) -> dict[str, Any]:
        """Extract ``job_done`` and ``walltime`` from the ``.wko`` text.

        The wall-time line in wann2kcp.x stdout starts with the (upper-cased)
        program name and has the time token as the second-to-last field, e.g.
        ``WANN2KCP   :      0.12s CPU      0.15s WALL``. Take the
        second-to-last token of the program line and parse it as a Fortran
        time string.
        """
        results: dict[str, Any] = {
            "job_done": False,
            "walltime": None,
            "walltime_units": "s",
        }
        for line in stdout.splitlines():
            if "JOB DONE" in line:
                results["job_done"] = True
            if line.strip().startswith("WANN2KCP"):
                tokens = line.split()
                if len(tokens) >= 2:
                    try:
                        results["walltime"] = _time_string_to_seconds(tokens[-2])
                    except ValueError:
                        pass
        return results


def _time_string_to_seconds(time_str: str) -> float:
    """Convert strings like ``1d 2h 3m 4s`` / ``3m4s`` / ``4.5s`` to seconds.

    wann2kcp.x reports the wall time as a single compact token (e.g.
    ``0.15s``); the multi-unit handling keeps this robust to the longer
    ``Nm Ms`` shape kcp.x uses for longer runs.
    """
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
