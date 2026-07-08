"""Parser for Quantum ESPRESSO ``wann2kcp.x`` output files.

wann2kcp.x emits little structured output: the only signal worth capturing is
whether the run finished (``JOB DONE`` in stdout) and, when present, the
wall-time line. This is the same minimal "did it finish" check as the
``job_done`` extraction in :class:`~aiida_koopmans.parsers.kcp.KcpParser`.
"""

from __future__ import annotations

from typing import Any

from aiida import orm

from aiida_koopmans.parsers.base import KoopmansStdoutParser, _time_string_to_seconds


class Wann2kcpParser(KoopmansStdoutParser):
    """Parse the stdout of a ``Wann2kcpCalculation``.

    Emits a single ``output_parameters`` Dict with a ``job_done`` flag and an
    optional ``walltime`` (seconds). Returns ``ERROR_OUTPUT_STDOUT_INCOMPLETE``
    when the run did not reach ``JOB DONE``.
    """

    def parse(self, **kwargs: Any):
        """Entry point called by AiiDA after the CalcJob finishes."""
        stdout = self._read_stdout()
        if not isinstance(stdout, str):
            return stdout

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
        results = KoopmansStdoutParser._base_scalars(stdout)
        for line in stdout.splitlines():
            if line.strip().startswith("WANN2KCP"):
                tokens = line.split()
                if len(tokens) >= 2:
                    try:
                        results["walltime"] = _time_string_to_seconds(tokens[-2])
                    except ValueError:
                        pass
        return results
