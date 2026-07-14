"""Parser for Quantum ESPRESSO ``wann2kcp.x`` output files.

wann2kcp.x emits little structured stdout: the only signals worth capturing
are whether the run finished (``JOB DONE``) and, when present, the wall-time
line.

The real products are the folded ``evcw*.dat`` wavefunctions. In
``wannier2kcp`` mode the parser re-emits each retrieved one as a
``SinglefileData`` output (``evcw`` / ``evcw1`` / ``evcw2``) so the files are
first-class nodes: downstream ``merge_evc.x`` / ``kcp.x`` steps consume them
as enumerated inputs instead of reaching into the remote folder by filename
convention.
"""

from __future__ import annotations

from typing import Any

from aiida import orm

from aiida_koopmans.parsers.base import KoopmansStdoutParser, time_string_to_seconds


class Wann2kcpParser(KoopmansStdoutParser):
    """Parse the output of a ``Wann2kcpCalculation``.

    Emits ``output_parameters`` (``job_done`` flag, optional ``walltime``)
    plus, in ``wannier2kcp`` mode, one ``SinglefileData`` output per
    retrieved ``evcw*.dat`` wavefunction. Returns
    ``ERROR_OUTPUT_STDOUT_INCOMPLETE`` when the run did not reach ``JOB
    DONE`` and ``ERROR_OUTPUT_EVC_MISSING`` when a completed ``wannier2kcp``
    run produced no wavefunction at all.
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

        return self._attach_evcw_outputs(self.retrieved)

    def _attach_evcw_outputs(self, retrieved: orm.FolderData):
        """Re-emit the retrieved ``evcw*.dat`` files as ``SinglefileData`` outputs.

        Which files exist depends on the spin mode (``evcw.dat`` for a
        spin-resolved run, ``evcw1.dat`` + ``evcw2.dat`` otherwise), so every
        present one is attached under its extension-less name. A completed
        ``wannier2kcp`` run that produced none is an error; ``ks2kcp`` mode
        emits no wavefunction outputs.
        """
        from aiida_koopmans.calculations.wann2kcp import Wann2kcpCalculation

        # The stored input Dict is the caller's raw payload; keys are only
        # lower-cased at submission time, so normalise here too.
        parameters = (
            {key.lower(): value for key, value in self.node.inputs.parameters.get_dict().items()}
            if "parameters" in self.node.inputs
            else {}
        )
        wan_mode = parameters.get("wan_mode", Wann2kcpCalculation._DEFAULTS["wan_mode"])
        if wan_mode != "wannier2kcp":
            return None

        names = retrieved.base.repository.list_object_names()
        attached = False
        for filename in Wann2kcpCalculation._EVCW_FILES:
            if filename not in names:
                continue
            with retrieved.base.repository.open(filename, "rb") as handle:
                self.out(
                    filename.removesuffix(".dat"), orm.SinglefileData(handle, filename=filename)
                )
            attached = True

        if not attached:
            return self.exit_codes.ERROR_OUTPUT_EVC_MISSING
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
                        results["walltime"] = time_string_to_seconds(tokens[-2])
                    except ValueError:
                        pass
        return results
