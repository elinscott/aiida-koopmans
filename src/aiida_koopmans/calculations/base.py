"""Shared scaffolding for the Koopmans QE-fork CalcJobs.

Provides, for kcp.x / kcw.x / wann2kcp.x / merge_evc.x: the common exit
codes, the ``-in <input>`` :class:`~aiida.common.CodeInfo` block, the
``additional_retrieve_list`` settings hook, the Fortran ``&NL ... /``
namelist renderer, and the ``file_alpharef.txt`` writer.

Two tiers:

* :class:`KoopmansCalculation` -- the universal base. Declares only the
  ``ERROR_NO_RETRIEVED_FOLDER`` exit code (shared by every plugin, including
  ``merge_evc.x`` which reads no stdout), plus the code-info builder and the
  retrieve-list settings hook.
* :class:`KoopmansStdoutCalculation` -- for the namelist-driven tools that
  parse a Fortran stdout (kcp.x / kcw.x / wann2kcp.x). Adds the three
  stdout exit codes (302 / 303 / 310, with the tool name interpolated from
  ``_TOOL_NAME``), the namelist renderer, and the alpha-file writer.

``merge_evc.x`` deliberately extends only :class:`KoopmansCalculation`: it
takes no namelist and its 302 exit code means "merged file missing", not
"stdout missing", so it must not inherit the stdout trio.
"""

from __future__ import annotations

import abc
from typing import ClassVar

from aiida.common import CodeInfo
from aiida.engine import CalcJob
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry


class KoopmansCalculation(CalcJob, abc.ABC):
    """Universal base for the Koopmans QE-fork CalcJobs.

    Provides the one exit code every plugin shares, the ``CodeInfo`` builder,
    and the ``additional_retrieve_list`` settings hook. Subclasses set
    ``_OUTPUT_FILE`` (stdout name) and, for the ``-in <input>`` default
    command line, ``_INPUT_FILE``.

    Abstract: ``prepare_for_submission`` is left unimplemented, so this class
    (and the equally-abstract :class:`KoopmansStdoutCalculation`) cannot be
    instantiated -- only the concrete plugin subclasses can.
    """

    # Human-readable binary name, interpolated into the shared stdout exit
    # messages of :class:`KoopmansStdoutCalculation`. Subclasses override.
    _TOOL_NAME: ClassVar[str] = "the QE binary"

    # Set by every concrete subclass; declared here so the shared
    # ``CodeInfo`` builder can reference them.
    _INPUT_FILE: ClassVar[str]
    _OUTPUT_FILE: ClassVar[str]

    @classmethod
    def define(cls, spec):
        """Declare the exit code shared by every Koopmans QE-fork plugin."""
        super().define(spec)
        spec.exit_code(
            301,
            "ERROR_NO_RETRIEVED_FOLDER",
            message="The retrieved folder is missing.",
            invalidates_cache=True,
        )

    def _make_code_info(self, cmdline_params: list[str] | None = None) -> CodeInfo:
        """Build the single-code ``CodeInfo`` for this calc.

        Defaults the command line to ``["-in", self._INPUT_FILE]`` (the shape
        every namelist-driven tool uses); ``merge_evc.x`` passes its own
        ``cmdline_params`` instead.
        """
        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        base_params = cmdline_params if cmdline_params is not None else ["-in", self._INPUT_FILE]
        # Parallelization flags (e.g. ``-npool``) ride ``settings.cmdline`` and,
        # as in aiida-quantumespresso, precede the ``-in <input>`` block.
        code_info.cmdline_params = self._cmdline_from_settings() + base_params
        code_info.stdout_name = self._OUTPUT_FILE
        return code_info

    def _additional_retrieve_list(self) -> list[str]:
        """Return the ``additional_retrieve_list`` from ``settings`` (or an empty list)."""
        if "settings" in self.inputs:
            return list(self.inputs.settings.get_dict().get("additional_retrieve_list", []))
        return []

    def _cmdline_from_settings(self) -> list[str]:
        """Return the ``cmdline`` list from ``settings`` (or an empty list)."""
        if "settings" in self.inputs:
            return list(self.inputs.settings.get_dict().get("cmdline", []))
        return []

    @abc.abstractmethod
    def prepare_for_submission(self, folder):
        """Render the input files and build the ``CalcInfo`` for this calc.

        Abstract: every concrete plugin renders its own namelists / side
        files, so the shared bases leave this unimplemented.
        """


class KoopmansStdoutCalculation(KoopmansCalculation):
    """Base for the namelist-driven Koopmans plugins that parse a Fortran stdout.

    Adds the three stdout exit codes (with ``_TOOL_NAME`` interpolated), the
    ``&NL ... /`` namelist renderer, and the ``file_alpharef.txt`` writer.
    """

    # Screening-parameter side-file names (kcp.x and the kcw.x ``ham`` mode).
    _ALPHAREF_FILE: ClassVar[str] = "file_alpharef.txt"
    _ALPHAREF_EMPTY_FILE: ClassVar[str] = "file_alpharef_empty.txt"

    @classmethod
    def define(cls, spec):
        """Add the stdout exit codes shared by the kcp.x / kcw.x / wann2kcp.x plugins."""
        super().define(spec)
        spec.exit_code(
            302,
            "ERROR_OUTPUT_STDOUT_MISSING",
            message=f"The {cls._TOOL_NAME} stdout file was not retrieved.",
            invalidates_cache=True,
        )
        spec.exit_code(
            303,
            "ERROR_OUTPUT_STDOUT_READ",
            message=f"The {cls._TOOL_NAME} stdout could not be read.",
            invalidates_cache=True,
        )
        spec.exit_code(
            310,
            "ERROR_OUTPUT_STDOUT_INCOMPLETE",
            message=f"The {cls._TOOL_NAME} stdout ends before ``JOB DONE``.",
            invalidates_cache=True,
        )

    @staticmethod
    def render_namelist(name: str, options: dict) -> str:
        """Render a single Fortran namelist (``&NAME ... /``).

        Values are formatted with aiida-quantumespresso's
        ``convert_input_to_namelist_entry`` so booleans become ``.true.`` /
        ``.false.``, strings are quoted, and paths keep their trailing slash.
        """
        lines = [f"&{name}\n"]
        for key, val in options.items():
            lines.append(convert_input_to_namelist_entry(key, val))
        lines.append("/\n")
        return "".join(lines)

    @staticmethod
    def _write_alpha_file(folder, alphas: list[float], filename: str) -> None:
        """Write screening parameters in kcp.x/kcw.x ``file_alpharef[_empty].txt`` format.

        Format: first line is the orbital count, subsequent lines are
        ``{index} {alpha} 1.0`` (1-indexed). An empty ``alphas`` list yields a
        header-only file (count ``0`` and no orbital lines).
        """
        content = f"{len(alphas)}\n"
        content += "".join(f"{i + 1} {a} 1.0\n" for i, a in enumerate(alphas))
        with folder.open(filename, "w", encoding="utf-8") as handle:
            handle.write(content)


__all__ = ("KoopmansCalculation", "KoopmansStdoutCalculation")
