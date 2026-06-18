"""CalcJob for Quantum ESPRESSO's ``wann2kcp.x`` (Koopmans fork).

``wann2kcp.x`` converts Wannier90 (or plain Kohn-Sham) wavefunctions into the
``kcp.x``-friendly ``evcw`` format that the Koopmans Delta-SCF supercell run
consumes. It is part of the Koopmans QE fork and has no upstream
``aiida-quantumespresso`` equivalent, so this plugin is a standalone
``CalcJob`` (it is intentionally not a subclass of any vanilla QE post-
processing calculation).

The input is a single Fortran ``&inputpp`` namelist written to a ``.wki`` file
(the ASE ``Wann2KCP`` writer emits the same namelist name; see
``ase_koopmans.io.espresso._x2y.write_x2y_in``). Namelist value formatting is
shared with the ``kcp.x`` plugin via
``aiida_quantumespresso.utils.convert.convert_input_to_namelist_entry`` so
booleans render as ``.true.`` / ``.false.``, strings are quoted, and paths
keep their trailing slash.

Two modes are supported:

* ``wannier2kcp`` (the mode the FoldToSupercell workgraph needs) -- folds
  Wannier orbitals into the supercell. The caller must stage four inputs into
  the work directory (see ``KcpCalculation`` for the symlink idiom): the
  ``<seedname>_hr.dat`` Hamiltonian, the nscf ``outdir`` (recursive symlink),
  ``<seedname>.nnkp``, and ``<seedname>.chk``. The run writes ``evcw.dat`` for
  a spin-polarized calculation or ``evcw1.dat`` + ``evcw2.dat`` for a
  non-spin-polarized one.
* ``ks2kcp`` -- converts plain Kohn-Sham orbitals; no Wannier inputs required.

Downstream steps pick up the ``evcw*`` files from ``remote_folder`` (they are
also retrieved into the ``retrieved`` folder for provenance / a merge step).
"""

from __future__ import annotations

from typing import ClassVar

from aiida.common import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.orm import Dict, RemoteData
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry


class Wann2kcpCalculation(CalcJob):
    """AiiDA plugin for running ``wann2kcp.x`` from the Koopmans Quantum ESPRESSO fork."""

    _INPUT_FILE = "aiida.wki"
    _OUTPUT_FILE = "aiida.wko"
    _DEFAULT_OUTDIR = "TMP"
    _DEFAULT_PREFIX = "kc"
    _DEFAULT_SEEDNAME = "wannier90"

    # The wann2kcp input is a single Fortran namelist; the ASE writer and the
    # QE reader both spell it ``&inputpp`` (lowercase in the file, parsed as
    # ``data['inputpp']``). See ``ase_koopmans.io.espresso._x2y``.
    _NAMELIST = "INPUTPP"

    # Keys the CalcJob owns; users cannot set them in ``parameters``. ``outdir``
    # is owned because callers stage the nscf scratch under a fixed work-dir
    # path via a recursive symlink, and ``prefix`` matches that scratch tree.
    _BLOCKED_KEYS: ClassVar[frozenset[str]] = frozenset({"outdir", "prefix"})

    # The full set of valid wann2kcp keys, mirrored from the legacy
    # ``Wann2KCPSettingsDict`` (``koopmans/settings/_wann2kcp.py``). Unknown
    # keys are rejected so a typo doesn't silently produce a broken input.
    _VALID_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "outdir",
            "prefix",
            "seedname",
            "wan_mode",
            "spin_component",
            "gamma_trick",
            "print_rho",
            "wannier_plot",
            "wannier_plot_list",
        }
    )

    # Legacy defaults from ``Wann2KCPSettingsDict``. ``outdir`` is owned by the
    # CalcJob (see ``_BLOCKED_KEYS``) so its default lives in ``_DEFAULT_OUTDIR``
    # and is injected, not defaulted here.
    _DEFAULTS: ClassVar[dict[str, str]] = {
        "prefix": _DEFAULT_PREFIX,
        "seedname": _DEFAULT_SEEDNAME,
        "wan_mode": "wannier2kcp",
    }

    @classmethod
    def define(cls, spec):
        """Declare the inputs, outputs, and exit codes for the CalcJob."""
        super().define(spec)

        spec.input(
            "parameters",
            valid_type=Dict,
            required=False,
            help=(
                "Flat ``&inputpp`` namelist dictionary, e.g. "
                "``{'seedname': 'wannier90', 'wan_mode': 'wannier2kcp', "
                "'spin_component': 'up'}``. Keys are case-insensitive. Valid "
                "keys: ``seedname, wan_mode, spin_component, gamma_trick, "
                "print_rho, wannier_plot, wannier_plot_list`` (``outdir`` and "
                "``prefix`` are owned by the CalcJob). Defaults: "
                "``prefix=kc, seedname=wannier90, wan_mode=wannier2kcp``."
            ),
        )
        spec.input(
            "parent_folder",
            valid_type=RemoteData,
            required=False,
            help=(
                "Remote folder of the upstream nscf run. Its ``outdir`` tree is "
                "recursively symlinked into ``./TMP/`` so wann2kcp.x can read "
                "the Bloch wavefunctions. Required for ``wan_mode='wannier2kcp'`` "
                "and ``ks2kcp``."
            ),
        )
        spec.input(
            "settings",
            valid_type=Dict,
            required=False,
            help="Optional CalcJob-level settings (cmdline overrides, extra retrieve paths).",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "koopmans.wann2kcp"
        spec.inputs["metadata"]["options"]["input_filename"].default = cls._INPUT_FILE
        spec.inputs["metadata"]["options"]["output_filename"].default = cls._OUTPUT_FILE
        spec.inputs["metadata"]["options"]["withmpi"].default = True
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1}

        spec.output(
            "output_parameters",
            valid_type=Dict,
            required=True,
            help="Scalar results: ``job_done`` flag and ``walltime``.",
        )

        spec.exit_code(301, "ERROR_NO_RETRIEVED_FOLDER", message="The retrieved folder is missing.")
        spec.exit_code(
            302,
            "ERROR_OUTPUT_STDOUT_MISSING",
            message="The wann2kcp.x stdout file was not retrieved.",
        )
        spec.exit_code(
            303, "ERROR_OUTPUT_STDOUT_READ", message="The wann2kcp.x stdout could not be read."
        )
        spec.exit_code(
            310,
            "ERROR_OUTPUT_STDOUT_INCOMPLETE",
            message="The wann2kcp.x stdout ends before ``JOB DONE``.",
        )

    def prepare_for_submission(self, folder):
        """Render the ``.wki`` input file and build the ``CalcInfo``."""
        raw = self.inputs.parameters.get_dict() if "parameters" in self.inputs else {}
        parameters = self._normalize_parameters(raw)
        self._inject_owned_keys(parameters)

        content = self._render_namelist(parameters)
        with folder.open(self._INPUT_FILE, "w", encoding="utf-8") as handle:
            handle.write(content)

        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        code_info.cmdline_params = ["-in", self._INPUT_FILE]
        code_info.stdout_name = self._OUTPUT_FILE

        calc_info = CalcInfo()
        calc_info.codes_info = [code_info]
        calc_info.remote_symlink_list = self._build_remote_symlink_list()
        calc_info.retrieve_list = self._build_retrieve_list(parameters)

        return calc_info

    # ------------------------------------------------------------------
    # prepare_for_submission helpers
    # ------------------------------------------------------------------

    def _inject_owned_keys(self, parameters: dict) -> None:
        """Inject the CalcJob-owned ``outdir`` and fill in legacy defaults.

        ``outdir`` always points at the fixed ``./TMP/`` work-dir path the nscf
        scratch is symlinked into (see ``_build_remote_symlink_list``). The
        remaining defaults (``prefix``, ``seedname``, ``wan_mode``) match the
        legacy ``Wann2KCPSettingsDict`` and are only set when the caller did
        not supply them.
        """
        parameters["outdir"] = f"./{self._DEFAULT_OUTDIR}/"
        for key, default in self._DEFAULTS.items():
            parameters.setdefault(key, default)

    def _build_remote_symlink_list(self) -> list[tuple[str, str, str]]:
        """Recursively symlink the parent nscf ``outdir`` into ``./TMP/``.

        wann2kcp.x reads the Bloch wavefunctions from its ``outdir``; the
        legacy folding workflow links the nscf scratch in recursively (see
        ``_folding.py``). A directory-level symlink is sufficient here -- unlike
        the kcp.x case there are no per-file overlays -- so this stays a single
        entry. The ``<seedname>.nnkp``, ``<seedname>.chk`` and ``*_hr.dat``
        inputs are staged by the consuming workgraph (Phase B), not by this
        CalcJob, because their provenance lives on different upstream nodes.
        """
        if "parent_folder" not in self.inputs:
            return []
        parent = self.inputs.parent_folder
        source = parent.get_remote_path()
        return [(parent.computer.uuid, source, self._DEFAULT_OUTDIR)]

    def _build_retrieve_list(self, parameters: dict) -> list[str]:
        """Retrieve stdout plus the ``evcw`` wavefunction files for provenance.

        ``wannier2kcp`` writes ``evcw.dat`` when ``spin_component`` is set
        (spin-polarized) or ``evcw1.dat`` + ``evcw2.dat`` otherwise. We retrieve
        all three names unconditionally -- AiiDA silently skips any that the run
        did not produce -- so a downstream merge step finds whatever was
        written without the CalcJob having to know the spin mode. The files
        also remain available on ``remote_folder`` for symlink-based chaining.
        """
        retrieve_list: list[str] = [self._OUTPUT_FILE]
        if parameters.get("wan_mode", "wannier2kcp") == "wannier2kcp":
            retrieve_list += ["evcw.dat", "evcw1.dat", "evcw2.dat"]
        if "settings" in self.inputs:
            extra = self.inputs.settings.get_dict().get("additional_retrieve_list", [])
            retrieve_list.extend(extra)
        return retrieve_list

    # ------------------------------------------------------------------
    # Input-rendering helpers
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_parameters(cls, parameters: dict) -> dict:
        """Lowercase keys, reject blocked and unknown keys.

        Unlike the kcp.x plugin the wann2kcp input is a *flat* dict (a single
        namelist), so there is no namelist nesting to normalize -- just the
        keys.
        """
        if not isinstance(parameters, dict):
            raise ValueError(f"``parameters`` must be a dict, got {type(parameters).__name__}.")
        normalized: dict = {}
        for key, val in parameters.items():
            k = key.lower()
            if k in cls._BLOCKED_KEYS:
                raise ValueError(
                    f"Parameter ``{k}`` is set by the CalcJob and cannot be overridden."
                )
            if k not in cls._VALID_KEYS:
                raise ValueError(
                    f"Unknown wann2kcp parameter ``{k}``. Valid keys: "
                    f"{', '.join(sorted(cls._VALID_KEYS - cls._BLOCKED_KEYS))}."
                )
            normalized[k] = val
        return normalized

    @classmethod
    def _render_namelist(cls, parameters: dict) -> str:
        """Render the single ``&inputpp`` namelist for the ``.wki`` file."""
        lines = [f"&{cls._NAMELIST}\n"]
        for key, val in parameters.items():
            lines.append(convert_input_to_namelist_entry(key, val))
        lines.append("/\n")
        return "".join(lines)
