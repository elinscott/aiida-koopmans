"""CalcJob for Quantum ESPRESSO's ``wann2kcp.x`` (Koopmans fork).

``wann2kcp.x`` converts Wannier90 (or plain Kohn-Sham) wavefunctions into the
``kcp.x``-friendly ``evcw`` format that the Koopmans Delta-SCF supercell run
consumes. It is part of the Koopmans QE fork and has no upstream
``aiida-quantumespresso`` equivalent, so this plugin is a standalone
``CalcJob`` (it is intentionally not a subclass of any vanilla QE post-
processing calculation).

The input is a single Fortran ``&inputpp`` namelist written to a ``.wki``
file. Namelist value formatting is shared with the ``kcp.x`` plugin via
``aiida_quantumespresso.utils.convert.convert_input_to_namelist_entry`` so
booleans render as ``.true.`` / ``.false.``, strings are quoted, and paths
keep their trailing slash.

Two modes are supported:

* ``wannier2kcp`` (the mode the FoldToSupercell workgraph needs) -- folds
  Wannier orbitals into the supercell. Four inputs are staged into the work
  directory: the nscf ``outdir`` (``parent_folder``, symlinked as ``TMP``),
  ``<seedname>.nnkp`` (``nnkp_file``), and ``<seedname>.chk`` +
  ``<seedname>_hr.dat`` (both copied out of ``wannier_folder``, the wannier90
  ``retrieved`` folder). The run writes ``evcw.dat`` for a spin-polarized
  calculation or ``evcw1.dat`` + ``evcw2.dat`` for a non-spin-polarized one.
* ``ks2kcp`` -- converts plain Kohn-Sham orbitals; no Wannier inputs required.

Downstream steps pick up the ``evcw*`` files from ``remote_folder`` (they are
also retrieved into the ``retrieved`` folder for provenance / a merge step).
"""

from __future__ import annotations

from typing import ClassVar

from aiida.common import CalcInfo
from aiida.orm import Dict, RemoteData, SinglefileData

from aiida_koopmans.calculations.base import KoopmansStdoutCalculation


class Wann2kcpCalculation(KoopmansStdoutCalculation):
    """AiiDA plugin for running ``wann2kcp.x`` from the Koopmans Quantum ESPRESSO fork."""

    _TOOL_NAME = "wann2kcp.x"

    _INPUT_FILE = "aiida.wki"
    _OUTPUT_FILE = "aiida.wko"
    _DEFAULT_OUTDIR = "TMP"
    # ``prefix`` must match the prefix of the upstream pw.x nscf whose scratch
    # is symlinked in as ``TMP`` — aiida-quantumespresso's ``PwCalculation``
    # hard-codes ``_PREFIX = "aiida"``, so wann2kcp.x must look for
    # ``TMP/aiida.save``. (Legacy koopmans used ``kc`` because its pw.x runs
    # used that prefix.)
    _DEFAULT_PREFIX = "aiida"
    _DEFAULT_SEEDNAME = "wannier90"
    # Filenames inside ``wannier_folder`` (the wannier90 ``retrieved``
    # FolderData): aiida-wannier90's ``Wannier90Calculation`` hard-codes its
    # seedname to ``aiida``, so the checkpoint and Hamiltonian arrive as
    # ``aiida.chk`` / ``aiida_hr.dat``. Overridable via
    # ``settings['wannier_source_seedname']``.
    _WANNIER_SOURCE_SEEDNAME = "aiida"

    # The wann2kcp input is a single Fortran namelist spelled ``&inputpp``
    # (lowercase in the file, parsed as ``data['inputpp']``).
    _NAMELIST = "INPUTPP"

    # Keys the CalcJob owns; users cannot set them in ``parameters``. ``outdir``
    # is owned because callers stage the nscf scratch under a fixed work-dir
    # path via a recursive symlink, and ``prefix`` matches that scratch tree.
    _BLOCKED_KEYS: ClassVar[frozenset[str]] = frozenset({"outdir", "prefix"})

    # The full set of valid wann2kcp keys. Unknown keys are rejected so a
    # typo doesn't silently produce a broken input.
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

    # Default namelist values. ``outdir`` is owned by the CalcJob (see
    # ``_BLOCKED_KEYS``) so its default lives in ``_DEFAULT_OUTDIR`` and is
    # injected, not defaulted here.
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
            "nnkp_file",
            valid_type=SinglefileData,
            required=False,
            help=(
                "The ``.nnkp`` file emitted by the wannier90 post-processing "
                "(``-pp``) run. Copied into the work directory as "
                "``<seedname>.nnkp``. Required for ``wan_mode='wannier2kcp'``."
            ),
        )
        spec.input(
            "wannier_folder",
            valid_type=FolderData,
            required=False,
            help=(
                "The wannier90 ``retrieved`` folder holding the checkpoint and "
                "real-space Hamiltonian (``aiida.chk`` must be forced into the "
                "wannier90 retrieve list; ``aiida_hr.dat`` requires "
                "``write_hr``). They are copied into the work directory as "
                "``<seedname>.chk`` / ``<seedname>_hr.dat``. Required for "
                "``wan_mode='wannier2kcp'``."
            ),
        )
        spec.input(
            "settings",
            valid_type=Dict,
            required=False,
            help=(
                "Optional CalcJob-level settings: ``additional_retrieve_list`` "
                "adds retrieve paths; ``wannier_source_seedname`` overrides the "
                "seedname used to locate files inside ``wannier_folder`` "
                "(default ``aiida``)."
            ),
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

    def prepare_for_submission(self, folder):
        """Render the ``.wki`` input file and build the ``CalcInfo``."""
        raw = self.inputs.parameters.get_dict() if "parameters" in self.inputs else {}
        parameters = self._normalize_parameters(raw)
        self._inject_owned_keys(parameters)

        content = self._render_namelist(parameters)
        with folder.open(self._INPUT_FILE, "w", encoding="utf-8") as handle:
            handle.write(content)

        calc_info = CalcInfo()
        calc_info.codes_info = [self._make_code_info()]
        calc_info.remote_symlink_list = self._build_remote_symlink_list()
        calc_info.local_copy_list = self._build_local_copy_list(parameters)
        calc_info.retrieve_list = self._build_retrieve_list(parameters)

        return calc_info

    # ------------------------------------------------------------------
    # prepare_for_submission helpers
    # ------------------------------------------------------------------

    def _inject_owned_keys(self, parameters: dict) -> None:
        """Inject the CalcJob-owned ``outdir`` and fill in defaults.

        ``outdir`` always points at the fixed ``./TMP/`` work-dir path the nscf
        scratch is symlinked into (see ``_build_remote_symlink_list``). The
        remaining defaults (``prefix``, ``seedname``, ``wan_mode``) are only
        set when the caller did not supply them.
        """
        parameters["outdir"] = f"./{self._DEFAULT_OUTDIR}/"
        for key, default in self._DEFAULTS.items():
            parameters.setdefault(key, default)

    def _build_remote_symlink_list(self) -> list[tuple[str, str, str]]:
        """Symlink the parent nscf scratch into ``./TMP/``.

        wann2kcp.x reads the Bloch wavefunctions from ``<outdir>/<prefix>.save``.
        A directory-level symlink is sufficient here -- unlike the kcp.x case
        there are no per-file overlays -- so this stays a single entry. The
        parent is an aiida-quantumespresso pw.x run, whose scratch lives under
        ``<workdir>/out/`` (``PwCalculation._OUTPUT_SUBFOLDER``), so the ``out``
        subdirectory is what lands at ``TMP`` — combined with the owned
        ``prefix = aiida`` this resolves to ``TMP/aiida.save``.
        """
        if "parent_folder" not in self.inputs:
            return []
        parent = self.inputs.parent_folder
        source = f"{parent.get_remote_path()}/out"
        return [(parent.computer.uuid, source, self._DEFAULT_OUTDIR)]

    def _build_local_copy_list(self, parameters: dict) -> list[tuple[str, str, str]]:
        """Copy the Wannier inputs (``.nnkp``, ``.chk``, ``_hr.dat``) into place.

        The three files live on different upstream nodes (the ``-pp`` run's
        ``nnkp_file`` output; the wannier90 ``retrieved`` folder), so they are
        separate inputs rather than a single parent. Destination names follow
        the ``seedname`` the namelist declares; source names inside
        ``wannier_folder`` follow the wannier90 seedname (``aiida`` upstream,
        overridable via ``settings['wannier_source_seedname']``).
        """
        seedname = parameters.get("seedname", self._DEFAULT_SEEDNAME)
        copy_list: list[tuple[str, str, str]] = []
        if "nnkp_file" in self.inputs:
            nnkp = self.inputs.nnkp_file
            copy_list.append((nnkp.uuid, nnkp.filename, f"{seedname}.nnkp"))
        if "wannier_folder" in self.inputs:
            folder = self.inputs.wannier_folder
            source_seedname = self._WANNIER_SOURCE_SEEDNAME
            if "settings" in self.inputs:
                source_seedname = self.inputs.settings.get_dict().get(
                    "wannier_source_seedname", source_seedname
                )
            copy_list.append((folder.uuid, f"{source_seedname}.chk", f"{seedname}.chk"))
            copy_list.append((folder.uuid, f"{source_seedname}_hr.dat", f"{seedname}_hr.dat"))
        return copy_list

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
        retrieve_list.extend(self._additional_retrieve_list())
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
        return cls.render_namelist(cls._NAMELIST, parameters)
