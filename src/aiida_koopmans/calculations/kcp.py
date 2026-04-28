"""CalcJob for Quantum ESPRESSO's kcp.x (Koopmans-modified Car-Parrinello).

kcp.x is the Koopmans-spectral-functional variant of QE's CP code. It shares the
binary name and uses CG internally, but the algorithm is fundamentally different
from vanilla cp.x — this plugin is intentionally not a subclass of
``aiida_quantumespresso.calculations.cp.CpCalculation``.
"""

from __future__ import annotations

import copy
from pathlib import PurePosixPath
from typing import ClassVar

from aiida.common import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.orm import ArrayData, Dict, RemoteData, StructureData
from aiida.plugins import DataFactory
from aiida_quantumespresso.utils.convert import convert_input_to_namelist_entry

UpfData = DataFactory("pseudo.upf")


class KcpCalculation(CalcJob):
    """AiiDA plugin for running kcp.x, the Koopmans-modified CP code in Quantum ESPRESSO."""

    _INPUT_FILE = "aiida.cpi"
    _OUTPUT_FILE = "aiida.cpo"
    _CRASH_FILE = "CRASH"
    _PREFIX = "aiida"
    _OUTPUT_SUBFOLDER = "out"
    _PSEUDO_SUBFOLDER = "pseudo"
    _ALPHAREF_FILE = "file_alpharef.txt"
    _ALPHAREF_EMPTY_FILE = "file_alpharef_empty.txt"

    # Canonical kcp.x namelist order. Namelists outside this list are emitted
    # at the end of the input file in insertion order.
    _NAMELIST_ORDER = ("CONTROL", "SYSTEM", "ELECTRONS", "IONS", "CELL", "EE", "NKSIC")

    # Keys the CalcJob owns; users cannot set them in ``parameters``.
    _BLOCKED_KEYS: ClassVar[dict[str, frozenset[str]]] = {
        "CONTROL": frozenset({"outdir", "pseudo_dir", "prefix"}),
        "SYSTEM": frozenset({"nat", "ntyp", "ibrav"}),
    }

    @classmethod
    def define(cls, spec):
        """Declare the inputs, outputs, and exit codes for the CalcJob."""
        super().define(spec)

        spec.input(
            "structure",
            valid_type=StructureData,
            help="The input structure.",
        )
        spec.input(
            "parameters",
            valid_type=Dict,
            help=(
                "Nested namelist dictionary, e.g. "
                "``{'CONTROL': {...}, 'SYSTEM': {...}, 'ELECTRONS': {...}, "
                "'NKSIC': {...}, ...}``. Namelist names are case-insensitive."
            ),
        )
        spec.input(
            "alphas",
            valid_type=Dict,
            required=False,
            help=(
                "Orbital-dependent screening parameters. Dict with keys "
                "``'filled'`` and ``'empty'`` mapping to flat lists of floats. "
                "Required when the parameters request orbital-dependent screening."
            ),
        )
        spec.input(
            "parent_folder",
            valid_type=RemoteData,
            required=False,
            help=(
                "Remote folder of a prior kcp.x run. Its ``out/`` directory is "
                "symlinked into place so wavefunctions / densities can be reused."
            ),
        )
        spec.input(
            "settings",
            valid_type=Dict,
            required=False,
            help="Optional CalcJob-level settings (cmdline overrides, extra retrieve paths).",
        )
        spec.input_namespace(
            "pseudos",
            valid_type=UpfData,
            dynamic=True,
            required=True,
            help="Mapping of atomic kind name to UpfData.",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "koopmans.kcp"
        spec.inputs["metadata"]["options"]["input_filename"].default = cls._INPUT_FILE
        spec.inputs["metadata"]["options"]["output_filename"].default = cls._OUTPUT_FILE
        spec.inputs["metadata"]["options"]["withmpi"].default = True
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1}

        spec.output(
            "output_parameters",
            valid_type=Dict,
            required=True,
            help="Scalar results: energies, HOMO/LUMO, job_done, walltime, convergence summary.",
        )
        spec.output(
            "output_eigenvalues",
            valid_type=ArrayData,
            required=False,
            help="Kohn-Sham eigenvalues in eV, shape ``(nspin, nbnd)``.",
        )
        spec.output(
            "output_lambdas",
            valid_type=ArrayData,
            required=False,
            help="Hamiltonian lambda matrices (one per spin) in eV, read from hamiltonian*.xml.",
        )
        spec.output(
            "output_bare_lambdas",
            valid_type=ArrayData,
            required=False,
            help="Bare Hamiltonian lambda matrices, present when ``do_bare_eigs=.true.``.",
        )

        spec.exit_code(301, "ERROR_NO_RETRIEVED_FOLDER", message="The retrieved folder is missing.")
        spec.exit_code(
            302, "ERROR_OUTPUT_STDOUT_MISSING", message="The kcp.x stdout file was not retrieved."
        )
        spec.exit_code(
            303, "ERROR_OUTPUT_STDOUT_READ", message="The kcp.x stdout could not be read."
        )
        spec.exit_code(
            310,
            "ERROR_OUTPUT_STDOUT_INCOMPLETE",
            message="The kcp.x stdout ends before ``JOB DONE``.",
        )
        spec.exit_code(
            320,
            "ERROR_OUTPUT_HAM_MISSING",
            message="Expected hamiltonian XML file(s) missing from retrieved folder.",
        )
        spec.exit_code(
            400,
            "ERROR_JOB_NOT_CONVERGED",
            message="kcp.x finished but the outer loop did not converge.",
        )

    def prepare_for_submission(self, folder):
        """Render the input file and build the ``CalcInfo``."""
        parameters = copy.deepcopy(self.inputs.parameters.get_dict())
        parameters = self._normalize_parameters(parameters)

        structure = self.inputs.structure
        pseudos = dict(self.inputs.pseudos)

        self._inject_owned_keys(parameters, structure)
        ndw = int(parameters["CONTROL"]["ndw"])
        nspin = int(parameters["SYSTEM"].get("nspin", 1))
        do_orbdep = bool(parameters["SYSTEM"].get("do_orbdep", False))
        nksic = parameters.get("NKSIC", {})
        odd_nkscalfact = bool(nksic.get("odd_nkscalfact", False))
        do_bare_eigs = bool(nksic.get("do_bare_eigs", False))

        content = (
            self._render_namelists(parameters)
            + self._render_atomic_species(structure, pseudos)
            + self._render_atomic_positions(structure)
            + self._render_cell_parameters(structure)
        )
        with folder.open(self._INPUT_FILE, "w", encoding="utf-8") as handle:
            handle.write(content)

        self._write_alpha_files(folder, do_orbdep=do_orbdep, odd_nkscalfact=odd_nkscalfact)

        local_copy_list = self._build_local_copy_list(structure, pseudos)
        remote_symlink_list = self._build_remote_symlink_list()
        retrieve_list = self._build_retrieve_list()
        retrieve_temporary_list = self._build_retrieve_temporary_list(
            ndw=ndw, nspin=nspin, do_orbdep=do_orbdep, do_bare_eigs=do_bare_eigs
        )

        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        code_info.cmdline_params = ["-in", self._INPUT_FILE]
        code_info.stdout_name = self._OUTPUT_FILE

        calc_info = CalcInfo()
        calc_info.codes_info = [code_info]
        calc_info.local_copy_list = local_copy_list
        calc_info.remote_symlink_list = remote_symlink_list
        calc_info.retrieve_list = retrieve_list
        calc_info.retrieve_temporary_list = retrieve_temporary_list

        return calc_info

    # ------------------------------------------------------------------
    # prepare_for_submission helpers
    # ------------------------------------------------------------------

    def _inject_owned_keys(self, parameters: dict, structure: StructureData) -> None:
        """Set the CONTROL/SYSTEM keys the CalcJob owns (outdir, pseudo_dir, ibrav, ...)."""
        control = parameters.setdefault("CONTROL", {})
        control["outdir"] = f"./{self._OUTPUT_SUBFOLDER}/"
        control["pseudo_dir"] = f"./{self._PSEUDO_SUBFOLDER}/"
        control["prefix"] = self._PREFIX
        control.setdefault("calculation", "cp")
        control.setdefault("ndw", 50)

        system = parameters.setdefault("SYSTEM", {})
        system["ibrav"] = 0
        system["nat"] = len(structure.sites)
        system["ntyp"] = len(structure.kinds)

    def _build_local_copy_list(
        self, structure: StructureData, pseudos: dict
    ) -> list[tuple[str, str, str]]:
        """Assemble ``local_copy_list`` for the pseudopotential files."""
        local_copy_list: list[tuple[str, str, str]] = []
        seen_filenames: dict[str, str] = {}
        for kind in structure.kinds:
            upf = pseudos[kind.name]
            previous_uuid = seen_filenames.get(upf.filename)
            if previous_uuid is None:
                seen_filenames[upf.filename] = upf.uuid
                local_copy_list.append(
                    (upf.uuid, upf.filename, f"{self._PSEUDO_SUBFOLDER}/{upf.filename}")
                )
            elif previous_uuid != upf.uuid:
                raise ValueError(
                    f"Two different UpfData nodes were provided that share the filename "
                    f"``{upf.filename}``. Rename one before resubmission."
                )
        return local_copy_list

    def _write_alpha_files(self, folder, *, do_orbdep: bool, odd_nkscalfact: bool) -> None:
        """Emit ``file_alpharef[_empty].txt`` when orbital-dependent screening is requested."""
        alphas_requested = do_orbdep and odd_nkscalfact
        alphas_provided = "alphas" in self.inputs
        if alphas_requested and not alphas_provided:
            raise ValueError(
                "Parameters request orbital-dependent screening (do_orbdep=.true., "
                "odd_nkscalfact=.true.) but no ``alphas`` input was provided."
            )
        if alphas_provided and not alphas_requested:
            raise ValueError(
                "``alphas`` input was provided but the parameters do not enable orbital-dependent "
                "screening (need do_orbdep=.true. and odd_nkscalfact=.true.)."
            )
        if not alphas_requested:
            return
        alphas = self.inputs.alphas.get_dict()
        self._write_alpha_file(folder, alphas.get("filled", []), self._ALPHAREF_FILE)
        self._write_alpha_file(folder, alphas.get("empty", []), self._ALPHAREF_EMPTY_FILE)

    def _build_remote_symlink_list(self) -> list[tuple[str, str, str]]:
        """If a ``parent_folder`` input is provided, symlink its ``out/`` into place."""
        if "parent_folder" not in self.inputs:
            return []
        parent = self.inputs.parent_folder
        parent_out = str(PurePosixPath(parent.get_remote_path()) / self._OUTPUT_SUBFOLDER)
        return [(parent.computer.uuid, parent_out, self._OUTPUT_SUBFOLDER)]

    def _build_retrieve_list(self) -> list[str]:
        """Files persisted in the ``retrieved`` FolderData: stdout, CRASH, user extras."""
        retrieve_list: list[str] = [self._OUTPUT_FILE, self._CRASH_FILE]
        if "settings" in self.inputs:
            extra = self.inputs.settings.get_dict().get("additional_retrieve_list", [])
            retrieve_list.extend(extra)
        return retrieve_list

    def _build_retrieve_temporary_list(
        self, *, ndw: int, nspin: int, do_orbdep: bool, do_bare_eigs: bool
    ) -> list:
        """Files retrieved into a scratch folder for parsing then discarded.

        Hamiltonian XMLs are intermediate artefacts; the parser turns them into
        ``ArrayData`` outputs, after which the raw XMLs serve no purpose.
        Tuple form ``(remote, '.', depth)`` preserves the
        ``out/<prefix>_<ndw>.save/K00001/`` nesting AiiDA would otherwise flatten.
        """
        if not do_orbdep:
            return []
        ham_dir = f"{self._OUTPUT_SUBFOLDER}/{self._PREFIX}_{ndw}.save/K00001"
        temp_list: list = []
        for ispin in range(1, nspin + 1):
            tag = str(ispin) if nspin > 1 else ""
            names = [f"hamiltonian{tag}.xml", f"hamiltonian_emp{tag}.xml"]
            if do_bare_eigs:
                names += [f"hamiltonian0{tag}.xml", f"hamiltonian0_emp{tag}.xml"]
            for name in names:
                remote_path = f"{ham_dir}/{name}"
                temp_list.append((remote_path, ".", len(remote_path.split("/"))))
        return temp_list

    # ------------------------------------------------------------------
    # Input-rendering helpers
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_parameters(cls, parameters: dict) -> dict:
        """Uppercase namelist names, lowercase keys within, and reject blocked keys."""
        normalized: dict[str, dict] = {}
        for namelist, options in parameters.items():
            nl = namelist.upper()
            if not isinstance(options, dict):
                raise ValueError(
                    f"Namelist ``{namelist}`` must map to a dict, got {type(options).__name__}."
                )
            blocked = cls._BLOCKED_KEYS.get(nl, frozenset())
            row: dict = {}
            for key, val in options.items():
                k = key.lower()
                if k in blocked:
                    raise ValueError(
                        f"Parameter ``{nl}/{k}`` is set by the CalcJob and cannot be overridden."
                    )
                row[k] = val
            normalized[nl] = row
        return normalized

    @classmethod
    def _render_namelists(cls, parameters: dict) -> str:
        """Render namelists in canonical kcp.x order, then any unexpected ones."""
        out: list[str] = []
        rendered: set[str] = set()
        for nl in cls._NAMELIST_ORDER:
            options = parameters.get(nl)
            if not options:
                continue
            out.append(cls._render_one_namelist(nl, options))
            rendered.add(nl)
        for nl, options in parameters.items():
            if nl in rendered or not options:
                continue
            out.append(cls._render_one_namelist(nl, options))
        return "".join(out)

    @staticmethod
    def _render_one_namelist(name: str, options: dict) -> str:
        lines = [f"&{name}\n"]
        for key, val in options.items():
            lines.append(convert_input_to_namelist_entry(key, val))
        lines.append("/\n")
        return "".join(lines)

    @staticmethod
    def _render_atomic_species(structure: StructureData, pseudos: dict) -> str:
        lines = ["ATOMIC_SPECIES\n"]
        for kind in structure.kinds:
            upf = pseudos[kind.name]
            lines.append(f"  {kind.name}  {kind.mass:.6f}  {upf.filename}\n")
        return "".join(lines)

    @staticmethod
    def _render_atomic_positions(structure: StructureData) -> str:
        lines = ["ATOMIC_POSITIONS angstrom\n"]
        for site in structure.sites:
            x, y, z = site.position
            lines.append(f"  {site.kind_name}  {x:.10f}  {y:.10f}  {z:.10f}\n")
        return "".join(lines)

    @staticmethod
    def _render_cell_parameters(structure: StructureData) -> str:
        lines = ["CELL_PARAMETERS angstrom\n"]
        for vec in structure.cell:
            lines.append(f"  {vec[0]:.10f}  {vec[1]:.10f}  {vec[2]:.10f}\n")
        return "".join(lines)

    @staticmethod
    def _write_alpha_file(folder, alphas: list[float], filename: str) -> None:
        """Write screening parameters in kcp.x ``file_alpharef[_empty].txt`` format.

        Format: first line is the orbital count, subsequent lines are
        ``{index} {alpha} 1.0`` (1-indexed).
        """
        content = f"{len(alphas)}\n"
        content += "".join(f"{i + 1} {a} 1.0\n" for i, a in enumerate(alphas))
        with folder.open(filename, "w", encoding="utf-8") as handle:
            handle.write(content)
