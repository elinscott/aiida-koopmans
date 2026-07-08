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

from aiida.common import CalcInfo
from aiida.orm import ArrayData, Dict, RemoteData, StructureData
from aiida.plugins import DataFactory

from aiida_koopmans.calculations.base import KoopmansStdoutCalculation
from aiida_koopmans.utils import walk_remote_files

UpfData = DataFactory("pseudo.upf")


class KcpCalculation(KoopmansStdoutCalculation):
    """AiiDA plugin for running kcp.x, the Koopmans-modified CP code in Quantum ESPRESSO."""

    _TOOL_NAME = "kcp.x"

    _INPUT_FILE = "aiida.cpi"
    _OUTPUT_FILE = "aiida.cpo"
    _CRASH_FILE = "CRASH"
    _PREFIX = "aiida"
    _OUTPUT_SUBFOLDER = "out"
    _PSEUDO_SUBFOLDER = "pseudo"
    _K_SUBDIR = "K00001"
    # All koopmans kcp.x runs use the same ndr/ndw pair. AiiDA scratch
    # already isolates each calc, so per-step renumbering buys us nothing
    # — every calc reads from ``out/aiida_50.save/`` (symlinked from the
    # parent's writeout) and writes to ``out/aiida_60.save/``. Override
    # these in a subclass if you ever need a different scheme.
    _NDR = 50
    _NDW = 60

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
        spec.input_namespace(
            "alphas",
            required=False,
            help=(
                "Orbital-dependent screening parameters split into ``filled`` "
                "and ``empty`` sub-inputs. Each is an ``orm.Dict`` keyed by "
                "spin channel (``'none'`` for nspin=1, ``'up'`` / ``'down'`` "
                "for nspin=2) mapping to per-orbital alpha lists. Shape "
                "matches the :class:`~aiida_koopmans.types.AlphaScreening` "
                "TypedDict so a workgraph ``@task`` returning that TypedDict "
                "wires its namespace output straight through. Required when "
                "the parameters request orbital-dependent screening."
            ),
        )
        spec.input(
            "alphas.filled",
            valid_type=Dict,
            required=False,
            help="Per-spin filled-orbital alpha lists.",
        )
        spec.input(
            "alphas.empty",
            valid_type=Dict,
            required=False,
            help="Per-spin empty-orbital alpha lists.",
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
            "parent_folder_evcfixed",
            valid_type=RemoteData,
            required=False,
            help=(
                "Remote folder of a ``pz_print`` kcp.x run, for the empty-"
                "orbital branch of the Delta-SCF screening loop. Only "
                "``out/<prefix>_<NDW>.save/K00001/evcfixed_empty.dat`` is "
                "symlinked from this folder; the orbital save directory "
                "comes from ``parent_folder`` (the ``dft_n+1_dummy`` run)."
            ),
        )
        spec.input(
            "variational_orbital_overlays",
            valid_type=Dict,
            required=False,
            help=(
                "Mapping ``{source_stem: dest_stem}`` of K00001 "
                "wavefunction overlays to layer on top of ``parent_folder``'s "
                "save. Stems are ``.dat`` filenames *without* the extension "
                "(AiiDA's attribute store rejects keys containing ``.``); "
                "the CalcJob appends ``.dat`` at submission time. Source "
                "paths resolve against "
                "``parent_folder/out/<prefix>_<NDW>.save/K00001/``; "
                "destination paths land in "
                "``out/<prefix>_<NDR>.save/K00001/`` (this calc's read slot). "
                "Matching destinations are skipped during the primary "
                "parent walk so the overlay symlinks are the only entries "
                "at those paths. Used to force ``evc0N = evcN`` for "
                "``init_orbitals='kohn-sham'`` (legacy "
                "``_koopmans_dscf.py:521-522`` + variational-orbital map at "
                "``_koopmans_dscf.py:1340-1347``) — without this the kcp.x "
                "inner loop picks up a stale variational guess and converges "
                "to a different canonical KI basis."
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
        # Declare each calc as a single-rank job; callers wanting multi-rank
        # kcp.x runs override via ``metadata.options.resources``.
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
            nspin=nspin, do_orbdep=do_orbdep, do_bare_eigs=do_bare_eigs
        )

        calc_info = CalcInfo()
        calc_info.codes_info = [self._make_code_info()]
        calc_info.local_copy_list = local_copy_list
        calc_info.remote_symlink_list = remote_symlink_list
        calc_info.retrieve_list = retrieve_list
        calc_info.retrieve_temporary_list = retrieve_temporary_list

        return calc_info

    # ------------------------------------------------------------------
    # prepare_for_submission helpers
    # ------------------------------------------------------------------

    def _inject_owned_keys(self, parameters: dict, structure: StructureData) -> None:
        """Set the CONTROL/SYSTEM keys the CalcJob owns (outdir, pseudo_dir, ibrav, ...).

        Force the universal ``ndr`` / ``ndw`` so the symlink-rename in
        ``_build_remote_symlink_list`` always knows which save directory
        to map. Any caller-supplied ``ndr`` / ``ndw`` is overwritten on
        purpose — chaining is via ``parent_folder``, not by manual
        renumbering.
        """
        control = parameters.setdefault("CONTROL", {})
        control["outdir"] = f"./{self._OUTPUT_SUBFOLDER}/"
        control["pseudo_dir"] = f"./{self._PSEUDO_SUBFOLDER}/"
        control["prefix"] = self._PREFIX
        control.setdefault("calculation", "cp")
        control["ndr"] = self._NDR
        control["ndw"] = self._NDW

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
        from aiida_koopmans.types import SpinChannel

        alphas_requested = do_orbdep and odd_nkscalfact
        # ``alphas`` is an input namespace, so ``"alphas" in self.inputs`` is
        # True even when the caller never wires it — the empty namespace
        # always exists. Detect actual provision by checking whether either
        # leaf sub-port (``filled`` / ``empty``) is populated.
        alphas_ns = self.inputs.get("alphas", None)
        alphas_provided = alphas_ns is not None and ("filled" in alphas_ns or "empty" in alphas_ns)
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
        # ``alphas`` is a namespace with ``filled`` and ``empty`` Dict
        # sub-inputs (matching :class:`AlphaScreening`). Each Dict's payload
        # is keyed by spin channel.
        filled_per_spin: dict[str, list[float]] = self.inputs.alphas.filled.get_dict()
        empty_per_spin: dict[str, list[float]] = self.inputs.alphas.empty.get_dict()
        nspin = int(self.inputs.parameters.get_dict().get("SYSTEM", {}).get("nspin", 1))
        order = [SpinChannel.NONE] if nspin == 1 else [SpinChannel.UP, SpinChannel.DOWN]

        def _flatten(per_spin: dict) -> list[float]:
            # Closed-shell case: upstream packs the single representative
            # channel under ``SpinChannel.NONE`` (see
            # ``aiida_koopmans.workgraphs.kcp.build_filled_iter_source`` and
            # ``generate_alphas``). When the kcp.x run is ``nspin=2`` we
            # mirror that one list onto every spin slot before writing the
            # block-spin ``file_alpharef``.
            if (
                SpinChannel.NONE in per_spin
                and SpinChannel.NONE not in order
                and len(per_spin) == 1
            ):
                return list(per_spin[SpinChannel.NONE]) * len(order)
            return [a for spin in order for a in per_spin.get(spin, [])]

        filled_flat = _flatten(filled_per_spin)
        empty_flat = _flatten(empty_per_spin)
        self._write_alpha_file(folder, filled_flat, self._ALPHAREF_FILE)
        self._write_alpha_file(folder, empty_flat, self._ALPHAREF_EMPTY_FILE)

    def _build_remote_symlink_list(self) -> list[tuple[str, str, str]]:
        """Stage prior-run save directories under the child's read slot.

        Every ``KcpCalculation`` reads from ``out/<prefix>_50.save/`` and
        writes to ``out/<prefix>_60.save/`` (see ``_NDR`` / ``_NDW``). When
        chained, the child's ``aiida_50.save/`` is populated with one
        symlink per file from the primary ``parent_folder``'s freshly-
        written ``aiida_60.save/`` tree. Per-file (rather than directory-
        level) symlinks are essential for the overlay case below: layering
        a symlink on top of a directory symlink would resolve through to
        the parent's actual scratch and either fail with ``FileExistsError``
        or, worse, mutate the parent.

        For the ``dft_n+1`` step of the Delta-SCF empty-orbital branch we
        also need ``evcfixed_empty.dat`` from a separate ``pz_print`` run.
        Supply that run's ``RemoteData`` as ``parent_folder_evcfixed`` and
        we (a) skip the corresponding ``evc_occupied{1,2}.dat`` entries
        from the primary parent's tree and (b) emit overlay symlinks
        sourcing those two files from the evcfixed parent. AiiDA scratch
        already isolates each calc, so no naming collisions arise.

        Walking the parent's tree goes through the AiiDA transport
        (``RemoteData.get_authinfo().get_transport()``), so this is
        transport-agnostic. kcp ``.save`` directories are shallow
        (top-level files + a single ``K00001/`` subdir) so the recursion
        is bounded and the per-submission listdir cost is small.
        """
        symlinks: list[tuple[str, str, str]] = []
        target_save = f"{self._OUTPUT_SUBFOLDER}/{self._PREFIX}_{self._NDR}.save"
        parent_save_relpath = f"{self._OUTPUT_SUBFOLDER}/{self._PREFIX}_{self._NDW}.save"

        # When the evcfixed overlay is present, these two files come from a
        # *different* parent — exclude them from the primary parent's walk
        # so the overlay symlinks below are the only entries that land at
        # those paths.
        overlay_skip: set[str] = set()
        if "parent_folder_evcfixed" in self.inputs:
            overlay_skip |= {f"{self._K_SUBDIR}/evc_occupied{ispin}.dat" for ispin in (1, 2)}
        # ``variational_orbital_overlays`` also overrides entries from the
        # primary parent — different *source name* (e.g. ``evc1.dat`` instead
        # of ``evc01.dat``) at the *same destination*. Skip the destinations
        # during the walk so the explicit overlay below wins.
        overlays_map: dict[str, str] = (
            self.inputs.variational_orbital_overlays.get_dict()
            if "variational_orbital_overlays" in self.inputs
            else {}
        )
        overlay_skip |= {f"{self._K_SUBDIR}/{dest}.dat" for dest in overlays_map.values()}

        if "parent_folder" in self.inputs:
            parent = self.inputs.parent_folder
            parent_root = PurePosixPath(parent.get_remote_path())
            parent_save_abs = parent_root / parent_save_relpath
            for rel_file in walk_remote_files(parent, parent_save_relpath):
                if rel_file in overlay_skip:
                    continue
                abs_source = str(parent_save_abs / rel_file)
                rel_dest = f"{target_save}/{rel_file}"
                symlinks.append((parent.computer.uuid, abs_source, rel_dest))

        if "parent_folder_evcfixed" in self.inputs:
            # ``pz_print`` writes per-spin ``evcfixed_empty{ispin}.dat``;
            # the ``dft_n+1`` step reads them under the *renamed*
            # ``evc_occupied{ispin}.dat`` (kcp.x's
            # ``restart_from_wannier_pwscf`` machinery hard-codes that
            # filename). Two symlinks per call, one per spin. The KI-DSCF
            # flow is always nspin=2, so both source files are guaranteed
            # to exist on the pz_print parent.
            evc_parent = self.inputs.parent_folder_evcfixed
            evc_save = (
                PurePosixPath(evc_parent.get_remote_path())
                / self._OUTPUT_SUBFOLDER
                / f"{self._PREFIX}_{self._NDW}.save"
                / self._K_SUBDIR
            )
            for ispin in (1, 2):
                evc_source = evc_save / f"evcfixed_empty{ispin}.dat"
                evc_target = f"{target_save}/{self._K_SUBDIR}/evc_occupied{ispin}.dat"
                symlinks.append((evc_parent.computer.uuid, str(evc_source), evc_target))

        if overlays_map:
            # Per-spin variational/KS overlays for the trial-KI step:
            # ``parent_folder`` already supplies the canonical save dir,
            # but kcp.x's NKSIC inner loop reads its variational starting
            # guess from ``evc0{ispin}.dat`` / ``evc0_empty{ispin}.dat``.
            # For ``init_orbitals='kohn-sham'`` we want the variational
            # guess to *be* the canonical KS basis, so re-point those four
            # filenames at the corresponding ``evc{ispin}.dat`` /
            # ``evc_empty{ispin}.dat`` from the same parent. Without this,
            # the inner loop minimises from a stale guess and lands on a
            # rotated KI basis — same total energy, different canonical
            # eigenvalues, different per-orbital alphas.
            parent = self.inputs.parent_folder
            parent_save_abs = PurePosixPath(parent.get_remote_path()) / parent_save_relpath
            for source_stem, dest_stem in overlays_map.items():
                source_abs = str(parent_save_abs / self._K_SUBDIR / f"{source_stem}.dat")
                dest_rel = f"{target_save}/{self._K_SUBDIR}/{dest_stem}.dat"
                symlinks.append((parent.computer.uuid, source_abs, dest_rel))

        return symlinks

    def _build_retrieve_list(self) -> list[str]:
        """Files persisted in the ``retrieved`` FolderData: stdout, CRASH, user extras."""
        retrieve_list: list[str] = [self._OUTPUT_FILE, self._CRASH_FILE]
        retrieve_list.extend(self._additional_retrieve_list())
        return retrieve_list

    def _build_retrieve_temporary_list(
        self, *, nspin: int, do_orbdep: bool, do_bare_eigs: bool
    ) -> list:
        """Files retrieved into a scratch folder for parsing then discarded.

        Hamiltonian XMLs are intermediate artefacts; the parser turns them into
        ``ArrayData`` outputs, after which the raw XMLs serve no purpose.
        Tuple form ``(remote, '.', depth)`` preserves the
        ``out/<prefix>_<ndw>.save/K00001/`` nesting AiiDA would otherwise flatten.
        """
        if not do_orbdep:
            return []
        ham_dir = f"{self._OUTPUT_SUBFOLDER}/{self._PREFIX}_{self._NDW}.save/{self._K_SUBDIR}"
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
            out.append(cls.render_namelist(nl, options))
            rendered.add(nl)
        for nl, options in parameters.items():
            if nl in rendered or not options:
                continue
            out.append(cls.render_namelist(nl, options))
        return "".join(out)

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
