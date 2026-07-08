"""CalcJobs for Quantum ESPRESSO's kcw.x (Koopmans-spectral Wannier code).

One kcw.x binary runs three distinct calculation modes selected via the
``CONTROL.calculation`` namelist flag:

* ``wann2kcw`` -- convert a prior pw.x (nscf) run plus Wannier90 matrices into
  the kcw.x internal format (:class:`Wann2kcCalculation`);
* ``screen`` -- compute the orbital-dependent screening parameters (alphas)
  with DFPT (:class:`KcwScreenCalculation`);
* ``ham`` -- build, interpolate, and diagonalize the Koopmans Hamiltonian
  (:class:`KcwHamCalculation`).

kcw.x has no upstream aiida-quantumespresso coverage, so these are standalone
CalcJobs (mirroring the conventions of the in-repo ``kcp.py`` plugin, which
see for the parent-scratch symlink idiom).

Chaining contract
-----------------

kcw.x reads *everything* about the electronic structure from a prior run's
scratch directory: each CalcJob requires a ``parent_folder`` whose ``out/``
tree is symlinked per-file into the calc's own ``out/``. For
``Wann2kcCalculation`` the parent is the pw.x **nscf** run (its
``out/aiida.save`` + ``out/aiida.xml``); for ``screen`` / ``ham`` the parent
is the **wann2kcw** run (whose ``out/`` additionally holds the ``kcw/``
conversion products). Per-file (not directory-level) symlinks mean any file
kcw.x writes lands in the child's own scratch instead of mutating the
parent's.

When Wannier functions are used (``kcw_at_ks=.false.``,
``read_unitary_matrix=.true.``), kcw.x additionally reads the Wannier90
products from its *working directory*: ``<seedname>_u.mat``,
``<seedname>_hr.dat``, ``<seedname>_centres.xyz`` for the occupied manifold
and ``<seedname>_emp_u.mat`` / ``_emp_u_dis.mat`` / ``_emp_hr.dat`` /
``_emp_centres.xyz`` for the empty one. Stage them via the ``wannier_files``
``FolderData`` input -- its contents are copied into the workdir root with
their stored names (the DFPT workgraph assembles that folder from the
wannier90 ``retrieved`` outputs; see
``aiida_koopmans.workgraphs.dfpt.prepare_kcw_wannier_files``).

Input namelists are validated against the ``pydantic_espresso`` kcw models
(develop version) so typos and off-spec values fail at submission rather
than producing a silently broken input file.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import ClassVar

from aiida import orm
from aiida.common import CalcInfo
from pydantic import ValidationError
from pydantic_espresso.models.kcw.develop import (
    ControlNamelist,
    HamNamelist,
    ScreenNamelist,
    WannierNamelist,
)

from aiida_koopmans.calculations.base import KoopmansStdoutCalculation
from aiida_koopmans.utils import walk_remote_files


class KcwCalculation(KoopmansStdoutCalculation):
    """Shared machinery for the three kcw.x calculation modes.

    Not registered as an entry point itself -- use one of the three
    subclasses, which pin ``CONTROL.calculation`` and the parser.
    """

    _TOOL_NAME = "kcw.x"

    _PREFIX = "aiida"
    _OUTPUT_SUBFOLDER = "out"

    # Subclasses pin these.
    _CALCULATION: ClassVar[str] = ""
    _INPUT_FILE: ClassVar[str] = "aiida.kcwi"
    _OUTPUT_FILE: ClassVar[str] = "aiida.kcwo"
    _DEFAULT_PARSER: ClassVar[str] = ""
    # The mode-specific namelist appended after CONTROL + WANNIER (or None).
    _MODE_NAMELIST: ClassVar[str | None] = None

    _NAMELIST_ORDER = ("CONTROL", "WANNIER", "SCREEN", "HAM")
    _NAMELIST_MODELS: ClassVar[dict] = {
        "CONTROL": ControlNamelist,
        "WANNIER": WannierNamelist,
        "SCREEN": ScreenNamelist,
        "HAM": HamNamelist,
    }

    # Keys the CalcJob owns; users cannot set them in ``parameters``.
    # ``prefix`` / ``outdir`` must match the parent scratch that is symlinked
    # in; ``calculation`` is what distinguishes the three subclasses.
    _BLOCKED_CONTROL_KEYS: ClassVar[frozenset[str]] = frozenset({"prefix", "outdir", "calculation"})

    @classmethod
    def define(cls, spec):
        """Declare the inputs, outputs, and exit codes shared by all kcw.x modes."""
        super().define(spec)

        spec.input(
            "parameters",
            valid_type=orm.Dict,
            help=(
                "Nested namelist dictionary, e.g. ``{'CONTROL': {...}, "
                "'WANNIER': {...}}`` plus the mode namelist (``SCREEN`` / "
                "``HAM``). Namelist names are case-insensitive; keys are "
                "validated against the pydantic_espresso kcw models. "
                "``CONTROL.prefix``, ``CONTROL.outdir`` and "
                "``CONTROL.calculation`` are owned by the CalcJob."
            ),
        )
        spec.input(
            "parent_folder",
            valid_type=orm.RemoteData,
            required=True,
            help=(
                "Remote folder of the upstream run (pw.x nscf for wann2kcw; "
                "the wann2kcw run for screen / ham). Its ``out/`` tree is "
                "symlinked per-file into this calc's ``out/``."
            ),
        )
        spec.input(
            "wannier_files",
            valid_type=orm.FolderData,
            required=False,
            help=(
                "Wannier90 products (``<seedname>[_emp]_u.mat``, ``_hr.dat``, "
                "``_centres.xyz``, ``_emp_u_dis.mat``) copied into the working "
                "directory root under their stored names. Required whenever "
                "``CONTROL.read_unitary_matrix`` is true."
            ),
        )
        spec.input(
            "settings",
            valid_type=orm.Dict,
            required=False,
            help="Optional CalcJob-level settings (extra retrieve paths).",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = cls._DEFAULT_PARSER
        spec.inputs["metadata"]["options"]["input_filename"].default = cls._INPUT_FILE
        spec.inputs["metadata"]["options"]["output_filename"].default = cls._OUTPUT_FILE
        spec.inputs["metadata"]["options"]["withmpi"].default = True
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1}

        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            required=True,
            help="Scalar results: ``job_done`` flag and ``walltime``.",
        )

    def prepare_for_submission(self, folder):
        """Render the input file and build the ``CalcInfo``."""
        parameters = self._normalize_parameters(self.inputs.parameters.get_dict())
        self._inject_owned_keys(parameters)
        self._validate_parameters(parameters)

        content = self._render_namelists(parameters) + self._render_extra_cards(parameters)
        with folder.open(self._INPUT_FILE, "w", encoding="utf-8") as handle:
            handle.write(content)

        self._write_extra_input_files(folder, parameters)

        calc_info = CalcInfo()
        calc_info.codes_info = [self._make_code_info()]
        calc_info.local_copy_list = self._build_local_copy_list()
        calc_info.remote_symlink_list = self._build_remote_symlink_list()
        calc_info.retrieve_list = self._build_retrieve_list(parameters)

        return calc_info

    # ------------------------------------------------------------------
    # prepare_for_submission helpers
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_parameters(cls, parameters: dict) -> dict:
        """Uppercase namelist names, lowercase keys within, and reject blocked keys."""
        allowed = {"CONTROL", "WANNIER"}
        if cls._MODE_NAMELIST is not None:
            allowed.add(cls._MODE_NAMELIST)

        normalized: dict[str, dict] = {}
        for namelist, options in parameters.items():
            nl = namelist.upper()
            if nl not in allowed:
                raise ValueError(
                    f"Namelist ``{nl}`` is not valid for a "
                    f"``calculation='{cls._CALCULATION}'`` kcw.x run. Valid namelists: "
                    f"{', '.join(sorted(allowed))}."
                )
            if not isinstance(options, dict):
                raise ValueError(
                    f"Namelist ``{namelist}`` must map to a dict, got {type(options).__name__}."
                )
            row: dict = {}
            for key, val in options.items():
                k = key.lower()
                if nl == "CONTROL" and k in cls._BLOCKED_CONTROL_KEYS:
                    raise ValueError(
                        f"Parameter ``CONTROL/{k}`` is set by the CalcJob and cannot be overridden."
                    )
                row[k] = val
            normalized[nl] = row
        return normalized

    @classmethod
    def _inject_owned_keys(cls, parameters: dict) -> None:
        """Set the CONTROL keys the CalcJob owns.

        ``prefix`` / ``outdir`` must match the aiida-quantumespresso pw.x
        conventions because kcw.x reads the parent scratch tree that a
        ``PwCalculation`` wrote (``out/aiida.save``).
        """
        control = parameters.setdefault("CONTROL", {})
        control["prefix"] = cls._PREFIX
        control["outdir"] = f"./{cls._OUTPUT_SUBFOLDER}/"
        control["calculation"] = cls._CALCULATION

    @classmethod
    def _validate_parameters(cls, parameters: dict) -> None:
        """Validate every namelist against its pydantic_espresso model.

        The models carry ``extra='forbid'`` plus per-field type/literal
        constraints, so unknown keys and off-spec values raise here instead
        of producing an input file kcw.x rejects (or worse, silently
        misreads).
        """
        for nl, options in parameters.items():
            model = cls._NAMELIST_MODELS[nl]
            try:
                model(**options)
            except ValidationError as exc:
                raise ValueError(f"Invalid ``{nl}`` namelist for kcw.x: {exc}") from exc

    @classmethod
    def _render_namelists(cls, parameters: dict) -> str:
        """Render namelists in canonical kcw.x order (CONTROL, WANNIER, mode)."""
        out: list[str] = []
        for nl in cls._NAMELIST_ORDER:
            options = parameters.get(nl)
            if not options:
                continue
            out.append(cls.render_namelist(nl, options))
        return "".join(out)

    def _render_extra_cards(self, parameters: dict) -> str:
        """Render mode-specific cards after the namelists (ham's K_POINTS)."""
        return ""

    def _write_extra_input_files(self, folder, parameters: dict) -> None:
        """Write mode-specific side files (ham's ``file_alpharef.txt``)."""

    def _build_local_copy_list(self) -> list[tuple[str, str, str]]:
        """Copy every ``wannier_files`` object into the workdir root as-is."""
        if "wannier_files" not in self.inputs:
            return []
        wannier_files = self.inputs.wannier_files
        return [
            (wannier_files.uuid, name, name)
            for name in wannier_files.base.repository.list_object_names()
        ]

    def _build_remote_symlink_list(self) -> list[tuple[str, str, str]]:
        """Symlink the parent's ``out/`` tree per-file into the child's ``out/``.

        Per-file symlinks (rather than one directory-level symlink) keep any
        file kcw.x writes -- the ``kcw/`` conversion products of wann2kcw, the
        screen response files -- inside the child's own scratch instead of
        mutating the parent's. The walk goes through the AiiDA transport, so
        it is transport-agnostic; the parent tree may itself already consist
        of symlinks (a screen/ham parent is a wann2kcw run whose ``out/``
        links back to the nscf scratch), which ``transport.isdir`` follows.
        """
        parent = self.inputs.parent_folder
        parent_root = PurePosixPath(parent.get_remote_path())
        parent_out = parent_root / self._OUTPUT_SUBFOLDER
        symlinks: list[tuple[str, str, str]] = []
        for rel_file in walk_remote_files(parent, self._OUTPUT_SUBFOLDER):
            symlinks.append(
                (
                    parent.computer.uuid,
                    str(parent_out / rel_file),
                    f"{self._OUTPUT_SUBFOLDER}/{rel_file}",
                )
            )
        return symlinks

    def _build_retrieve_list(self, parameters: dict) -> list[str]:
        """Retrieve stdout plus any user extras."""
        retrieve_list: list[str] = [self._OUTPUT_FILE]
        retrieve_list.extend(self._additional_retrieve_list())
        return retrieve_list


class Wann2kcCalculation(KcwCalculation):
    """kcw.x in ``wann2kcw`` mode: convert pw.x + Wannier90 outputs to kcw format.

    ``parent_folder`` must be the **nscf** pw.x remote folder. The run writes
    its conversion products under ``out/kcw/``, which downstream screen / ham
    calcs pick up by chaining on *this* calc's ``remote_folder``.
    """

    _CALCULATION = "wann2kcw"
    _INPUT_FILE = "aiida.w2ki"
    _OUTPUT_FILE = "aiida.w2ko"
    _DEFAULT_PARSER = "koopmans.kcw_wann2kc"
    _MODE_NAMELIST = None


class KcwScreenCalculation(KcwCalculation):
    """kcw.x in ``screen`` mode: DFPT calculation of the screening parameters.

    ``parent_folder`` must be the ``Wann2kcCalculation`` remote folder. The
    parser emits the per-orbital alphas as an ``orm.List`` output.
    """

    _CALCULATION = "screen"
    _INPUT_FILE = "aiida.ksi"
    _OUTPUT_FILE = "aiida.kso"
    _DEFAULT_PARSER = "koopmans.kcw_screen"
    _MODE_NAMELIST = "SCREEN"

    @classmethod
    def define(cls, spec):
        """Add the screen-specific ``alphas`` output."""
        super().define(spec)
        spec.output(
            "alphas",
            valid_type=orm.List,
            required=True,
            help=(
                "Per-orbital screening parameters (the ``alpha`` column of the "
                "``iwann ... relaxed ... alpha ...`` stdout lines), in orbital "
                "order for this run's ``spin_component``."
            ),
        )
        spec.exit_code(
            320,
            "ERROR_OUTPUT_ALPHAS_MISSING",
            message="No screening parameters found in the kcw.x screen output.",
        )


class KcwHamCalculation(KcwCalculation):
    """kcw.x in ``ham`` mode: construct and interpolate the Koopmans Hamiltonian.

    ``parent_folder`` must be the ``Wann2kcCalculation`` remote folder. The
    screening parameters computed by ``KcwScreenCalculation`` (or provided as
    a guess) are passed via the ``alphas`` input and written to
    ``file_alpharef.txt``. When ``HAM.do_bands`` is true a ``kpoints`` input
    holding the explicit band path is required and rendered as a
    ``K_POINTS crystal_b`` card (all points explicit, zero intermediate
    points).
    """

    _CALCULATION = "ham"
    _INPUT_FILE = "aiida.khi"
    _OUTPUT_FILE = "aiida.kho"
    _DEFAULT_PARSER = "koopmans.kcw_ham"
    _MODE_NAMELIST = "HAM"

    @classmethod
    def define(cls, spec):
        """Add the ham-specific ``alphas`` / ``kpoints`` inputs and ``bands`` output."""
        super().define(spec)
        spec.input(
            "alphas",
            valid_type=orm.List,
            required=True,
            help=(
                "Screening parameters for *all* orbitals (occupied then "
                "empty), written to ``file_alpharef.txt``. kcw.x ham takes a "
                "single file rather than a filled/empty split."
            ),
        )
        spec.input(
            "kpoints",
            valid_type=orm.KpointsData,
            required=False,
            help=(
                "Explicit k-point path for the band interpolation. Required "
                "when ``HAM.do_bands`` is true."
            ),
        )
        spec.output(
            "bands",
            valid_type=orm.BandsData,
            required=False,
            help="Interpolated Koopmans eigenvalues along the input k-path (eV).",
        )
        spec.exit_code(
            320,
            "ERROR_OUTPUT_BANDS_MISSING",
            message="``do_bands`` was requested but no interpolated eigenvalues were found.",
        )

    def _render_extra_cards(self, parameters: dict) -> str:
        """Render the ``K_POINTS crystal_b`` card when ``do_bands`` is on."""
        if not parameters.get("HAM", {}).get("do_bands", False):
            return ""
        if "kpoints" not in self.inputs:
            raise ValueError("``HAM.do_bands`` is true but no ``kpoints`` input was provided.")
        kpoints = self.inputs.kpoints.get_kpoints()
        lines = ["K_POINTS crystal_b\n", f"{len(kpoints)}\n"]
        for kpt in kpoints:
            lines.append(f"{kpt[0]:.14f} {kpt[1]:.14f} {kpt[2]:.14f} 0\n")
        return "".join(lines)

    def _write_extra_input_files(self, folder, parameters: dict) -> None:
        """Write ``file_alpharef.txt`` from the ``alphas`` input.

        Every alpha is written to the "filled" file with an empty companion
        file (kcw.x ham takes a single alpha file, not a filled/empty split).
        """
        self._write_alpha_file(folder, self.inputs.alphas.get_list(), self._ALPHAREF_FILE)
        self._write_alpha_file(folder, [], self._ALPHAREF_EMPTY_FILE)

    def _build_retrieve_list(self, parameters: dict) -> list[str]:
        """Also retrieve the real-space Koopmans Hamiltonians when written.

        ``write_hr=.true.`` makes kcw.x drop ``<prefix>.kcw_hr_occ.dat`` /
        ``<prefix>.kcw_hr_emp.dat`` in the working directory; the unfold-and-
        interpolate postprocessing consumes them. AiiDA silently skips
        retrieve entries the run did not produce (e.g. no ``_emp`` file when
        there is no empty manifold).
        """
        retrieve_list = super()._build_retrieve_list(parameters)
        if parameters.get("HAM", {}).get("write_hr", False):
            retrieve_list += [
                f"{self._PREFIX}.kcw_hr_occ.dat",
                f"{self._PREFIX}.kcw_hr_emp.dat",
            ]
        return retrieve_list
