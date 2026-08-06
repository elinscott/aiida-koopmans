"""CalcJob for a ``wan_mode='decompose'`` run of ``pw2wannier90.x``.

This is a *second* ``pw2wannier90.x`` pass, run after a full wannierization,
that decomposes each Wannier-function density onto an orthonormalized
Gaussian-radial x real-spherical-harmonic basis about its own centre --
entirely in reciprocal space (Quantum ESPRESSO ``wann-decompose`` branch;
``PP/src/pw2wannier90_decompose.f90``). It replaces the legacy kcp.x
real-space orbital-density postprocessing as the source of the
``power_spectrum`` descriptor.

Upstream ``aiida-quantumespresso`` provides a ``Pw2wannier90Calculation``,
but it cannot stage the wannier90 read-back files this mode requires
(the ``<seed>.nnkp`` post-processing file, ``<seed>_u.mat``, the optional
``<seed>_u_dis.mat`` and ``<seed>_centres.xyz``), so this is a standalone
``CalcJob``.

Inputs staged into the work directory:

* ``parent_folder`` -- the pw.x nscf scratch (a ``RemoteData``), symlinked
  as ``./TMP/<prefix>.save`` exactly like every QE post-processing parent.
* ``nnkp`` -- the wannier90 ``<seed>.nnkp`` post-processing file
  (``SinglefileData``), copied in as ``<seedname>.nnkp``. The decompose
  pass opens it (``read_nnkp``) before any density work, so it is required;
  it also fixes the band count (``num_bands``) and excluded-band mask the
  decomposition applies.
* ``u_mat`` / ``u_dis_mat`` / ``centres_xyz`` -- the enumerated wannier90
  products (``SinglefileData``), copied in as ``<seedname>_u.mat`` /
  ``<seedname>_u_dis.mat`` / ``<seedname>_centres.xyz``. ``u_dis_mat`` is
  required whenever the manifold disentangles (``num_bands`` > ``num_wann``).
* ``centres_file`` -- optional extra centres for the group-density channel
  (``SinglefileData``, one Cartesian-Angstrom triple per line). When given,
  the run additionally decomposes the group density (sum of the normalized
  Wannier densities) about each listed centre into ``<seed>_gc_NNNNN.coeff``.
  The Koopmans cross-power descriptor passes every Wannier centre here so
  the group density is sampled about each orbital's own centre.

The run writes, per Wannier function ``N`` (1-indexed, zero-padded to five
digits): ``<seed>_NNNNN.coeff`` (``n_max*(l_max+1)^2`` values) and
``<seed>_NNNNN.power`` (the orbital-only power spectrum,
``(l_max+1)*n_max*(n_max+1)/2`` values), plus ``<seed>_gc_NNNNN.coeff`` per
entry of ``centres_file``. The parser stacks these into ``ArrayData``
outputs.
"""

from __future__ import annotations

from typing import Any, ClassVar

from aiida.common import CalcInfo
from aiida.orm import ArrayData, Dict, RemoteData, SinglefileData

from aiida_koopmans.calculations.base import KoopmansStdoutCalculation
from aiida_koopmans.ml import DECOMPOSE_KEY_PREFIX, RADIAL_BASIS_DEFAULTS


class Pw2wannierDecomposeCalculation(KoopmansStdoutCalculation):
    """AiiDA plugin for ``pw2wannier90.x`` with ``wan_mode='decompose'``."""

    _TOOL_NAME = "pw2wannier90.x"

    _INPUT_FILE = "aiida.decompose.in"
    _OUTPUT_FILE = "aiida.decompose.out"
    _DEFAULT_OUTDIR = "TMP"
    # ``prefix`` / ``seedname`` must match the upstream pw.x nscf and
    # wannier90 runs. aiida-quantumespresso's ``PwCalculation`` hard-codes
    # ``_PREFIX = "aiida"`` and the wannier90 workflow uses seedname
    # ``aiida``, so both default to ``aiida``.
    _DEFAULT_PREFIX = "aiida"
    _DEFAULT_SEEDNAME = "aiida"
    # The group-density external-centres file staged from ``centres_file``.
    _CENTRES_FILE = "gc_centres.dat"

    _NAMELIST = "INPUTPP"

    # Keys the CalcJob owns; users cannot set them in ``parameters``.
    # ``wan_mode`` is fixed to ``decompose`` (that is the whole point of this
    # plugin), ``seedname`` names the staged wannier90 products, and
    # ``decompose_centres_file`` is driven by the ``centres_file`` input.
    _BLOCKED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"outdir", "prefix", "seedname", "wan_mode", "decompose_centres_file"}
    )

    # The full set of valid keys. Unknown keys are rejected so a typo does
    # not silently produce a broken input. ``spin_component`` is caller-set
    # (not owned): a decompose pass over an nspin=2 scratch must read one
    # channel at a time, so the graph passes ``'up'`` / ``'down'`` per block;
    # on an nspin=1 scratch it is omitted and QE defaults to the single
    # channel.
    _VALID_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "outdir",
            "prefix",
            "seedname",
            "wan_mode",
            "spin_component",
            "decompose_centres_file",
            "decompose_n_max",
            "decompose_l_max",
            "decompose_r_min",
            "decompose_r_max",
        }
    )

    # ``spin_component`` selects which half of an nspin=2 k list the pass
    # reads. QE compares the string exactly and falls through to the
    # unpolarized branch on anything it does not recognise, so a value
    # outside this set would decompose the wrong states without any
    # complaint from the binary.
    _VALID_SPIN_COMPONENTS: ClassVar[frozenset[str]] = frozenset({"up", "down", "none"})

    # Radial-basis defaults matching the legacy koopmans ``ml`` settings
    # (``n_max=4, l_max=4, r_min=0.5, r_max=4.0``); the QE binary itself
    # defaults ``n_max=l_max=6`` but the Koopmans descriptor is defined
    # against the legacy values, so they are the injected defaults here.
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        f"{DECOMPOSE_KEY_PREFIX}{key}": value for key, value in RADIAL_BASIS_DEFAULTS.items()
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
                "Flat ``&inputpp`` namelist dictionary of ``decompose_*`` "
                "overrides, e.g. ``{'decompose_n_max': 4, 'decompose_l_max': 4, "
                "'decompose_r_min': 0.5, 'decompose_r_max': 4.0}``. Keys are "
                "case-insensitive. ``outdir``, ``prefix``, ``seedname``, "
                "``wan_mode`` and ``decompose_centres_file`` are owned by the "
                "CalcJob; ``spin_component`` (``'up'`` / ``'down'``) is caller-set "
                "and required per channel on an nspin=2 scratch. Defaults: "
                "n_max=4, l_max=4, r_min=0.5, r_max=4.0."
            ),
        )
        spec.input(
            "parent_folder",
            valid_type=RemoteData,
            required=True,
            help=(
                "Remote folder of the upstream pw.x nscf run. Its ``.save`` tree "
                "is recursively symlinked into ``./TMP/<prefix>.save`` so the "
                "decompose pass can read the Bloch wavefunctions."
            ),
        )
        spec.input(
            "nnkp",
            valid_type=SinglefileData,
            required=True,
            help=(
                "The wannier90 ``<seed>.nnkp`` post-processing file. Copied in as "
                "``<seedname>.nnkp``. The decompose pass reads it (``read_nnkp``) "
                "before any density work, so it is mandatory; it also supplies the "
                "band count and excluded-band mask the decomposition applies."
            ),
        )
        spec.input(
            "u_mat",
            valid_type=SinglefileData,
            required=True,
            help=(
                "The wannier90 gauge matrix (``write_u_matrices=.true.``). "
                "Copied into the work directory as ``<seedname>_u.mat``."
            ),
        )
        spec.input(
            "u_dis_mat",
            valid_type=SinglefileData,
            required=False,
            help=(
                "The wannier90 disentanglement matrix. Required whenever the "
                "manifold disentangles (``num_bands`` > ``num_wann``); the QE "
                "decompose pass errors without it in that case. Copied in as "
                "``<seedname>_u_dis.mat``."
            ),
        )
        spec.input(
            "centres_xyz",
            valid_type=SinglefileData,
            required=True,
            help=(
                "The wannier90 Wannier-centre file (``write_xyz=.true.``). "
                "Copied into the work directory as ``<seedname>_centres.xyz``."
            ),
        )
        spec.input(
            "centres_file",
            valid_type=SinglefileData,
            required=False,
            help=(
                "Optional external centres for the group-density channel (one "
                "Cartesian-Angstrom triple per line, ``#`` comments allowed). "
                "Copied in as ``gc_centres.dat``; when present the run writes "
                "``<seed>_gc_NNNNN.coeff`` per listed centre."
            ),
        )
        spec.input(
            "settings",
            valid_type=Dict,
            required=False,
            help="Optional CalcJob-level settings (extra retrieve paths).",
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "koopmans.pw2wannier_decompose"
        spec.inputs["metadata"]["options"]["input_filename"].default = cls._INPUT_FILE
        spec.inputs["metadata"]["options"]["output_filename"].default = cls._OUTPUT_FILE
        spec.inputs["metadata"]["options"]["withmpi"].default = True

        spec.output(
            "output_parameters",
            valid_type=Dict,
            required=True,
            help=(
                "Scalar results: ``job_done`` flag, ``walltime``, and the basis "
                "sizes (``n_max``, ``l_max``, ``n_coeff``, ``n_power``, "
                "``num_wann``, ``num_group_centres``)."
            ),
        )
        spec.output(
            "coefficients",
            valid_type=ArrayData,
            required=True,
            help=(
                "Per-Wannier-function orbital-density expansion coefficients, "
                "an ``ArrayData`` with array ``coefficients`` of shape "
                "``(num_wann, n_coeff)`` (row ``i`` is WF ``i+1``)."
            ),
        )
        spec.output(
            "power",
            valid_type=ArrayData,
            required=True,
            help=(
                "Per-Wannier-function orbital-only power spectrum as written by "
                "the QE binary, an ``ArrayData`` with array ``power`` of shape "
                "``(num_wann, n_power)``."
            ),
        )
        spec.output(
            "group_coefficients",
            valid_type=ArrayData,
            required=False,
            help=(
                "Group-density expansion coefficients about each external "
                "centre, an ``ArrayData`` with array ``group_coefficients`` of "
                "shape ``(num_group_centres, n_coeff)``. Present only when a "
                "``centres_file`` was supplied."
            ),
        )

        spec.exit_code(
            311,
            "ERROR_CODE_LACKS_DECOMPOSE",
            message=(
                "pw2wannier90.x aborted reading a ``decompose_*`` key of the "
                "``&inputpp`` namelist. Register a pw2wannier90.x built from the "
                "``wann-decompose`` branch of Quantum ESPRESSO as the code for this "
                "calculation; if it already is one, check the ``decompose_*`` values "
                "in ``parameters`` against the retrieved stdout."
            ),
            invalidates_cache=True,
        )
        spec.exit_code(
            330,
            "ERROR_OUTPUT_COEFF_MISSING",
            message=(
                "A completed decompose run retrieved no "
                "``<seedname>_NNNNN.coeff`` files. Read the retrieved stdout for "
                "what the decomposition reported."
            ),
            invalidates_cache=True,
        )
        spec.exit_code(
            331,
            "ERROR_OUTPUT_COEFF_MALFORMED",
            message="A retrieved coefficient/power file could not be parsed.",
            invalidates_cache=True,
        )

    def prepare_for_submission(self, folder):
        """Render the ``&inputpp`` input file and build the ``CalcInfo``."""
        raw = self.inputs.parameters.get_dict() if "parameters" in self.inputs else {}
        parameters = self._normalize_parameters(raw)
        self._inject_owned_keys(parameters)
        self._reject_pool_parallelism()

        content = self._render_namelist(parameters)
        with folder.open(self._INPUT_FILE, "w", encoding="utf-8") as handle:
            handle.write(content)

        # ``TMP`` is a real per-calculation directory into which only the
        # parent ``.save`` tree is symlinked (see ``_build_remote_symlink_list``).
        folder.get_subfolder(self._DEFAULT_OUTDIR, create=True)

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
        """Inject the CalcJob-owned keys and fill in the radial-basis defaults.

        ``wan_mode`` is fixed to ``decompose``; ``outdir`` / ``prefix`` /
        ``seedname`` name the staged scratch and wannier90 products; and
        ``decompose_centres_file`` is set to the staged ``gc_centres.dat``
        only when a ``centres_file`` input was provided.
        """
        parameters["outdir"] = f"./{self._DEFAULT_OUTDIR}/"
        parameters["prefix"] = self._DEFAULT_PREFIX
        parameters["seedname"] = self._DEFAULT_SEEDNAME
        parameters["wan_mode"] = "decompose"
        if "centres_file" in self.inputs:
            parameters["decompose_centres_file"] = self._CENTRES_FILE
        for key, default in self._DEFAULTS.items():
            parameters.setdefault(key, default)

    def _reject_pool_parallelism(self) -> None:
        """Reject a ``-npool`` greater than one in ``settings.cmdline``.

        ``wan_mode='decompose'`` aborts with ``pool parallelism not
        implemented``: the pass reconstructs each Wannier density from the
        whole k list at once, which a pool-distributed k list cannot serve.
        """
        cmdline = self._cmdline_from_settings()
        for position, flag in enumerate(cmdline):
            if flag != "-npool":
                continue
            value = cmdline[position + 1] if position + 1 < len(cmdline) else ""
            if value.isdigit() and int(value) == 1:
                continue
            raise ValueError(
                f"``-npool {value}`` was requested, but a wan_mode='decompose' pass "
                "runs on a single k-point pool. Drop ``npool`` for pw2wannier90, or "
                "set it to 1; use ``ntasks`` to parallelize the pass instead."
            )

    def _build_remote_symlink_list(self) -> list[tuple[str, str, str]]:
        """Symlink the parent nscf ``.save`` into ``./TMP/<prefix>.save``.

        The decompose pass reads the Bloch wavefunctions from
        ``<outdir>/<prefix>.save``; the parent is an aiida-quantumespresso
        pw.x run whose scratch lives under ``<workdir>/out/`` (the
        ``PwCalculation._OUTPUT_SUBFOLDER``). Only the ``.save`` tree is
        symlinked, matching the wann2kcp plugin.
        """
        parent = self.inputs.parent_folder
        prefix = self._DEFAULT_PREFIX
        source = f"{parent.get_remote_path()}/out/{prefix}.save"
        return [(parent.computer.uuid, source, f"{self._DEFAULT_OUTDIR}/{prefix}.save")]

    def _build_local_copy_list(self, parameters: dict) -> list[tuple[str, str, str]]:
        """Copy the wannier90 read-back files (and optional gc centres) into place.

        The wannier90 products are enumerated ``SinglefileData`` inputs whose
        provenance lives on the per-block wannierization; destination names
        follow the ``seedname`` the namelist declares. ``u_dis_mat`` is
        optional (only disentangling blocks produce it).
        """
        seedname = parameters.get("seedname", self._DEFAULT_SEEDNAME)
        destinations = {
            "nnkp": f"{seedname}.nnkp",
            "u_mat": f"{seedname}_u.mat",
            "u_dis_mat": f"{seedname}_u_dis.mat",
            "centres_xyz": f"{seedname}_centres.xyz",
            "centres_file": self._CENTRES_FILE,
        }
        copy_list: list[tuple[str, str, str]] = []
        for input_name, destination in destinations.items():
            if input_name in self.inputs:
                node = self.inputs[input_name]
                copy_list.append((node.uuid, node.filename, destination))
        return copy_list

    def _build_retrieve_list(self, parameters: dict) -> list:
        """Retrieve stdout plus every ``*.coeff`` / ``*.power`` file.

        The per-WF count is not known at submission time, so the coefficient
        and power files are retrieved by glob (``<seedname>_*.coeff`` also
        matches the ``<seedname>_gc_*.coeff`` group-density files).
        """
        seedname = parameters.get("seedname", self._DEFAULT_SEEDNAME)
        retrieve_list: list = [
            self._OUTPUT_FILE,
            [f"{seedname}_*.coeff", ".", 0],
            [f"{seedname}_*.power", ".", 0],
        ]
        retrieve_list.extend(self._additional_retrieve_list())
        return retrieve_list

    # ------------------------------------------------------------------
    # Input-rendering helpers
    # ------------------------------------------------------------------

    @classmethod
    def _normalize_parameters(cls, parameters: dict) -> dict:
        """Lowercase keys, reject blocked and unknown keys."""
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
                    f"Unknown pw2wannier90 decompose parameter ``{k}``. Valid keys: "
                    f"{', '.join(sorted(cls._VALID_KEYS - cls._BLOCKED_KEYS))}."
                )
            normalized[k] = val
        cls._validate_spin_component(normalized)
        return normalized

    @classmethod
    def _validate_spin_component(cls, parameters: dict) -> None:
        """Reject a ``spin_component`` value QE would not recognise."""
        if "spin_component" not in parameters:
            return
        value = parameters["spin_component"]
        if not isinstance(value, str) or value not in cls._VALID_SPIN_COMPONENTS:
            raise ValueError(
                f"``spin_component`` must be one of "
                f"{', '.join(sorted(cls._VALID_SPIN_COMPONENTS))}, got {value!r}. "
                "Use ``up`` or ``down`` to decompose one channel of an nspin=2 "
                "scratch, ``none`` (or omit the key) for an nspin=1 one."
            )

    @classmethod
    def _render_namelist(cls, parameters: dict) -> str:
        """Render the single ``&inputpp`` namelist for the input file."""
        return cls.render_namelist(cls._NAMELIST, parameters)
