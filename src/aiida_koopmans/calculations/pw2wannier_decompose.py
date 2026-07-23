"""CalcJob for a ``wan_mode='decompose'`` run of ``pw2wannier90.x``.

This is a *second* ``pw2wannier90.x`` pass, run after a full wannierization,
that decomposes each Wannier-function density onto an orthonormalized
Gaussian-radial x real-spherical-harmonic basis about its own centre --
entirely in reciprocal space (Quantum ESPRESSO ``wann-decompose`` branch;
``PP/src/pw2wannier90_decompose.f90``). It replaces the legacy kcp.x
real-space orbital-density postprocessing as the source of the
``orbital_density`` power-spectrum descriptor.

Upstream ``aiida-quantumespresso`` provides a ``Pw2wannier90Calculation``,
but it cannot stage the wannier90 read-back files this mode requires
(``<seed>_u.mat``, the optional ``<seed>_u_dis.mat`` and
``<seed>_centres.xyz``), so this is a standalone ``CalcJob``.

Inputs staged into the work directory:

* ``parent_folder`` -- the pw.x nscf scratch (a ``RemoteData``), symlinked
  as ``./TMP/<prefix>.save`` exactly like every QE post-processing parent.
* ``u_mat`` / ``u_dis_mat`` / ``centres_xyz`` -- the enumerated wannier90
  products (``SinglefileData``), copied in as ``<seedname>_u.mat`` /
  ``<seedname>_u_dis.mat`` / ``<seedname>_centres.xyz``.
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

from typing import ClassVar

from aiida.common import CalcInfo
from aiida.orm import ArrayData, Dict, RemoteData, SinglefileData

from aiida_koopmans.calculations.base import KoopmansStdoutCalculation


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
    # not silently produce a broken input.
    _VALID_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "outdir",
            "prefix",
            "seedname",
            "wan_mode",
            "decompose_centres_file",
            "decompose_n_max",
            "decompose_l_max",
            "decompose_r_min",
            "decompose_r_max",
        }
    )

    # Radial-basis defaults matching the legacy koopmans ``ml`` settings
    # (``n_max=4, l_max=4, r_min=0.5, r_max=4.0``); the QE binary itself
    # defaults ``n_max=l_max=6`` but the Koopmans descriptor is defined
    # against the legacy values, so they are the injected defaults here.
    _DEFAULTS: ClassVar[dict[str, float | int]] = {
        "decompose_n_max": 4,
        "decompose_l_max": 4,
        "decompose_r_min": 0.5,
        "decompose_r_max": 4.0,
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
                "CalcJob. Defaults: n_max=4, l_max=4, r_min=0.5, r_max=4.0."
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
                "The wannier90 disentanglement matrix, only present when the "
                "block disentangles. Copied in as ``<seedname>_u_dis.mat``."
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
            330,
            "ERROR_OUTPUT_COEFF_MISSING",
            message="A completed decompose run retrieved no ``*.coeff`` files.",
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
        return normalized

    @classmethod
    def _render_namelist(cls, parameters: dict) -> str:
        """Render the single ``&inputpp`` namelist for the input file."""
        return cls.render_namelist(cls._NAMELIST, parameters)
