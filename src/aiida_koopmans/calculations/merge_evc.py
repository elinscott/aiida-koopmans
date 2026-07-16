"""CalcJob for Quantum ESPRESSO's ``merge_evc.x`` (Koopmans fork).

``merge_evc.x`` concatenates the per-block ``evcw`` wavefunction files produced
by :class:`~aiida_koopmans.calculations.wann2kcp.Wann2kcpCalculation` into a
single supercell ``evc`` file. Unlike ``kcp.x`` / ``wann2kcp.x`` it takes no
Fortran namelist -- it is a pure command-line tool. The command is::

    merge_evc.x -nr <prod(kgrid)> -i input_0.dat -i input_1.dat ... -o <dest>

where ``-nr`` is the total number of real-space grid points (the product of
the k-grid). There is no upstream ``aiida-quantumespresso`` equivalent, so this
is a standalone ``CalcJob``.

The source ``evc`` files arrive as a dynamic ``source_files`` namespace of
``SinglefileData`` nodes — the enumerated ``evcw`` outputs of the upstream
``wann2kcp.x`` runs. Each is copied into the work directory as
``input_{i}.dat`` in the namespace's sorted-key order. The merged file is
written to ``dest_filename``,
retrieved, and re-emitted by the parser as the ``merged_file``
``SinglefileData`` output so the whole fold pipeline is explicit dataflow.
"""

from __future__ import annotations

import math

from aiida.common import CalcInfo
from aiida.orm import Dict, List, SinglefileData, Str

from aiida_koopmans.calculations.base import KoopmansCalculation


class MergeEvcCalculation(KoopmansCalculation):
    """AiiDA plugin for running ``merge_evc.x`` from the Koopmans Quantum ESPRESSO fork."""

    _OUTPUT_FILE = "aiida.out"

    @classmethod
    def _validate_serial_resources(cls, value, port_namespace):
        """Reject parallel resources at submission time (serial concatenation tool)."""
        try:
            resources = value["metadata"]["options"]["resources"]
        except (KeyError, TypeError):
            return None
        nprocs = resources.get("tot_num_mpiprocs") or (
            resources.get("num_machines", 1) * resources.get("num_mpiprocs_per_machine", 1)
        )
        if nprocs > 1:
            return "merge_evc.x is a serial tool; run it on a single MPI rank."
        return None

    @classmethod
    def define(cls, spec):
        """Declare the inputs, outputs, and exit codes for the CalcJob."""
        super().define(spec)
        spec.inputs.validator = cls._validate_serial_resources

        spec.input(
            "kgrid",
            valid_type=List,
            help=(
                "The Monkhorst-Pack k-grid as a 3-element list, e.g. ``[2, 2, 2]``. "
                "Its product is passed as ``-nr`` (the number of real-space grid "
                "points merge_evc.x folds over)."
            ),
        )
        spec.input(
            "dest_filename",
            valid_type=Str,
            help="Name of the merged output file, e.g. ``evcw.dat`` (passed as ``-o``).",
        )
        spec.input_namespace(
            "source_files",
            valid_type=SinglefileData,
            dynamic=True,
            help=(
                "The source ``evc`` wavefunctions to merge (e.g. the ``evcw`` "
                "outputs of the upstream wann2kcp.x runs). Each is copied into "
                "the work dir as ``input_{i}.dat`` in sorted-key order and "
                "passed as a ``-i`` argument — key the namespace so sorting "
                "reproduces the intended band order."
            ),
        )
        spec.input(
            "settings",
            valid_type=Dict,
            required=False,
            help=(
                "Optional CalcJob-level settings; ``additional_retrieve_list`` "
                "adds extra retrieve paths."
            ),
        )

        spec.inputs["metadata"]["options"]["parser_name"].default = "koopmans.merge_evc"
        spec.inputs["metadata"]["options"]["output_filename"].default = cls._OUTPUT_FILE
        # merge_evc.x is a serial concatenation tool; never launched under MPI.
        spec.inputs["metadata"]["options"]["withmpi"].default = False
        spec.inputs["metadata"]["options"]["resources"].default = {"num_machines": 1}

        spec.output(
            "output_parameters",
            valid_type=Dict,
            required=True,
            help="Scalar results: a ``merged`` flag confirming the output file was written.",
        )
        spec.output(
            "merged_file",
            valid_type=SinglefileData,
            required=True,
            help="The merged ``evc`` wavefunction, named ``dest_filename``.",
        )

        spec.exit_code(
            302,
            "ERROR_OUTPUT_FILE_MISSING",
            message="The merged evc output file was not retrieved.",
            invalidates_cache=True,
        )

    def prepare_for_submission(self, folder):
        """Build the ``merge_evc.x`` command line and the ``CalcInfo``."""
        kgrid = self.inputs.kgrid.get_list()
        dest_filename = self.inputs.dest_filename.value

        local_copy_list: list[tuple[str, str, str]] = []
        input_names: list[str] = []
        for i, key in enumerate(sorted(self.inputs.source_files.keys())):
            source = self.inputs.source_files[key]
            dest_name = f"input_{i}.dat"
            local_copy_list.append((source.uuid, source.filename, dest_name))
            input_names.append(dest_name)

        cmdline_params = self._build_cmdline(kgrid, input_names, dest_filename)

        calc_info = CalcInfo()
        calc_info.codes_info = [self._make_code_info(cmdline_params)]
        calc_info.local_copy_list = local_copy_list
        calc_info.retrieve_list = self._build_retrieve_list(dest_filename)

        return calc_info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cmdline(kgrid: list[int], input_names: list[str], dest_filename: str) -> list[str]:
        """Assemble the ``merge_evc.x`` argument list.

        Produces ``-nr <prod(kgrid)> -i input_0.dat ... -o <dest>``. ``-nr``
        is the product of the k-grid (``math.prod`` avoids a numpy dependency
        here).
        """
        nr = math.prod(kgrid)
        params: list[str] = ["-nr", str(nr)]
        for name in input_names:
            params += ["-i", name]
        params += ["-o", dest_filename]
        return params

    def _build_retrieve_list(self, dest_filename: str) -> list[str]:
        """Retrieve the merged output file plus stdout and any user extras."""
        retrieve_list: list[str] = [dest_filename, self._OUTPUT_FILE]
        retrieve_list.extend(self._additional_retrieve_list())
        return retrieve_list
