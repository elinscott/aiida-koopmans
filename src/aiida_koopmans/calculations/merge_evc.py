"""CalcJob for Quantum ESPRESSO's ``merge_evc.x`` (Koopmans fork).

``merge_evc.x`` concatenates the per-block ``evcw`` wavefunction files produced
by :class:`~aiida_koopmans.calculations.wann2kcp.Wann2kcpCalculation` into a
single supercell ``evc`` file. Unlike ``kcp.x`` / ``wann2kcp.x`` it takes no
Fortran namelist -- it is a pure command-line tool. Mirroring the legacy
``MergeEVCProcess`` (``koopmans/processes/merge_evc.py``), the command is::

    merge_evc.x -nr <prod(kgrid)> -i input_0.dat -i input_1.dat ... -o <dest>

where ``-nr`` is the total number of real-space grid points (the product of
the k-grid). There is no upstream ``aiida-quantumespresso`` equivalent, so this
is a standalone ``CalcJob``.

The source ``evc`` files arrive as a dynamic ``source_files`` namespace of
``RemoteData`` nodes. Each is symlinked into the work directory as
``input_{i}.dat`` in the namespace's sorted-key order, matching the legacy
``input_{i}.dat`` naming. The merged file is written to ``dest_filename`` and
retrieved into the ``retrieved`` folder (it also remains on
``remote_folder``).
"""

from __future__ import annotations

import math

from aiida.common import CalcInfo, CodeInfo
from aiida.engine import CalcJob
from aiida.orm import Dict, List, RemoteData, Str


class MergeEvcCalculation(CalcJob):
    """AiiDA plugin for running ``merge_evc.x`` from the Koopmans Quantum ESPRESSO fork."""

    _OUTPUT_FILE = "aiida.out"

    @classmethod
    def define(cls, spec):
        """Declare the inputs, outputs, and exit codes for the CalcJob."""
        super().define(spec)

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
            valid_type=RemoteData,
            dynamic=True,
            help=(
                "Remote folders holding the source ``evc`` files to merge. Each is "
                "symlinked into the work dir as ``input_{i}.dat`` in sorted-key "
                "order and passed as a ``-i`` argument. The file inside each "
                "RemoteData is selected via ``settings['source_filenames']`` "
                "(defaults to the dest filename if unset)."
            ),
        )
        spec.input(
            "settings",
            valid_type=Dict,
            required=False,
            help=(
                "Optional CalcJob-level settings. ``source_filenames`` maps each "
                "``source_files`` namespace key to the file name to pick out of "
                "that RemoteData; ``additional_retrieve_list`` adds extra "
                "retrieve paths."
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

        spec.exit_code(301, "ERROR_NO_RETRIEVED_FOLDER", message="The retrieved folder is missing.")
        spec.exit_code(
            302,
            "ERROR_OUTPUT_FILE_MISSING",
            message="The merged evc output file was not retrieved.",
        )

    def prepare_for_submission(self, folder):
        """Build the ``merge_evc.x`` command line and the ``CalcInfo``."""
        kgrid = self.inputs.kgrid.get_list()
        dest_filename = self.inputs.dest_filename.value

        source_keys = sorted(self.inputs.source_files.keys())
        source_filenames = (
            self.inputs.settings.get_dict().get("source_filenames", {})
            if "settings" in self.inputs
            else {}
        )

        remote_symlink_list: list[tuple[str, str, str]] = []
        input_names: list[str] = []
        for i, key in enumerate(source_keys):
            remote = self.inputs.source_files[key]
            src_name = source_filenames.get(key, dest_filename)
            source_path = f"{remote.get_remote_path()}/{src_name}"
            dest_name = f"input_{i}.dat"
            remote_symlink_list.append((remote.computer.uuid, source_path, dest_name))
            input_names.append(dest_name)

        cmdline_params = self._build_cmdline(kgrid, input_names, dest_filename)

        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        code_info.cmdline_params = cmdline_params
        code_info.stdout_name = self._OUTPUT_FILE

        calc_info = CalcInfo()
        calc_info.codes_info = [code_info]
        calc_info.remote_symlink_list = remote_symlink_list
        calc_info.retrieve_list = self._build_retrieve_list(dest_filename)

        return calc_info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cmdline(kgrid: list[int], input_names: list[str], dest_filename: str) -> list[str]:
        """Assemble the ``merge_evc.x`` argument list.

        Produces ``-nr <prod(kgrid)> -i input_0.dat ... -o <dest>`` -- the same
        shape as the legacy ``MergeEVCProcess.command``. ``-nr`` is the product
        of the k-grid (``math.prod`` avoids a numpy dependency here).
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
        if "settings" in self.inputs:
            extra = self.inputs.settings.get_dict().get("additional_retrieve_list", [])
            retrieve_list.extend(extra)
        return retrieve_list
