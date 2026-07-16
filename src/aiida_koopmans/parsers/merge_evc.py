"""Parser for Quantum ESPRESSO's ``merge_evc.x`` output.

``merge_evc.x`` produces essentially no structured stdout -- its sole product
is the merged ``evc`` file. The parser confirms that file landed in the
retrieved folder, re-emits it as the ``merged_file`` ``SinglefileData``
output (making the merged wavefunction a first-class node for the downstream
kcp.x staging), and records a minimal ``output_parameters`` Dict with a
``merged`` flag.
"""

from __future__ import annotations

from typing import Any

from aiida import orm
from aiida.parsers import Parser


class MergeEvcParser(Parser):
    """Parse the output of a ``MergeEvcCalculation``.

    Verifies that the merged ``dest_filename`` file is present in the retrieved
    folder and emits ``output_parameters`` with a ``merged`` flag. Returns
    ``ERROR_OUTPUT_FILE_MISSING`` when the file is absent.
    """

    def parse(self, **kwargs: Any):
        """Entry point called by AiiDA after the CalcJob finishes."""
        try:
            retrieved = self.retrieved
        except Exception:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        dest_filename = self.node.inputs.dest_filename.value
        names = retrieved.base.repository.list_object_names()
        merged = dest_filename in names

        self.out("output_parameters", orm.Dict(dict={"merged": merged}))

        if not merged:
            return self.exit_codes.ERROR_OUTPUT_FILE_MISSING

        with retrieved.base.repository.open(dest_filename, "rb") as handle:
            self.out("merged_file", orm.SinglefileData(handle, filename=dest_filename))

        return None
