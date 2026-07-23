"""Parser for a ``wan_mode='decompose'`` run of ``pw2wannier90.x``.

The run writes one ASCII file per Wannier function ``N`` (1-indexed,
zero-padded to five digits):

* ``<seed>_NNNNN.coeff`` -- the orbital-density expansion coefficients,
  ``n_max*(l_max+1)^2`` values ordered outer ``n`` (0..n_max-1), then ``l``
  (0..l_max), then inner ``m`` (0..2l).
* ``<seed>_NNNNN.power`` -- the orbital-only power spectrum,
  ``(l_max+1)*n_max*(n_max+1)/2`` values ordered outer ``n1``, then
  ``n2>=n1``, then ``l``; each entry is ``sum_m c(n1,l,m) c(n2,l,m)``.
* ``<seed>_gc_NNNNN.coeff`` -- (optional) the group-density coefficients
  about external centre ``N``, same layout as the orbital ``.coeff``.

Every file carries a ``#``-commented header giving ``n_max`` / ``l_max`` /
``r_min`` / ``r_max``. The parser stacks the per-WF vectors into
``ArrayData`` outputs (``coefficients`` / ``power`` / ``group_coefficients``)
so the power-spectrum descriptor can be assembled downstream by
:mod:`aiida_koopmans.ml_helpers`.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from aiida import orm

from aiida_koopmans.parsers.base import KoopmansStdoutParser, time_string_to_seconds

# ``<seed>_00001.coeff`` (orbital) vs ``<seed>_gc_00001.coeff`` (group). The
# orbital pattern deliberately excludes the ``gc`` marker.
_ORBITAL_COEFF_RE = re.compile(r"^(?P<seed>.+?)_(?P<index>\d{5})\.coeff$")
_GROUP_COEFF_RE = re.compile(r"^(?P<seed>.+?)_gc_(?P<index>\d{5})\.coeff$")
_POWER_RE = re.compile(r"^(?P<seed>.+?)_(?P<index>\d{5})\.power$")


class Pw2wannierDecomposeParser(KoopmansStdoutParser):
    """Parse the output of a :class:`Pw2wannierDecomposeCalculation`.

    Emits ``output_parameters`` (``job_done`` flag, optional ``walltime``,
    basis sizes), ``coefficients`` / ``power`` ``ArrayData`` (one row per
    Wannier function) and, when a ``centres_file`` was supplied,
    ``group_coefficients``. Returns ``ERROR_OUTPUT_STDOUT_INCOMPLETE`` when
    the run did not reach ``JOB DONE``, ``ERROR_OUTPUT_COEFF_MISSING`` when a
    completed run produced no coefficient files, and
    ``ERROR_OUTPUT_COEFF_MALFORMED`` when a retrieved file could not be
    parsed or the per-WF vectors disagree in length.
    """

    def parse(self, **kwargs: Any):
        """Entry point called by AiiDA after the CalcJob finishes."""
        stdout = self._read_stdout()
        if not isinstance(stdout, str):
            return stdout

        parsed = self._parse_stdout(stdout)

        if not parsed.get("job_done", False):
            self.out("output_parameters", orm.Dict(dict=parsed))
            return self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE

        try:
            arrays = self._collect_arrays(self.retrieved)
        except ValueError as exc:
            parsed["parse_error"] = str(exc)
            self.out("output_parameters", orm.Dict(dict=parsed))
            return self.exit_codes.ERROR_OUTPUT_COEFF_MALFORMED

        if arrays is None:
            self.out("output_parameters", orm.Dict(dict=parsed))
            return self.exit_codes.ERROR_OUTPUT_COEFF_MISSING

        coefficients, power, group_coefficients, meta = arrays

        coeff_node = orm.ArrayData()
        coeff_node.set_array("coefficients", coefficients)
        self.out("coefficients", coeff_node)

        power_node = orm.ArrayData()
        power_node.set_array("power", power)
        self.out("power", power_node)

        if group_coefficients is not None:
            group_node = orm.ArrayData()
            group_node.set_array("group_coefficients", group_coefficients)
            self.out("group_coefficients", group_node)

        parsed.update(meta)
        self.out("output_parameters", orm.Dict(dict=parsed))
        return None

    # ------------------------------------------------------------------
    # File collection
    # ------------------------------------------------------------------

    def _collect_arrays(self, retrieved: orm.FolderData):
        """Stack the retrieved coeff/power files into per-WF arrays.

        Returns ``(coefficients, power, group_coefficients, meta)`` where the
        first three are ``np.ndarray`` (``group_coefficients`` is ``None``
        when no group-density file was written) and ``meta`` holds the basis
        sizes. Returns ``None`` when no orbital coefficient file was found.
        Raises ``ValueError`` on a malformed file or inconsistent shapes.
        """
        names = retrieved.base.repository.list_object_names()

        orbital = self._indexed_files(names, _ORBITAL_COEFF_RE, exclude=_GROUP_COEFF_RE)
        group = self._indexed_files(names, _GROUP_COEFF_RE)
        power = self._indexed_files(names, _POWER_RE)

        if not orbital:
            return None

        coeff_rows, coeff_header = self._stack(retrieved, orbital)
        power_rows, _ = self._stack(retrieved, power)
        group_rows: np.ndarray | None = None
        if group:
            group_rows, _ = self._stack(retrieved, group)

        n_max = coeff_header.get("n_max")
        l_max = coeff_header.get("l_max")
        if n_max is not None and l_max is not None:
            expected_coeff = n_max * (l_max + 1) ** 2
            if coeff_rows.shape[1] != expected_coeff:
                raise ValueError(
                    f"coefficient vector length {coeff_rows.shape[1]} does not match "
                    f"n_max*(l_max+1)^2 = {expected_coeff} (n_max={n_max}, l_max={l_max})"
                )

        meta: dict[str, Any] = {
            "num_wann": int(coeff_rows.shape[0]),
            "n_coeff": int(coeff_rows.shape[1]),
            "n_power": int(power_rows.shape[1]) if power_rows.size else 0,
            "num_group_centres": int(group_rows.shape[0]) if group_rows is not None else 0,
        }
        if n_max is not None:
            meta["n_max"] = int(n_max)
        if l_max is not None:
            meta["l_max"] = int(l_max)
        for key in ("r_min", "r_max"):
            if key in coeff_header:
                meta[key] = float(coeff_header[key])

        return coeff_rows, power_rows, group_rows, meta

    @staticmethod
    def _indexed_files(
        names: list[str], pattern: re.Pattern, exclude: re.Pattern | None = None
    ) -> list[tuple[int, str]]:
        """Return ``(index, filename)`` for every ``names`` entry matching ``pattern``.

        ``exclude`` lets the orbital ``.coeff`` scan skip the ``_gc_`` group
        files, whose names also end in ``.coeff``. Sorted by the 1-indexed
        Wannier/centre number so the stacked rows keep WF order.
        """
        matched: list[tuple[int, str]] = []
        for name in names:
            if exclude is not None and exclude.match(name):
                continue
            match = pattern.match(name)
            if match:
                matched.append((int(match.group("index")), name))
        return sorted(matched)

    def _stack(
        self, retrieved: orm.FolderData, indexed: list[tuple[int, str]]
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Read and vertically stack a set of indexed value files.

        Returns the ``(n_files, n_values)`` array and the header of the first
        file. Raises ``ValueError`` if the files disagree in length.
        """
        rows: list[np.ndarray] = []
        header: dict[str, float] = {}
        for position, (_, name) in enumerate(indexed):
            content = retrieved.base.repository.get_object_content(name, mode="r")
            values, file_header = self._parse_value_file(content, name)
            if position == 0:
                header = file_header
            rows.append(values)
        widths = {row.size for row in rows}
        if len(widths) > 1:
            raise ValueError(
                f"inconsistent value-file lengths {sorted(widths)} across {len(rows)} files"
            )
        return np.array(rows, dtype=float) if rows else np.empty((0, 0)), header

    @staticmethod
    def _parse_value_file(content: str, name: str) -> tuple[np.ndarray, dict[str, float]]:
        """Parse one ``.coeff`` / ``.power`` file into (values, header).

        The header is the ``#``-commented lines of the form ``# key = value``;
        the body is one float per line. Raises ``ValueError`` when a body line
        is not a float.
        """
        header: dict[str, float] = {}
        values: list[float] = []
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                body = stripped.lstrip("#").strip()
                if "=" in body:
                    key, _, val = body.partition("=")
                    key = key.strip()
                    val = val.strip()
                    try:
                        header[key] = int(val) if key in ("n_max", "l_max") else float(val)
                    except ValueError:
                        pass
                continue
            try:
                values.append(float(stripped))
            except ValueError as exc:
                raise ValueError(f"non-numeric line in {name!r}: {stripped!r}") from exc
        return np.array(values, dtype=float), header

    @staticmethod
    def _parse_stdout(stdout: str) -> dict[str, Any]:
        """Extract ``job_done`` and ``walltime`` from the decompose stdout.

        The wall-time line starts with the (upper-cased) program name
        ``PW2WANNIER`` and carries the time token as its second-to-last field.
        """
        results = KoopmansStdoutParser._base_scalars(stdout)
        for line in stdout.splitlines():
            if line.strip().startswith("PW2WANNIER"):
                tokens = line.split()
                if len(tokens) >= 2:
                    try:
                        results["walltime"] = time_string_to_seconds(tokens[-2])
                    except ValueError:
                        pass
        return results
