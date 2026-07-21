"""Tests for the ``Pw2wannierDecomposeParser``.

The pure file-parsing helpers are exercised against synthetic ``.coeff`` /
``.power`` files whose byte layout mirrors the QE ``wann-decompose`` writer
(``PP/src/pw2wannier90_decompose.f90``: a ``#``-commented header followed by
one value per line), and ``_collect_arrays`` against a ``FolderData`` staged
with the same. ``TestRealSiFixtures`` additionally runs against real files
from a live Si run and checks the internal-consistency relation (our orb-orb
power block equals the binary's own ``.power`` file).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aiida_koopmans import ml_helpers
from aiida_koopmans.parsers.pw2wannier_decompose import (
    _GROUP_COEFF_RE,
    _ORBITAL_COEFF_RE,
    _POWER_RE,
    Pw2wannierDecomposeParser,
)

# Real ``.coeff`` / ``.power`` files from a live Si wan_mode='decompose' run
# (pw.x -> nscf -> wannier90 -> decompose pw2wannier90.x, n_max=l_max=4).
_SI_FIXTURES = Path(__file__).parent / "data" / "decompose_si"


def _coeff_file(values, n_max=4, l_max=4, r_min=0.5, r_max=4.0):
    """Render a synthetic ``.coeff`` / ``.power`` file the way QE writes it."""
    lines = [
        "# Wannier density decomposition coefficients",
        f"# n_max = {n_max}",
        f"# l_max = {l_max}",
        f"# r_min = {r_min:.15E}",
        f"# r_max = {r_max:.15E}",
        "# r_cut = none (reciprocal-space projection, untruncated basis)",
        "# ordering: outer n (0..n_max-1), then l (0..l_max), then inner m (0..2l)",
        f"# number of coefficients = {len(values)}",
    ]
    lines += [f"{v:.15E}" for v in values]
    return "\n".join(lines) + "\n"


class TestParseValueFile:
    """``_parse_value_file`` header + body extraction."""

    def test_reads_values_and_header(self):
        content = _coeff_file([1.0, 2.0, -3.5], n_max=4, l_max=4)
        values, header = Pw2wannierDecomposeParser._parse_value_file(content, "aiida_00001.coeff")
        assert np.allclose(values, [1.0, 2.0, -3.5])
        assert header["n_max"] == 4
        assert header["l_max"] == 4
        assert header["r_min"] == pytest.approx(0.5)
        assert header["r_max"] == pytest.approx(4.0)

    def test_rejects_non_numeric_body(self):
        content = "# n_max = 4\n1.0\nnot_a_number\n"
        with pytest.raises(ValueError, match="non-numeric line"):
            Pw2wannierDecomposeParser._parse_value_file(content, "bad.coeff")


class TestFilenamePatterns:
    """Orbital / group / power filename discrimination."""

    def test_orbital_excludes_group(self):
        assert _ORBITAL_COEFF_RE.match("aiida_00001.coeff")
        # A group file also ends in .coeff, so the orbital scan must exclude it.
        assert _GROUP_COEFF_RE.match("aiida_gc_00001.coeff")
        assert _ORBITAL_COEFF_RE.match("aiida_gc_00001.coeff")  # matches shape...
        # ...which is why _indexed_files passes exclude=_GROUP_COEFF_RE.

    def test_indexed_files_sorts_and_excludes(self):
        names = [
            "aiida_00002.coeff",
            "aiida_00001.coeff",
            "aiida_gc_00001.coeff",
            "aiida.decompose.out",
        ]
        orbital = Pw2wannierDecomposeParser._indexed_files(
            names, _ORBITAL_COEFF_RE, exclude=_GROUP_COEFF_RE
        )
        assert orbital == [(1, "aiida_00001.coeff"), (2, "aiida_00002.coeff")]
        group = Pw2wannierDecomposeParser._indexed_files(names, _GROUP_COEFF_RE)
        assert group == [(1, "aiida_gc_00001.coeff")]
        power = Pw2wannierDecomposeParser._indexed_files(names, _POWER_RE)
        assert power == []


class TestParseStdout:
    """``_parse_stdout`` job-done + walltime extraction."""

    def test_detects_job_done(self):
        assert Pw2wannierDecomposeParser._parse_stdout("...\n JOB DONE.\n")["job_done"] is True
        assert Pw2wannierDecomposeParser._parse_stdout("crashed early")["job_done"] is False

    def test_extracts_walltime(self):
        stdout = "JOB DONE.\nPW2WANNIER    :      0.12s CPU      0.15s WALL\n"
        parsed = Pw2wannierDecomposeParser._parse_stdout(stdout)
        assert parsed["walltime"] == pytest.approx(0.15)


class TestCollectArrays:
    """``_collect_arrays`` stacks the retrieved files into per-WF arrays."""

    @staticmethod
    def _folder(files: dict[str, str]):
        """Build a stored ``FolderData`` holding the given ``{name: content}``."""
        import io

        from aiida import orm

        folder = orm.FolderData()
        for name, content in files.items():
            folder.base.repository.put_object_from_filelike(io.BytesIO(content.encode()), name)
        return folder

    def test_stacks_orbital_group_and_power(self, aiida_profile):
        n_max, l_max = 2, 1
        n_coeff = n_max * (l_max + 1) ** 2  # 8
        rng = np.random.default_rng(1)
        orb = rng.standard_normal((3, n_coeff))
        grp = rng.standard_normal((3, n_coeff))

        files = {"aiida.decompose.out": "JOB DONE.\n"}
        for i in range(3):
            files[f"aiida_{i + 1:05d}.coeff"] = _coeff_file(orb[i], n_max, l_max)
            files[f"aiida_gc_{i + 1:05d}.coeff"] = _coeff_file(grp[i], n_max, l_max)
            # A stand-in power file (its exact values are irrelevant to shape).
            files[f"aiida_{i + 1:05d}.power"] = _coeff_file(
                np.zeros((l_max + 1) * n_max * (n_max + 1) // 2), n_max, l_max
            )

        parser = Pw2wannierDecomposeParser.__new__(Pw2wannierDecomposeParser)
        coeff, power, group, meta = parser._collect_arrays(self._folder(files))

        assert coeff.shape == (3, n_coeff)
        assert np.allclose(coeff, orb)
        assert group.shape == (3, n_coeff)
        assert np.allclose(group, grp)
        assert power.shape == (3, (l_max + 1) * n_max * (n_max + 1) // 2)
        assert meta["num_wann"] == 3
        assert meta["n_coeff"] == n_coeff
        assert meta["num_group_centres"] == 3
        assert meta["n_max"] == n_max
        assert meta["l_max"] == l_max

    def test_no_orbital_files_returns_none(self, aiida_profile):
        parser = Pw2wannierDecomposeParser.__new__(Pw2wannierDecomposeParser)
        folder = self._folder({"aiida.decompose.out": "JOB DONE.\n"})
        assert parser._collect_arrays(folder) is None

    def test_absent_group_channel(self, aiida_profile):
        n_max, l_max = 2, 1
        n_coeff = n_max * (l_max + 1) ** 2
        files = {"aiida_00001.coeff": _coeff_file(np.arange(n_coeff), n_max, l_max)}
        parser = Pw2wannierDecomposeParser.__new__(Pw2wannierDecomposeParser)
        coeff, _power, group, meta = parser._collect_arrays(self._folder(files))
        assert coeff.shape == (1, n_coeff)
        assert group is None
        assert meta["num_group_centres"] == 0

    def test_coeff_length_mismatch_raises(self, aiida_profile):
        # Header says n_max=4,l_max=4 (expect 100), but body has 8 values.
        files = {"aiida_00001.coeff": _coeff_file(np.zeros(8), n_max=4, l_max=4)}
        parser = Pw2wannierDecomposeParser.__new__(Pw2wannierDecomposeParser)
        with pytest.raises(ValueError, match="does not match"):
            parser._collect_arrays(self._folder(files))


class TestRealSiFixtures:
    """Regression against real files from a live Si decompose run (n_max=l_max=4)."""

    def test_parses_real_coeff_shapes(self):
        values, header = Pw2wannierDecomposeParser._parse_value_file(
            (_SI_FIXTURES / "si_00001.coeff").read_text(), "si_00001.coeff"
        )
        assert header["n_max"] == 4
        assert header["l_max"] == 4
        assert values.shape == (4 * (4 + 1) ** 2,)  # 100

    def test_orbital_power_matches_qe_power_file(self):
        """Internal consistency: our orb-orb block == the binary's own ``.power``.

        The single strongest correctness check — it confirms both the
        coefficient ordering and the power formula against the real QE
        ``wann-decompose`` output, not a transcription.
        """
        n_max, l_max = 4, 4
        for i in (1, 2):
            coeff, _ = Pw2wannierDecomposeParser._parse_value_file(
                (_SI_FIXTURES / f"si_{i:05d}.coeff").read_text(), f"si_{i:05d}.coeff"
            )
            qe_power, _ = Pw2wannierDecomposeParser._parse_value_file(
                (_SI_FIXTURES / f"si_{i:05d}.power").read_text(), f"si_{i:05d}.power"
            )
            mine = ml_helpers.orbital_power_from_coefficients(coeff, n_max, l_max)
            assert (
                mine.shape
                == qe_power.shape
                == (ml_helpers.orbital_power_block_length(n_max, l_max),)
            )
            assert np.allclose(mine, qe_power, rtol=1e-10, atol=1e-12)

    def test_cross_power_orb_block_equals_qe_power(self):
        """The full descriptor's orb-orb block reproduces the QE ``.power`` file."""
        n_max, l_max = 4, 4
        coeff, _ = Pw2wannierDecomposeParser._parse_value_file(
            (_SI_FIXTURES / "si_00001.coeff").read_text(), "si_00001.coeff"
        )
        group, _ = Pw2wannierDecomposeParser._parse_value_file(
            (_SI_FIXTURES / "si_gc_00001.coeff").read_text(), "si_gc_00001.coeff"
        )
        qe_power, _ = Pw2wannierDecomposeParser._parse_value_file(
            (_SI_FIXTURES / "si_00001.power").read_text(), "si_00001.power"
        )
        power = ml_helpers.cross_power_spectra(coeff[None, :], group[None, :], n_max, l_max)
        block = ml_helpers.orbital_power_block_length(n_max, l_max)
        assert power.shape == (1, 3 * block)
        assert np.allclose(power[0, :block], qe_power, rtol=1e-10, atol=1e-12)
