"""Import-closure guard for the vocabulary modules the koopmans schema imports."""

import os
import subprocess
import sys

# Every module the koopmans input-file schema (or another config-less
# context, e.g. a docs build) imports from this package. Their import
# closure must not reach ``aiida_workgraph`` or ``aiida_pythonjob``:
# ``aiida_pythonjob`` loads the AiiDA configuration at import time and
# raises on a machine that has none.
VOCABULARY_MODULES = (
    "aiida_koopmans.functionals",
    "aiida_koopmans.ml",
    "aiida_koopmans.occupations",
    "aiida_koopmans.parallelization",
    "aiida_koopmans.projections",
    "aiida_koopmans.screening",
    "aiida_koopmans.spin",
    "aiida_koopmans.variational_orbitals",
)


def test_vocabulary_imports_without_aiida_configuration(tmp_path):
    """Import every vocabulary module in a subprocess that has no AiiDA config.

    ``AIIDA_PATH`` points at an empty directory, so any import in the
    closure that touches the AiiDA configuration fails the subprocess;
    the sys.modules assertion additionally rejects a workgraph or
    pythonjob import that happens to survive.
    """
    code = "\n".join(
        [
            "import sys",
            f"for name in {VOCABULARY_MODULES!r}:",
            "    __import__(name)",
            "bad = sorted(m for m in sys.modules if 'workgraph' in m or 'pythonjob' in m)",
            "assert not bad, f'vocabulary import closure pulls {bad}'",
        ]
    )
    env = dict(os.environ, AIIDA_PATH=str(tmp_path))
    result = subprocess.run(  # noqa: S603 -- fixed argv: this interpreter + a literal
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
