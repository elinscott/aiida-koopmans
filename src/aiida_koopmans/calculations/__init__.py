"""CalcJob plugins for aiida-koopmans.

Only add a new CalcJob here when no upstream AiiDA plugin
(aiida-quantumespresso, aiida-wannier90) covers the binary. CalcJobs are
registered via the ``aiida.calculations`` entry points in ``pyproject.toml``;
import them from their own modules.
"""
