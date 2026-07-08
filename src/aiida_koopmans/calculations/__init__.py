"""CalcJob plugins for aiida-koopmans.

Only add a new CalcJob here when no upstream AiiDA plugin
(aiida-quantumespresso, aiida-wannier90) covers the binary.
"""

from aiida_koopmans.calculations.kcp import KcpCalculation

__all__ = ("KcpCalculation",)
