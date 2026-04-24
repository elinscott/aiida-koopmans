"""CalcJob plugins for aiida-koopmans.

Only add a new CalcJob here when no upstream AiiDA plugin (aiida-quantumespresso,
aiida-wannier90) covers the binary. First port target: ``KcpCalculation`` for
kcp.x.
"""

from aiida_koopmans.calculations.kcp import KcpCalculation

__all__ = ("KcpCalculation",)
