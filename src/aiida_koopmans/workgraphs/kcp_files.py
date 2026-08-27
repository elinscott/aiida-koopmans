"""Naming for the files kcp.x retrieves under ``write_hr``.

Split out of :mod:`aiida_koopmans.workgraphs.kcp` so
:mod:`aiida_koopmans.workgraphs.ui.dscf` can import it without a circular
import back into ``kcp``.
"""

#: Glob patterns naming the Koopmans Hamiltonians kcp.x prints under
#: ``write_hr``, one occupied and one empty file per spin channel, in the
#: working directory (QE ``CPV/write_hamiltonian.f90``).
KCP_HAMILTONIAN_PATTERNS = ("ham_occ_*.dat", "ham_emp_*.dat")


def kcp_hamiltonian_filename(*, filled: bool, spin_index: int) -> str:
    """Name the Koopmans Hamiltonian kcp.x prints for one manifold.

    ``spin_index`` is kcp.x's 1-based spin index: 1 for up (and for the
    single channel of an unpolarized run), 2 for down.
    """
    if spin_index not in (1, 2):
        raise ValueError(f"spin_index must be 1 or 2, got {spin_index!r}")
    return f"ham_{'occ' if filled else 'emp'}_{spin_index}.dat"
