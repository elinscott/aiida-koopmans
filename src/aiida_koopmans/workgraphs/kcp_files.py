"""Naming for the files kcp.x retrieves under ``write_hr``."""

from aiida_koopmans.spin import SpinChannel

#: Glob patterns naming the Koopmans Hamiltonians kcp.x prints under
#: ``write_hr``, one occupied and one empty file per spin channel, in the
#: working directory (QE ``CPV/write_hamiltonian.f90``).
KCP_HAMILTONIAN_PATTERNS = ("ham_occ_*.dat", "ham_emp_*.dat")


def kcp_hamiltonian_filename(*, filled: bool, spin: SpinChannel) -> str:
    """Name the Koopmans Hamiltonian kcp.x prints for one manifold.

    kcp.x indexes its printed files by its own 1-based spin index: 1 for
    up, and for the single channel of an unpolarized run; 2 for down.

    Raises:
        ValueError: ``spin`` is the spinor channel, which kcp.x has no
            mode for.
    """
    channel = SpinChannel(spin)
    if channel == SpinChannel.SPINOR:
        raise ValueError(
            "kcp.x has no noncollinear mode and prints no Hamiltonian for a spinor "
            "manifold; run the unpolarized or the collinear ('up'/'down') route."
        )
    spin_index = 2 if channel == SpinChannel.DOWN else 1
    return f"ham_{'occ' if filled else 'emp'}_{spin_index}.dat"
