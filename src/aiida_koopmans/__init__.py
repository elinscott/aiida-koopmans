"""AiiDA plugin for running Koopmans functional calculations."""

__version__ = "0.1.0a0"


def _patch_hyperqueue_accepts_computer_default() -> None:
    """Make ``aiida-hyperqueue`` honour the Computer's mpiprocs default.

    ``aiida-hyperqueue`` 0.3.x ships ``HyperQueueJobResource`` with
    ``accepts_default_mpiprocs_per_machine() == False`` and a backward-
    compat path that hard-codes ``num_mpiprocs_per_machine=1`` when only
    ``num_machines`` is supplied (see ``aiida_hyperqueue/scheduler.py``
    lines 99-102 and 72-73). The combination silently drops the
    ``Computer.default_mpiprocs_per_machine`` koopmans sets during
    ``koopmans install`` — every CalcJob then runs with ``mpirun -np 1``
    regardless of the user's ``--procs-per-calc`` setting.

    Until upstream fixes that, flip the classmethod to ``True`` once at
    import time. This module is imported by the AiiDA daemon worker via
    plugin entry points, so the patch is in effect everywhere we submit.

    Track upstream: https://github.com/aiidateam/aiida-hyperqueue/issues
    """
    try:
        from aiida_hyperqueue.scheduler import HyperQueueJobResource
    except ImportError:  # plugin not installed in this env
        return
    HyperQueueJobResource.accepts_default_mpiprocs_per_machine = classmethod(lambda cls: True)


_patch_hyperqueue_accepts_computer_default()
