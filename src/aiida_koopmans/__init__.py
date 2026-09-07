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
    import time.

    Where the patch is in effect is not something this module can promise:
    a daemon worker imports it only once something pulls in an entry point
    it owns, and koopmans has seen the same calculation stored with 14
    ranks and with 1 in one run. Treat it as a floor for a rank count
    nothing else declares, not as a guarantee — a CalcJob whose count
    matters names it in ``metadata.options.resources``.

    Track upstream: https://github.com/aiidateam/aiida-hyperqueue/issues
    """
    try:
        from aiida_hyperqueue.scheduler import HyperQueueJobResource
    except ImportError:  # plugin not installed in this env
        return
    HyperQueueJobResource.accepts_default_mpiprocs_per_machine = classmethod(lambda cls: True)


_patch_hyperqueue_accepts_computer_default()
