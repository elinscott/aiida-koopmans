"""Shared types for the koopmans AiiDA workgraph layer.

Cross-module data shapes live here so workgraph builders, the kcp.x
CalcJob, parser, and tests can all import a single canonical definition.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, TypedDict, get_args


class MLMode(str, Enum):
    """What the trajectory workflow does with the machine-learning dataset.

    ``NONE`` just runs the snapshots; ``TRAIN`` fits a screening model on
    the computed alphas; ``TEST`` scores a previously trained model
    against them; ``PREDICT`` applies a previously trained model in place
    of the per-orbital Delta-SCF refinement.
    """

    NONE = "none"
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"


class MLDescriptor(str, Enum):
    """Descriptor feeding the machine-learned screening model.

    Both descriptors are derived from the variational orbitals' densities:
    ``SELF_HARTREE`` is the scalar self-Hartree energy kcp.x reports, and
    ``POWER_SPECTRUM`` is the rotationally-invariant power spectrum of a
    pw2wannier90.x ``wan_mode='decompose'`` expansion of each density.
    """

    SELF_HARTREE = "self_hartree"
    POWER_SPECTRUM = "power_spectrum"


# The QE code vocabulary, defined once. ``CODE_NAMES`` is the runtime tuple
# (the koopmans2 parallelization schema imports it for its ``ALL_CODES``); the
# ``CodeName`` ``Literal`` types dict keys / helper args so a typo is a static
# error, and ``validate_parallelization`` catches one that slips in at runtime.
CodeName = Literal["pw", "kcp", "kcw", "ph", "projwfc", "pw2wannier90", "wann2kcp", "wannier90"]
CODE_NAMES: tuple[str, ...] = get_args(CodeName)


class CodeParallelization(TypedDict, total=False):
    """One code's parallelization directive: MPI ranks, k-point pools, pencil decomp, threads.

    ``ntasks`` sets ``metadata.options.resources`` (``num_mpiprocs_per_machine``);
    ``npool`` becomes ``-npool`` and ``pd`` becomes ``-pd true`` on the QE
    command line; ``omp`` sets the per-rank OpenMP/BLAS thread count via a
    ``metadata.options.prepend_text`` export block (overriding the
    computer-level pin of one thread). Every field is optional
    (``total=False``); an absent one means the QE/AiiDA default. Mirrors the
    koopmans2 ``CodeParallelization`` pydantic model that produces these dicts.
    """

    ntasks: int
    npool: int
    pd: bool
    omp: int


# Per-code parallelization mapping threaded into every top-level graph: a plain
# dict keyed by code name, each value a :class:`CodeParallelization`. A dict
# alias (not a fixed-key TypedDict) so ``aiida-workgraph`` keeps it as one
# opaque input socket rather than expanding a typed namespace, and so a dynamic
# ``code`` lookup types cleanly. Which flags each code accepts is enforced by
# ``POOL_SUPPORTING_CODES`` / ``PD_SUPPORTING_CODES`` in ``aiida_koopmans.workgraphs``.
ParallelizationDict = dict[CodeName, CodeParallelization]
