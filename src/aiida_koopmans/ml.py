"""Machine-learning vocabulary for the screening-parameter workflows."""

from __future__ import annotations

from enum import Enum


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


class ModelMismatchError(ValueError):
    """The trained ML model does not fit the run asking for predictions.

    One piece of user advice, one class: this one exists so the koopmans
    package can advise retraining under the run's settings or changing
    ``model_file``. Subclassing ``ValueError`` keeps every existing
    handler catching; ``field`` names the mismatched model stamp when
    the raise site knows it.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        """Store ``message`` and, when known, the mismatched stamp ``field``."""
        super().__init__(message)
        self.field = field
