"""Machine-learning vocabulary for the screening-parameter workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from math import isclose
from typing import Any

#: The radial-basis settings that define a ``power_spectrum`` descriptor,
#: with the values the Koopmans descriptor is defined against (the legacy
#: ``ml`` defaults). Two models trained under different values describe
#: different quantities, so these are stamped into a trained model and
#: re-checked before it predicts.
RADIAL_BASIS_DEFAULTS: dict[str, float | int] = {
    "n_max": 4,
    "l_max": 4,
    "r_min": 0.5,
    "r_max": 4.0,
}

#: pw2wannier90.x spells the same settings ``decompose_<key>`` in its
#: ``&inputpp`` namelist.
DECOMPOSE_KEY_PREFIX = "decompose_"


def resolve_radial_basis(decompose_parameters: Mapping[str, Any] | None) -> dict[str, float | int]:
    """Return the radial basis a decompose pass runs with.

    ``decompose_parameters`` is the pw2wannier90.x ``&inputpp`` override
    dict, whose keys carry the ``decompose_`` prefix; anything it leaves
    out falls back to :data:`RADIAL_BASIS_DEFAULTS`, the same fallback the
    CalcJob injects. Callers stamp the result into a trained model and
    compare it against the model's stamp before predicting.
    """
    supplied = {str(key).lower(): value for key, value in (decompose_parameters or {}).items()}
    resolved: dict[str, float | int] = {}
    for key, default in RADIAL_BASIS_DEFAULTS.items():
        value = supplied.get(f"{DECOMPOSE_KEY_PREFIX}{key}", default)
        resolved[key] = int(value) if key in ("n_max", "l_max") else float(value)
    return resolved


def radial_basis_mismatches(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    keys: Iterable[str] | None = None,
) -> list[str]:
    """Return the radial-basis keys on which two settings disagree.

    ``n_max`` / ``l_max`` compare as integers and ``r_min`` / ``r_max``
    as floats within a relative tolerance, so a value that has been
    through a text header or a JSON round-trip still matches. A key
    missing from either side counts as a disagreement, which is what
    makes an unstamped model fail the comparison; pass ``keys`` to
    restrict the comparison to the settings a source actually reports.
    """
    mismatched: list[str] = []
    for key in RADIAL_BASIS_DEFAULTS if keys is None else keys:
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            mismatched.append(key)
        elif key in ("n_max", "l_max"):
            if int(a) != int(b):
                mismatched.append(key)
        elif not isclose(float(a), float(b), rel_tol=1e-9):
            mismatched.append(key)
    return mismatched


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
