"""Machine-learning vocabulary for the screening-parameter workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from math import isclose
from typing import Any, TypedDict


class RadialBasis(TypedDict):
    """The radial basis a ``power_spectrum`` descriptor is expanded on.

    ``n_max`` and ``l_max`` are the radial and angular expansion orders;
    ``r_min`` and ``r_max`` bound the radial window in Bohr. Two models
    trained under different values describe different quantities, so a
    trained model carries these and re-checks them before it predicts.
    """

    n_max: int
    l_max: int
    r_min: float
    r_max: float


#: Every :class:`RadialBasis` field, for the callers that walk the
#: settings by name (a model's stamp, a pw2wannier90.x namelist).
RADIAL_BASIS_KEYS: tuple[str, ...] = ("n_max", "l_max", "r_min", "r_max")

#: The fields that compare as integers; the rest compare as floats.
_INTEGER_BASIS_KEYS = frozenset({"n_max", "l_max"})

#: The values the Koopmans descriptor is defined against (the legacy
#: ``ml`` defaults), used wherever a decompose pass leaves a setting out.
RADIAL_BASIS_DEFAULTS: RadialBasis = {
    "n_max": 4,
    "l_max": 4,
    "r_min": 0.5,
    "r_max": 4.0,
}

#: pw2wannier90.x spells the same settings ``decompose_<key>`` in its
#: ``&inputpp`` namelist.
DECOMPOSE_KEY_PREFIX = "decompose_"


def resolve_radial_basis(decompose_parameters: Mapping[str, Any] | None) -> RadialBasis:
    """Return the radial basis a decompose pass runs with.

    ``decompose_parameters`` is the pw2wannier90.x ``&inputpp`` override
    dict, whose keys carry the ``decompose_`` prefix; anything it leaves
    out falls back to :data:`RADIAL_BASIS_DEFAULTS`, the same fallback the
    CalcJob injects. Callers stamp the result into a trained model and
    compare it against the model's stamp before predicting.
    """
    supplied = {str(key).lower(): value for key, value in (decompose_parameters or {}).items()}

    def setting(key: str, default: float | int) -> Any:
        return supplied.get(f"{DECOMPOSE_KEY_PREFIX}{key}", default)

    return RadialBasis(
        n_max=int(setting("n_max", RADIAL_BASIS_DEFAULTS["n_max"])),
        l_max=int(setting("l_max", RADIAL_BASIS_DEFAULTS["l_max"])),
        r_min=float(setting("r_min", RADIAL_BASIS_DEFAULTS["r_min"])),
        r_max=float(setting("r_max", RADIAL_BASIS_DEFAULTS["r_max"])),
    )


def radial_basis_mismatches(
    stamped: Mapping[str, Any],
    wanted: Mapping[str, Any],
    keys: Iterable[str] | None = None,
) -> list[str]:
    """Return the radial-basis keys on which two settings disagree.

    Both sides are loose mappings because both have been through
    storage: ``stamped`` is a trained model's own record or a decompose
    run's reported namelist, ``wanted`` a basis that reached this process
    as an ``orm.Dict``. So the values are compared by value, not by type:
    ``n_max`` / ``l_max`` as integers and ``r_min`` / ``r_max`` as floats
    within a relative tolerance. A key missing from either side counts as
    a disagreement; pass ``keys`` to restrict the comparison to the
    settings a source actually reports.
    """
    mismatched: list[str] = []
    for key in RADIAL_BASIS_KEYS if keys is None else keys:
        a, b = stamped.get(key), wanted.get(key)
        if a is None or b is None:
            mismatched.append(key)
        elif key in _INTEGER_BASIS_KEYS:
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
