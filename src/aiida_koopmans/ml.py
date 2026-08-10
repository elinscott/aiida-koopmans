"""Machine-learning vocabulary for the screening-parameter workflows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from math import isclose
from typing import Any, TypedDict

from pydantic import BaseModel, ConfigDict

#: pw2wannier90.x spells the radial-basis settings ``decompose_<key>`` in
#: its ``&inputpp`` namelist.
DECOMPOSE_KEY_PREFIX = "decompose_"


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


class RadialBasisSettings(BaseModel):
    """The ``decompose_*`` keys of a pw2wannier90.x ``&inputpp`` namelist.

    Each field is read under its ``decompose_``-prefixed alias and only
    that alias; every other ``&inputpp`` key is ignored. The defaults are
    the values the Koopmans descriptor is defined against (the legacy
    ``ml`` settings, not the binary's own ``n_max=l_max=6``), so the
    CalcJob writes all four into the namelist.
    """

    model_config = ConfigDict(alias_generator=lambda name: f"{DECOMPOSE_KEY_PREFIX}{name}")

    n_max: int = 4
    l_max: int = 4
    r_min: float = 0.5
    r_max: float = 4.0


#: Every :class:`RadialBasis` field, for the callers that walk the
#: settings by name (a model's stamp, a pw2wannier90.x namelist).
RADIAL_BASIS_KEYS: tuple[str, ...] = tuple(RadialBasisSettings.model_fields)

#: The fields that compare as integers; the rest compare as floats.
_INTEGER_BASIS_KEYS = frozenset(
    name for name, field in RadialBasisSettings.model_fields.items() if field.annotation is int
)


def resolve_radial_basis(decompose_parameters: Mapping[str, Any] | None) -> RadialBasis:
    """Return the radial basis a decompose pass runs with.

    ``decompose_parameters`` is the pw2wannier90.x ``&inputpp`` override
    dict; keys are matched case-insensitively and anything it leaves out
    takes the :class:`RadialBasisSettings` default, the same fallback the
    CalcJob injects. Callers stamp the result into a trained model and
    compare it against the model's stamp before predicting.
    """
    # A graph body receives these as an ``orm.Dict``, which pydantic will
    # not validate; iterating ``.items()`` is what makes them parseable.
    supplied = {str(key).lower(): value for key, value in (decompose_parameters or {}).items()}
    settings = RadialBasisSettings.model_validate(supplied)
    return RadialBasis(
        n_max=settings.n_max,
        l_max=settings.l_max,
        r_min=settings.r_min,
        r_max=settings.r_max,
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
