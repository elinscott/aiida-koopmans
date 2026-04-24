"""Shared test data, helper classes, and pytest fixtures.

Definitions live here; ``conftest.py`` just re-exports the fixtures so
pytest's collection machinery picks them up for every test module.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

# Ozone geometry taken from koopmans/tutorials/tutorial_1/ozone.json.
_OZONE_CELL = [[14.1738, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.66]]
_OZONE_POSITIONS = [
    ("O", [7.0869, 6.0, 5.89]),
    ("O", [8.1738, 6.0, 6.55]),
    ("O", [6.0, 6.0, 6.55]),
]


class _FakeUpf(SimpleNamespace):
    """Stand-in for an ``aiida-pseudo`` ``UpfData`` node.

    Exposes only the attributes our rendering / electron-counting helpers read
    (``filename``, ``uuid``, ``z_valence``).
    """


def _build_ozone_structure(pbc: bool):
    from aiida.orm import StructureData

    struct = StructureData(cell=_OZONE_CELL, pbc=pbc)
    for symbol, position in _OZONE_POSITIONS:
        struct.append_atom(position=position, symbols=symbol, name=symbol)
    return struct


@pytest.fixture
def ozone_structure(aiida_profile):
    """Return an ozone (O3) ``StructureData`` with the tutorial_1 geometry, non-periodic."""
    return _build_ozone_structure(pbc=False)


@pytest.fixture
def periodic_ozone_structure(aiida_profile):
    """Return the ozone geometry with ``pbc=True`` for exercising periodic scope guards."""
    return _build_ozone_structure(pbc=True)


@pytest.fixture
def fake_upf():
    """Return a factory class for stand-in UpfData objects.

    Usage in tests::

        def test_something(fake_upf):
            upf = fake_upf(filename="O.upf", uuid="abc", z_valence=6.0)
    """
    return _FakeUpf


@pytest.fixture
def ozone_pseudos(fake_upf):
    """Return the ozone pseudos dict ``{"O": FakeUpf(...)}`` with oxygen's valence."""
    return {"O": fake_upf(filename="O.upf", uuid="fake-upf-uuid", z_valence=6.0)}
