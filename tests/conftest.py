"""Shared pytest fixtures for aiida-koopmans tests.

Loads AiiDA's own pytest fixtures (``aiida_profile``, ``aiida_localhost``, ...) so
individual tests can request them without having to boot a profile manually.
Project-specific fixtures are defined in ``tests/fixtures.py`` and re-exported
below so pytest picks them up for every test module.
"""

import pytest

from tests.fixtures import (  # noqa: F401
    fake_upf,
    ozone_pseudos,
    ozone_structure,
    periodic_ozone_structure,
)

pytest_plugins = ["aiida.tools.pytest_fixtures"]


# The deprecated ``aiida.manage.tests.pytest_fixtures`` module (transitively loaded
# via aiida-core) registers an autouse ``clear_database_auto`` fixture that chains
# into ``clear_database_after_test``, which calls ``Profile.clear_profile()`` — a
# method that no longer exists on modern Profile objects. Override the chain to
# no-ops so tests that don't need a clean DB aren't tripped by the broken teardown.
# Tests that *do* need an isolated profile should request ``aiida_profile_clean``
# explicitly.


@pytest.fixture(scope="function")
def clear_database_after_test(aiida_profile):  # noqa: D401
    """Override the deprecated-and-broken upstream fixture with a no-op yield."""
    yield aiida_profile


@pytest.fixture(scope="function")
def clear_database(clear_database_after_test):  # noqa: ARG001
    """Override the deprecated alias to avoid the broken teardown chain."""
    yield
