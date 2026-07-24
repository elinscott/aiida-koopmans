"""Shared pytest fixtures for aiida-koopmans tests.

Loads AiiDA's own pytest fixtures (``aiida_profile``, ``aiida_localhost``, ...) so
individual tests can request them without having to boot a profile manually.
Project-specific fixtures are defined in ``tests/fixtures.py`` and re-exported
below so pytest picks them up for every test module.

The ``generate_calc_job``, ``generate_calc_job_node``, ``generate_parser``,
``fixture_sandbox`` and ``fixture_localhost`` fixtures mirror the naming used
by ``aiida-quantumespresso`` so porting patterns between the two projects is
straightforward.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from tests.fixtures import (  # noqa: F401
    auto_codes,
    fake_cutoffs_family,
    fake_upf,
    generate_upf_data,
    kcp_code,
    kmesh,
    kpath,
    nscf_remote,
    ozone_pseudo_family,
    ozone_pseudos,
    ozone_real_pseudos,
    ozone_structure,
    periodic_ozone_structure,
    si_reference,
    silicon_structure,
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
def clear_database_after_test(aiida_profile):
    """Override the deprecated-and-broken upstream fixture with a no-op yield."""
    yield aiida_profile


@pytest.fixture(scope="function")
def clear_database(clear_database_after_test):
    """Override the deprecated alias to avoid the broken teardown chain."""
    yield


# ----------------------------------------------------------------------
# aiida-qe-style fixtures
# ----------------------------------------------------------------------


@pytest.fixture(scope="session")
def filepath_tests():
    """Return the absolute filepath of the ``tests`` folder.

    Mirrors the fixture of the same name in ``aiida-quantumespresso``.
    """
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def fixture_sandbox():
    """Yield an ephemeral ``SandboxFolder`` for CalcJob input rendering."""
    from aiida.common.folders import SandboxFolder

    with SandboxFolder() as folder:
        yield folder


@pytest.fixture
def fixture_localhost(aiida_localhost):
    """Return a ``localhost`` Computer with one MPI proc per machine."""
    localhost = aiida_localhost
    localhost.set_default_mpiprocs_per_machine(1)
    return localhost


@pytest.fixture
def generate_calc_job():
    """Return a factory that runs ``prepare_for_submission`` on a CalcJob.

    Usage::

        calc_info = generate_calc_job(fixture_sandbox, "koopmans.kcp", inputs)

    Returns the ``CalcInfo`` from ``prepare_for_submission``; the sandbox
    folder is mutated in place with the rendered input files.
    """

    def _generate_calc_job(folder, entry_point_name, inputs=None):
        from aiida.engine.utils import instantiate_process
        from aiida.manage.manager import get_manager
        from aiida.plugins import CalculationFactory

        manager = get_manager()
        runner = manager.get_runner()

        process_class = CalculationFactory(entry_point_name)
        process = instantiate_process(runner, process_class, **(inputs or {}))
        return process.prepare_for_submission(folder)

    return _generate_calc_job


@pytest.fixture
def generate_parser():
    """Return a factory that resolves the ``Parser`` class for an entry point."""

    def _generate_parser(entry_point_name):
        from aiida.plugins import ParserFactory

        return ParserFactory(entry_point_name)

    return _generate_parser


@pytest.fixture
def generate_calc_job_node(fixture_localhost, filepath_tests, tmp_path_factory):  # noqa: C901
    """Return a factory that builds a stored ``CalcJobNode`` with a retrieved folder.

    Follows the ``aiida-quantumespresso`` pattern: a test-name directory under
    ``tests/parsers/fixtures/<subfolder>/<test_name>`` is copied into the
    retrieved folder so the parser reads real files rather than mocks.
    """

    def flatten_inputs(inputs, prefix=""):
        flat = []
        for key, value in inputs.items():
            if isinstance(value, Mapping):
                flat.extend(flatten_inputs(value, prefix=prefix + key + "__"))
            else:
                flat.append((prefix + key, value))
        return flat

    def _generate_calc_job_node(
        entry_point_name="koopmans.kcp",
        computer=None,
        test_name=None,
        inputs=None,
        attributes=None,
        fixture_subdir=None,
        input_filename="aiida.cpi",
        output_filename="aiida.cpo",
    ):
        from aiida import orm
        from aiida.common import LinkType
        from aiida.plugins.entry_point import format_entry_point_string

        if computer is None:
            computer = fixture_localhost

        entry_point = format_entry_point_string("aiida.calculations", entry_point_name)

        node = orm.CalcJobNode(computer=computer, process_type=entry_point)
        node.base.attributes.set("input_filename", input_filename)
        node.base.attributes.set("output_filename", output_filename)
        node.base.attributes.set("error_filename", "aiida.err")
        node.set_option("resources", {"num_machines": 1, "num_mpiprocs_per_machine": 1})
        node.set_option("max_wallclock_seconds", 1800)

        if attributes:
            node.base.attributes.set_many(attributes)

        filepath_folder = None
        if test_name is not None:
            subdir = fixture_subdir or entry_point_name.split(".", 1)[-1]
            filepath_folder = os.path.join(filepath_tests, "parsers", "fixtures", subdir, test_name)

        if inputs:
            for link_label, input_node in flatten_inputs(inputs):
                if not input_node.is_stored:
                    input_node.store()
                node.base.links.add_incoming(
                    input_node, link_type=LinkType.INPUT_CALC, link_label=link_label
                )

        node.store()

        if filepath_folder is not None:
            if not Path(filepath_folder).exists():
                raise FileNotFoundError(f"Fixture folder not found: {filepath_folder}")
            retrieved = orm.FolderData()
            retrieved.base.repository.put_object_from_tree(filepath_folder)
            retrieved.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label="retrieved"
            )
            retrieved.store()

            remote_folder = orm.RemoteData(
                computer=computer,
                remote_path=tmp_path_factory.mktemp("cj-tmp").as_posix(),
            )
            remote_folder.base.links.add_incoming(
                node, link_type=LinkType.CREATE, link_label="remote_folder"
            )
            remote_folder.store()

        return node

    return _generate_calc_job_node
