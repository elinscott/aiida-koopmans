============
Contributing
============

Every command below is a `hatch <https://hatch.pypa.io>`_ environment
declared in ``pyproject.toml``, so none of them needs a virtual environment
prepared by hand. The fork dependencies still have to be checked out as
siblings first — see :doc:`../installation`.

Tests
=====

::

    hatch test                  # your current Python
    hatch test --all            # every Python in the matrix
    hatch test --coverage
    hatch test --pdb            # drop into ipdb on failure

Arbitrary pytest flags go at the end of the command. Tests live in
``tests/``, use the AiiDA test profile through the fixtures in
``conftest.py``, and share their builders through ``tests/fixtures.py`` —
define a fixture there rather than module-locally when a sibling module
could use it too. Do not mock AiiDA; run against a throwaway profile.

Formatting, linting, and types
==============================

::

    hatch fmt --check           # ruff format and lint, no changes
    hatch fmt                   # apply the fixable ones
    hatch run mypy:check

To run the checks before each commit::

    pip install -e .[pre-commit]
    pre-commit install

Documentation
=============

::

    hatch run docs:build

The build runs ``sphinx-apidoc`` over ``src/aiida_koopmans`` first, so the
API reference follows the package without a page per module being written by
hand. Warnings are errors: a broken reference or a page missing from a
toctree fails the build, locally and in continuous integration alike. Open
``docs/build/html/index.html`` to read the result.

Continuous integration
======================

``.github/workflows/ci.yml`` runs four jobs on every pull request: the test
suite on each supported Python, the documentation build, the formatter and
linter, and mypy. Each job that imports the package clones the fork
dependencies next to the checkout first; that clone list and
``[tool.uv.sources]`` have to agree, and both shrink as a fork merges
upstream.

Read the Docs builds the same documentation from ``.readthedocs.yml``, with
its own copy of the clone list.

A change to a graph's shape — a task name, a socket, the layout of a
namespace — needs a branch of the same name in the ``koopmans`` repository,
even an empty one. That repository's continuous integration clones the
same-named branch here, so without one it tests the new code against the old
sibling and goes red only after the merge.
