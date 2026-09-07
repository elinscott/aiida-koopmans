"""Tests for the package-level hyperqueue compatibility patch."""

from __future__ import annotations

import sys

import pytest

from aiida_koopmans import _patch_hyperqueue_accepts_computer_default


def test_patch_flips_the_classmethod():
    hq = pytest.importorskip("aiida_hyperqueue.scheduler")
    _patch_hyperqueue_accepts_computer_default()
    assert hq.HyperQueueJobResource.accepts_default_mpiprocs_per_machine() is True


def test_missing_plugin_is_tolerated(monkeypatch):
    # A None entry makes ``from aiida_hyperqueue.scheduler import ...`` raise
    # ImportError, exercising the not-installed branch.
    monkeypatch.setitem(sys.modules, "aiida_hyperqueue.scheduler", None)
    _patch_hyperqueue_accepts_computer_default()
