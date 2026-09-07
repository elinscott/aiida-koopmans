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


def test_already_fixed_upstream_is_left_alone(monkeypatch):
    """A future ``aiida-hyperqueue`` that already returns ``True`` is not touched."""
    hq = pytest.importorskip("aiida_hyperqueue.scheduler")
    fixed = classmethod(lambda cls: True)
    monkeypatch.setattr(hq.HyperQueueJobResource, "accepts_default_mpiprocs_per_machine", fixed)
    _patch_hyperqueue_accepts_computer_default()
    # Attribute access always rewraps a classmethod into a fresh bound method,
    # so compare the raw descriptor in the class ``__dict__`` for identity.
    assert hq.HyperQueueJobResource.__dict__["accepts_default_mpiprocs_per_machine"] is fixed


def test_renamed_hook_is_tolerated(monkeypatch):
    """A future ``aiida-hyperqueue`` that drops the hook does not raise."""
    hq = pytest.importorskip("aiida_hyperqueue.scheduler")
    monkeypatch.delattr(hq.HyperQueueJobResource, "accepts_default_mpiprocs_per_machine")
    _patch_hyperqueue_accepts_computer_default()
