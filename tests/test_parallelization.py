"""Unit tests for the per-code parallelization helpers.

Pure dict manipulation — no AiiDA profile or workgraph build needed.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.workgraphs import (
    apply_parallelization,
    apply_parallelization_present,
    inject_parallelization,
    resolve_parallelization,
)


class TestResolve:
    def test_ntasks_and_npool(self):
        options, settings = resolve_parallelization({"pw": {"ntasks": 8, "npool": 4}}, "pw")
        assert options == {"resources": {"num_machines": 1, "tot_num_mpiprocs": 8}}
        assert settings == {"cmdline": ["-npool", "4"]}

    def test_missing_code_and_empty(self):
        assert resolve_parallelization({"pw": {"npool": 2}}, "kcw") == ({}, {})
        assert resolve_parallelization(None, "pw") == ({}, {})
        assert resolve_parallelization({}, "pw") == ({}, {})

    def test_partial_fields(self):
        assert resolve_parallelization({"pw": {"npool": 2}}, "pw") == (
            {},
            {"cmdline": ["-npool", "2"]},
        )
        assert resolve_parallelization({"pw": {"ntasks": 3}}, "pw") == (
            {"resources": {"num_machines": 1, "tot_num_mpiprocs": 3}},
            {},
        )

    def test_npool_for_non_pool_code_raises(self):
        with pytest.raises(ValueError, match="does not parallelize over"):
            resolve_parallelization({"wannier90": {"npool": 2}}, "wannier90")


class TestApplyToCalcJob:
    def test_merges_and_preserves_existing(self):
        inputs = {"metadata": {"call_link_label": "screen"}, "settings": {"a": 1}}
        apply_parallelization(inputs, {"kcw": {"ntasks": 2, "npool": 4}}, "kcw")
        assert inputs["metadata"]["call_link_label"] == "screen"
        assert inputs["metadata"]["options"]["resources"]["tot_num_mpiprocs"] == 2
        assert inputs["settings"] == {"a": 1, "cmdline": ["-npool", "4"]}

    def test_no_config_is_a_noop(self):
        inputs = {"metadata": {"call_link_label": "x"}}
        apply_parallelization(inputs, None, "kcw")
        assert inputs == {"metadata": {"call_link_label": "x"}}


class TestInjectOverrides:
    def test_nested_and_direct_namespaces(self):
        overrides: dict = {}
        inject_parallelization(
            overrides,
            {"pw": {"npool": 4}, "projwfc": {"ntasks": 2}},
            [(("scf", "pw"), "pw"), (("projwfc",), "projwfc")],
        )
        assert overrides["scf"]["pw"]["settings"]["cmdline"] == ["-npool", "4"]
        assert overrides["projwfc"]["metadata"]["options"]["resources"]["tot_num_mpiprocs"] == 2


class TestApplyPresent:
    def test_only_existing_namespaces(self):
        data = {"wannier90": {"wannier90": {"parameters": {}}}}
        apply_parallelization_present(
            data,
            {"wannier90": {"ntasks": 4}, "projwfc": {"ntasks": 2}},
            [(("wannier90", "wannier90"), "wannier90"), (("projwfc", "projwfc"), "projwfc")],
        )
        w90 = data["wannier90"]["wannier90"]
        assert w90["metadata"]["options"]["resources"]["tot_num_mpiprocs"] == 4
        # The absent projwfc namespace is not created.
        assert "projwfc" not in data
