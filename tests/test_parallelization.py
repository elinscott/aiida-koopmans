"""Unit tests for the per-code parallelization helpers.

Pure dict manipulation — no AiiDA profile or workgraph build needed.
"""

from __future__ import annotations

import pytest

from aiida_koopmans.types import CODE_NAMES
from aiida_koopmans.workgraphs import (
    merge_parallelization_into_existing_namespaces,
    merge_parallelization_into_inputs,
    merge_parallelization_into_overrides,
    omp_prepend_text,
    resolve_parallelization,
    validate_parallelization,
)


class TestValidate:
    def test_unknown_key_raises_with_the_name(self):
        with pytest.raises(ValueError, match=r"unknown parallelization code name.*pww"):
            validate_parallelization({"pw": {"npool": 2}, "pww": {"npool": 2}})

    def test_valid_sparse_mapping_passes(self):
        # A subset of codes with assorted fields is accepted (no raise).
        validate_parallelization({"pw": {"npool": 2}, "kcw": {"pd": True}})

    def test_none_and_empty_pass(self):
        validate_parallelization(None)
        validate_parallelization({})


class TestResolve:
    def test_ntasks_and_npool(self):
        options, settings = resolve_parallelization({"pw": {"ntasks": 8, "npool": 4}}, "pw")
        assert options == {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 8}}
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
            {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 3}},
            {},
        )

    def test_npool_and_pd_ordering(self):
        """Both flags emit npool-before-pd; pd renders lowercase ``true``."""
        _, settings = resolve_parallelization({"kcw": {"npool": 4, "pd": True}}, "kcw")
        assert settings == {"cmdline": ["-npool", "4", "-pd", "true"]}

    def test_pd_only(self):
        _, settings = resolve_parallelization({"pw2wannier90": {"pd": True}}, "pw2wannier90")
        assert settings == {"cmdline": ["-pd", "true"]}

    @pytest.mark.parametrize("code", ["ph", "pw2wannier90"])
    def test_npool_and_pd_for_ph_and_pw2wannier90(self, code):
        """The ph and pw2wannier90 codes accept both flags (QE parses them globally)."""
        _, settings = resolve_parallelization({code: {"npool": 2, "pd": True}}, code)
        assert settings == {"cmdline": ["-npool", "2", "-pd", "true"]}

    @pytest.mark.parametrize("code", ["kcp", "wann2kcp", "wannier90"])
    def test_npool_for_non_pool_code_raises(self, code):
        with pytest.raises(ValueError, match="does not parallelize over"):
            resolve_parallelization({code: {"npool": 2}}, code)

    @pytest.mark.parametrize("code", ["kcp", "wann2kcp", "wannier90"])
    def test_pd_for_non_pd_code_raises(self, code):
        with pytest.raises(ValueError, match="pencil decomposition"):
            resolve_parallelization({code: {"pd": True}}, code)

    def test_pools_false_suppresses_npool_but_keeps_pd(self):
        """The kcw.x ham step drops -npool but still takes -pd."""
        _, settings = resolve_parallelization({"kcw": {"npool": 4, "pd": True}}, "kcw", pools=False)
        assert settings == {"cmdline": ["-pd", "true"]}


class TestOmp:
    def test_prepend_text_exports_all_three_at_the_count(self):
        options, settings = resolve_parallelization({"pw": {"omp": 4}}, "pw")
        assert settings == {}
        assert options == {
            "prepend_text": (
                "export OMP_NUM_THREADS=4\nexport OPENBLAS_NUM_THREADS=4\nexport MKL_NUM_THREADS=4"
            )
        }

    def test_omp_alongside_ntasks(self):
        options, _ = resolve_parallelization({"pw": {"ntasks": 8, "omp": 2}}, "pw")
        assert options["resources"] == {"num_machines": 1, "num_mpiprocs_per_machine": 8}
        assert options["prepend_text"] == omp_prepend_text(2)

    @pytest.mark.parametrize("code", list(CODE_NAMES))
    def test_omp_accepted_for_every_code(self, code):
        """Accept omp for every code — it is an env-level knob with no support matrix.

        Even the codes that reject npool/pd (kcp, wann2kcp, wannier90) take it.
        """
        options, _ = resolve_parallelization({code: {"omp": 3}}, code)
        assert options == {"prepend_text": omp_prepend_text(3)}

    def test_appends_to_existing_prepend_text(self):
        inputs = {"metadata": {"options": {"prepend_text": "module load qe"}}}
        merge_parallelization_into_inputs(inputs, {"kcw": {"omp": 2}}, "kcw")
        assert inputs["metadata"]["options"]["prepend_text"] == (
            f"module load qe\n{omp_prepend_text(2)}"
        )

    def test_preserves_resources_when_appending_prepend(self):
        inputs = {"metadata": {"options": {"max_wallclock_seconds": 3600}}}
        merge_parallelization_into_inputs(inputs, {"pw": {"omp": 2}}, "pw")
        options = inputs["metadata"]["options"]
        assert options["max_wallclock_seconds"] == 3600
        assert options["prepend_text"] == omp_prepend_text(2)


class TestApplyToCalcJob:
    def test_merges_and_preserves_existing(self):
        inputs = {"metadata": {"call_link_label": "screen"}, "settings": {"a": 1}}
        merge_parallelization_into_inputs(inputs, {"kcw": {"ntasks": 2, "npool": 4}}, "kcw")
        assert inputs["metadata"]["call_link_label"] == "screen"
        assert inputs["metadata"]["options"]["resources"]["num_mpiprocs_per_machine"] == 2
        assert inputs["settings"] == {"a": 1, "cmdline": ["-npool", "4"]}

    def test_pools_false_drops_npool(self):
        inputs = {"metadata": {"call_link_label": "ham"}}
        merge_parallelization_into_inputs(
            inputs, {"kcw": {"npool": 4, "pd": True}}, "kcw", pools=False
        )
        assert inputs["settings"]["cmdline"] == ["-pd", "true"]

    def test_no_config_is_a_noop(self):
        inputs = {"metadata": {"call_link_label": "x"}}
        merge_parallelization_into_inputs(inputs, None, "kcw")
        assert inputs == {"metadata": {"call_link_label": "x"}}


class TestInjectOverrides:
    def test_nested_and_direct_namespaces(self):
        overrides: dict = {}
        merge_parallelization_into_overrides(
            overrides,
            {"pw": {"npool": 4}, "projwfc": {"ntasks": 2}},
            [(("scf", "pw"), "pw"), (("projwfc",), "projwfc")],
        )
        assert overrides["scf"]["pw"]["settings"]["cmdline"] == ["-npool", "4"]
        assert (
            overrides["projwfc"]["metadata"]["options"]["resources"]["num_mpiprocs_per_machine"]
            == 2
        )


class TestApplyPresent:
    def test_only_existing_namespaces(self):
        data = {"wannier90": {"wannier90": {"parameters": {}}}}
        merge_parallelization_into_existing_namespaces(
            data,
            {"wannier90": {"ntasks": 4}, "projwfc": {"ntasks": 2}},
            [(("wannier90", "wannier90"), "wannier90"), (("projwfc", "projwfc"), "projwfc")],
        )
        w90 = data["wannier90"]["wannier90"]
        assert w90["metadata"]["options"]["resources"]["num_mpiprocs_per_machine"] == 4
        # The absent projwfc namespace is not created.
        assert "projwfc" not in data


class TestSchedulerCanary:
    """Pin how the installed schedulers interpret the emitted resource shape.

    The hyperqueue resource class silently drops unknown keys (a
    ``tot_num_mpiprocs``-only mapping once yielded single-rank jobs).  A
    revert to that shape is caught by the exact-dict assertions above; the
    canaries here instead catch a *plugin upgrade* that stops consuming the
    emitted pair, so the hyperqueue one only bites where aiida-hyperqueue is
    installed (a test dependency for exactly that reason).
    """

    def test_hyperqueue_consumes_emitted_shape(self):
        hq = pytest.importorskip("aiida_hyperqueue.scheduler")
        options, _ = resolve_parallelization({"pw": {"ntasks": 8}}, "pw")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            resources = hq.HyperQueueJobResource.validate_resources(**options["resources"])
        assert resources.num_cpus == 8

    def test_direct_scheduler_consumes_emitted_shape(self):
        """Check compatibility only: direct accepts old and new shapes alike."""
        from aiida.schedulers.plugins.direct import DirectJobResource

        options, _ = resolve_parallelization({"pw": {"ntasks": 8}}, "pw")
        resources = DirectJobResource(**options["resources"])
        assert resources.num_machines * resources.num_mpiprocs_per_machine == 8

    def test_unconfigured_code_is_skipped(self):
        data = {"projwfc": {"projwfc": {"parameters": {}}}}
        merge_parallelization_into_existing_namespaces(
            data,
            {"pw": {"ntasks": 4}},
            [(("projwfc", "projwfc"), "projwfc")],
        )
        # No pw entry in the mapping and no projwfc config: data is untouched.
        assert data == {"projwfc": {"projwfc": {"parameters": {}}}}
