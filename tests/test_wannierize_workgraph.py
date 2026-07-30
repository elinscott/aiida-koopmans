"""Construction-level tests for the full ``Wannierize`` graph (wannier90.py).

Builds the graph (no daemon, no real code execution) and checks the wrapped
``Wannier90WorkChain`` task and the returned output sockets. A run without
disentanglement leaves the ``disentanglement_data`` / ``spread_data`` sockets
absent; the graph's ``NotRequired`` outputs must link cleanly regardless.
"""

from __future__ import annotations

import pytest
from aiida_wannier90_workflows.common.types import WannierProjectionType

from aiida_koopmans.workgraphs.wannier90 import Wannierize
from tests.fixtures import assert_graph_roundtrips, si_external_projector_tables


class TestWannierizeGraphBuild:
    def test_full_graph_eager_builds(self, fake_cutoffs_family, silicon_structure, wannier_codes):
        """The whole ``Wannierize`` graph builds with its ``NotRequired`` outputs."""
        wg = Wannierize.build(
            codes=wannier_codes,
            structure=silicon_structure,
            pseudo_family=fake_cutoffs_family.label,
        )
        names = [t.name for t in wg.tasks]
        assert "Wannier90WorkChain" in names

        # Every declared output socket links; the ``NotRequired`` extras
        # (``disentanglement_data`` / ``spread_data``) do not break the linking
        # even though a plain run omits them.
        output_names = [socket._name for socket in wg.outputs]
        for expected in ("scf", "nscf", "wannier90", "wannier90_up", "wannier90_down", "projwfc"):
            assert expected in output_names, output_names

        assert_graph_roundtrips(wg)

    def test_exclude_semicore_with_external_projectors_raises(
        self, fake_cutoffs_family, silicon_structure, wannier_codes, tmp_path
    ):
        """Semicore exclusion cannot combine with label-free external tables."""
        with pytest.raises(ValueError, match="`exclude_semicore` is not supported"):
            Wannierize.build(
                codes=wannier_codes,
                structure=silicon_structure,
                pseudo_family=fake_cutoffs_family.label,
                projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
                external_projectors_path=str(tmp_path),
                external_projectors=si_external_projector_tables(),
                exclude_semicore=True,
            )
