"""Construction-level tests for the full ``Wannierize`` graph (wannier90.py).

Builds the graph (no daemon, no real code execution) and checks the wrapped
``Wannier90WorkChain`` task and the returned output sockets. A run without
disentanglement leaves the ``disentanglement_data`` / ``spread_data`` sockets
absent; the graph's ``NotRequired`` outputs must link cleanly regardless.
"""

from __future__ import annotations

from aiida_koopmans.workgraphs.wannier90 import Wannierize


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
