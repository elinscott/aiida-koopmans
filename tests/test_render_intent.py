"""``stamp_render_intent`` / ``mark_step``: the render-intent extras stamp.

``metadata`` is spec'd (``label``, ``description``, ``call_link_label``,
``store_provenance``, ``disable_cache``) — a render-intent flag cannot ride
it the way ``metadata.label`` does. Extras have no such spec and are
settable on any stored node, but only once that node exists: the stamp
lands on the calling process's own extras when ``stamp_render_intent``
actually *runs*, not when the graph that calls it is built. This file
tests that run-time behaviour directly; the wiring — which graph calls
``mark_step`` with which flag — is covered where the two callers live
(``TestKoopmansDSCFGraphBuild.test_render_intent_marker`` in
``test_kcp_workgraph.py``).
"""

from __future__ import annotations


class TestStampRenderIntent:
    def test_sets_the_callers_extras(self, aiida_profile):
        """Running ``stamp_render_intent`` as a child task stamps its caller, not itself."""
        from aiida_workgraph import WorkGraph

        from aiida_koopmans.workgraphs import stamp_render_intent

        wg = WorkGraph("render_intent_probe")
        wg.add_task(stamp_render_intent, name="marker", kind="transparent")
        wg.run()

        assert wg.process.base.extras.get("koopmans_render") == {"transparent": True}
        # The stamp is on the caller (the probe graph itself), not on the
        # marker task's own process node.
        marker_node = wg.tasks["marker"].process
        assert "koopmans_render" not in marker_node.base.extras

    def test_numbered_kind(self, aiida_profile):
        from aiida_workgraph import WorkGraph

        from aiida_koopmans.workgraphs import stamp_render_intent

        wg = WorkGraph("render_intent_probe_numbered")
        wg.add_task(stamp_render_intent, name="marker", kind="numbered")
        wg.run()

        assert wg.process.base.extras.get("koopmans_render") == {"numbered": True}

    def test_a_plain_run_carries_no_stamp(self, aiida_profile):
        """A graph that never calls ``stamp_render_intent`` gets no ``koopmans_render`` extra."""
        from aiida_workgraph import WorkGraph

        from aiida_koopmans.workgraphs.kcp import echo_alpha_screening

        wg = WorkGraph("plain_probe")
        wg.add_task(
            echo_alpha_screening,
            name="echo",
            alphas={"filled": {"none": [0.6]}, "empty": {"none": [0.6]}},
        )
        wg.run()

        assert "koopmans_render" not in wg.process.base.extras
