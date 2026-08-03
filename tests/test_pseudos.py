"""Unit tests for pseudopotential-family resolution (``utils/pseudos.py``).

``resolve_pseudo_family_task`` (the ``@task.workfunction`` variant) is
exercised at the graph-construction level in ``test_kcp_workgraph.py``; its
own body only runs inside the AiiDA process engine, which is out of scope
for these pure-function tests.
"""

from __future__ import annotations

from aiida_koopmans.utils.pseudos import resolve_pseudo_family


def test_resolves_every_kind_in_the_structure(fake_cutoffs_family, ozone_structure):
    pseudos = resolve_pseudo_family(fake_cutoffs_family.label, ozone_structure)
    assert set(pseudos) == {"O"}
    assert pseudos["O"].element == "O"
