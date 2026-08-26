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


def test_fixture_pseudos_state_their_valence_orbitals(
    fake_cutoffs_family, silicon_structure, ozone_structure
):
    """The fixture UPFs must name their valence orbitals the way a real one does.

    ``aiida-wannier90-workflows`` reads the orbitals of any pseudo outside its
    curated tables off the labels of the ``PP_CHI`` atomic wave functions, so a
    fixture whose wave functions carry only ``l`` resolves to nothing and takes
    every wannierization test down with it.
    """
    import warnings

    from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals
    from upf_tools import UPFDict

    for structure, element, orbitals in (
        (silicon_structure, "Si", ["3S", "3P"]),
        (ozone_structure, "O", ["2S", "2P"]),
    ):
        pseudos = resolve_pseudo_family(fake_cutoffs_family.label, structure)
        chi = UPFDict.from_str(pseudos[element].get_content())["pswfc"]["chi"]
        assert [entry["label"] for entry in chi] == orbitals
        with warnings.catch_warnings():
            # Out-of-table pseudos warn that no semicore states are excluded.
            warnings.simplefilter("ignore", UserWarning)
            assert get_pseudo_orbitals(pseudos)[element]["pswfcs"] == orbitals
