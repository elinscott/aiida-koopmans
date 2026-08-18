"""Tests for the keyword-ownership declarations the koopmans schema is built from."""

from __future__ import annotations

import pytest

from aiida_koopmans.owned_keywords import OWNED, ROUTE_CONDITIONAL, SEEDED, owned, seeded


def test_owned_accepts_a_declared_keyword():
    literal = {"epsil": True, "trans": False}
    assert owned("ph.INPUTPH", literal) is literal


def test_seeded_accepts_a_declared_keyword():
    literal = {"num_iter": 10000}
    assert seeded("wannier90", literal) is literal


def test_owned_rejects_an_unclassified_keyword():
    with pytest.raises(ValueError, match=r"forces ph.INPUTPH nogg.*OWNED"):
        owned("ph.INPUTPH", {"epsil": True, "nogg": True})


def test_seeded_rejects_an_unclassified_keyword():
    with pytest.raises(ValueError, match=r"seeds wannier90 fermi_energy"):
        seeded("wannier90", {"fermi_energy": 0.0})


def test_owned_rejects_a_keyword_that_is_only_seeded():
    # Seeding a keyword and forcing it are different promises to the user, so
    # the seeded classification must not satisfy the owned check.
    with pytest.raises(ValueError, match="num_iter"):
        owned("wannier90", {"num_iter": 10})


def test_conditional_keywords_pass_either_check():
    # The known gaps are settable and forced on different steps; both checks
    # let them through so no route has to work around the classification.
    assert owned("pw.SYSTEM", {"nosym": True}) == {"nosym": True}
    assert seeded("pw.SYSTEM", {"nosym": True}) == {"nosym": True}


def test_no_keyword_is_both_owned_and_seeded():
    for block in OWNED.keys() | SEEDED.keys():
        overlap = OWNED.get(block, frozenset()) & SEEDED.get(block, frozenset())
        assert not overlap, f"{block}: {sorted(overlap)}"


def test_conditional_keywords_are_not_also_classified():
    for block, keywords in ROUTE_CONDITIONAL.items():
        classified = OWNED.get(block, frozenset()) | SEEDED.get(block, frozenset())
        assert not keywords & classified, f"{block}: {sorted(keywords & classified)}"
