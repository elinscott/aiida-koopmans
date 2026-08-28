"""Tests for the keyword-ownership declarations the koopmans schema is built from."""

from __future__ import annotations

import pytest

from aiida_koopmans.owned_keywords import (
    OWNED,
    ROUTE_CONDITIONAL,
    SEEDED,
    SEEDED_VALUES,
    owned,
    seeded,
)


def test_owned_accepts_a_declared_keyword():
    literal = {"epsil": True, "trans": False}
    assert owned("ph.INPUTPH", literal) is literal


def test_seeded_accepts_a_declared_keyword():
    literal = {"num_iter": 10000}
    assert seeded("wannier90", literal) is literal


def test_seeded_rejects_a_value_that_disagrees_with_the_roster():
    # koopmans reads SEEDED_VALUES as the generated field's default, so a
    # route literal that disagrees would make the schema lie about what the
    # route actually seeds.
    with pytest.raises(ValueError, match=r"kcw\.SCREEN tr2=1e-10.*roster: 1e-18"):
        seeded("kcw.SCREEN", {"tr2": 1e-10})


def test_seeded_accepts_a_keyword_with_no_roster_value():
    # pw.SYSTEM.starting_magnetization is seeded by name only: the route's
    # value depends on the spin regime at runtime, so SEEDED_VALUES carries
    # no entry for it and seeded() cannot value-check it.
    literal = {"starting_magnetization": 0.1}
    assert seeded("pw.SYSTEM", literal) is literal


def test_every_seeded_keyword_with_a_roster_value_is_in_seeded():
    # SEEDED_VALUES is a source for SEEDED, not a separate registry: every
    # keyword it pins must also be an accepted name.
    for block, values in SEEDED_VALUES.items():
        assert set(values) <= SEEDED[block], block


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
    # One route forces these and the others honour them, so both checks let
    # them through; koopmans refuses each on the route that forces it.
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
