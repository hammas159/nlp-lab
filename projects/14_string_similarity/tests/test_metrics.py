"""Tests for the seven string measures.

No dataset and no network. Every distance is checked against a pair whose answer can be
counted by hand, because an edit distance that is off by one still ranks candidates
plausibly and produces a believable accuracy figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import (
    SIMILARITIES,
    bigram,
    damerau,
    jaro,
    jaro_winkler,
    keyboard,
    levenshtein,
    metaphone,
    similarity,
    soundex,
)

# --- Levenshtein --------------------------------------------------------------------------


def test_identical_strings_are_zero_apart():
    assert levenshtein("kitten", "kitten") == 0


def test_the_textbook_example():
    """kitten -> sitten -> sittin -> sitting is three edits."""
    assert levenshtein("kitten", "sitting") == 3


def test_a_single_deletion():
    assert levenshtein("cart", "car") == 1


def test_an_empty_string_costs_its_length():
    assert levenshtein("", "abc") == 3


def test_it_is_symmetric():
    assert levenshtein("flaw", "lawn") == levenshtein("lawn", "flaw")


# --- Damerau ------------------------------------------------------------------------------


def test_a_transposition_costs_one_under_damerau():
    assert damerau("form", "from") == 1


def test_the_same_transposition_costs_two_under_levenshtein():
    """This is the whole reason Damerau exists: swapping adjacent letters is one of the
    commonest typing errors, and plain Levenshtein charges double for it."""
    assert levenshtein("form", "from") == 2


def test_damerau_agrees_with_levenshtein_when_no_transposition_helps():
    assert damerau("cart", "car") == levenshtein("cart", "car")


def test_damerau_is_never_larger_than_levenshtein():
    for a, b in (("abcd", "badc"), ("hello", "hlelo"), ("string", "sitrng")):
        assert damerau(a, b) <= levenshtein(a, b)


# --- keyboard-weighted --------------------------------------------------------------------


def test_a_near_key_substitution_is_cheaper_than_a_far_one():
    """`a` and `s` are adjacent; `a` and `p` are not."""
    assert keyboard("cat", "cst") < keyboard("cat", "cpt")


def test_a_near_key_substitution_is_cheaper_than_plain_levenshtein_charges():
    assert keyboard("cat", "cst") < levenshtein("cat", "cst")


def test_identical_strings_cost_nothing():
    assert keyboard("hello", "hello") == 0.0


def test_a_far_substitution_still_costs_a_full_unit():
    assert keyboard("cat", "cpt") == pytest.approx(1.0)


# --- Jaro and Jaro-Winkler ----------------------------------------------------------------


def test_jaro_of_identical_strings_is_one():
    assert jaro("martha", "martha") == pytest.approx(1.0)


def test_jaro_of_disjoint_strings_is_zero():
    assert jaro("abc", "xyz") == pytest.approx(0.0)


def test_the_classic_jaro_value():
    """MARTHA against MARHTA is the example in every description of the measure."""
    assert jaro("martha", "marhta") == pytest.approx(0.944, abs=0.002)


def test_winkler_rewards_a_shared_prefix():
    assert jaro_winkler("martha", "marhta") > jaro("martha", "marhta")


def test_winkler_adds_nothing_without_a_shared_prefix():
    assert jaro_winkler("abcde", "xbcde") == pytest.approx(jaro("abcde", "xbcde"))


def test_winkler_never_exceeds_one():
    assert jaro_winkler("same", "same") == pytest.approx(1.0)


# --- phonetic -----------------------------------------------------------------------------


def test_soundex_matches_the_canonical_example():
    assert soundex("Robert") == "R163"


def test_soundex_groups_a_spelling_variant():
    assert soundex("Robert") == soundex("Rupert")


def test_soundex_is_four_characters():
    assert len(soundex("Washington")) == 4


def test_soundex_pads_a_short_name():
    assert soundex("Lee") == "L000"


def test_soundex_truncation_collides_longer_names():
    """It was designed to group surnames on index cards, and everything past the fourth
    code is discarded. Used as a general matcher that truncation is severe."""
    assert soundex("Robert") == soundex("Roberts")


def test_soundex_of_nothing_is_empty():
    assert soundex("") == ""


def test_metaphone_folds_ph_to_f():
    assert metaphone("phone") == metaphone("fone")


def test_metaphone_folds_ck_to_k():
    assert metaphone("back") == metaphone("bak")


def test_metaphone_drops_non_initial_vowels():
    assert metaphone("aeiou").startswith("a")


def test_metaphone_of_nothing_is_empty():
    assert metaphone("") == ""


# --- bigram overlap -----------------------------------------------------------------------


def test_bigram_of_identical_strings_is_one():
    assert bigram("hello", "hello") == pytest.approx(1.0)


def test_bigram_of_disjoint_strings_is_zero():
    assert bigram("abcd", "wxyz") == pytest.approx(0.0)


def test_bigram_forgives_reordering_that_edit_distance_does_not():
    """`abcd` and `cdab` are a block swap. They share `ab` and `cd` and differ only in the
    junction bigram, so overlap scores them 0.5 - while Levenshtein needs four edits out of
    four characters and scores them 0.0.

    That gap is the weakness of the overlap family, asserted rather than described: it is
    nearly blind to order, which is exactly what a transposition is.
    """
    assert bigram("abcd", "cdab") == pytest.approx(0.5)
    assert similarity("levenshtein", "abcd", "cdab") < bigram("abcd", "cdab")


# --- the dispatcher -----------------------------------------------------------------------


@pytest.mark.parametrize("measure", sorted(SIMILARITIES))
def test_every_measure_scores_a_string_against_itself_at_one(measure):
    assert similarity(measure, "example", "example") == pytest.approx(1.0)


@pytest.mark.parametrize("measure", sorted(SIMILARITIES))
def test_every_measure_returns_a_value_in_range(measure):
    value = similarity(measure, "example", "sample")
    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize("measure", sorted(SIMILARITIES))
def test_every_measure_handles_empty_input(measure):
    assert 0.0 <= similarity(measure, "", "word") <= 1.0


def test_an_unknown_measure_is_rejected():
    with pytest.raises(ValueError, match="unknown measure"):
        similarity("cosine", "a", "b")


def test_the_phonetic_measures_are_equivalence_classes_not_distances():
    """They return exactly 1 or 0, so they tie constantly. `run.py` therefore refuses to
    count a tie as a hit - otherwise they would be scored on the size of the class rather
    than on whether they found the word."""
    for measure in ("soundex", "metaphone"):
        assert similarity(measure, "phone", "fone") in (0.0, 1.0)
        assert similarity(measure, "phone", "zebra") in (0.0, 1.0)
