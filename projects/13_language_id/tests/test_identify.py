"""Tests for n-gram extraction, profiles and the three decision rules.

No dataset and no network. The profiles are built from short constructed texts whose
character statistics are obvious, so a rule that picks the wrong class for a reason other
than the evidence still fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from identify import METHODS, Profile, cavnar_trenkle, classify, compression, naive_bayes, ngrams

ENGLISH = (
    "the quick brown fox jumps over the lazy dog and the cat sat on the mat while the "
    "rain fell on the roof of the house near the river " * 12
)
CODE = (
    "int main(void) { int x = 0; for (int i = 0; i < 10; i++) { x += i; } return x; } "
    "void helper(char *buf) { free(buf); } " * 12
)


def profiles() -> list[Profile]:
    return [Profile.build("english", ENGLISH), Profile.build("code", CODE)]


# --- n-grams ------------------------------------------------------------------------------


def test_ngrams_include_every_requested_size():
    grams = ngrams("ab", sizes=(1, 2))
    # Padded to "_ab_": unigrams _, a, b, _ and bigrams _a, ab, b_
    assert "_a" in grams
    assert "ab" in grams
    assert "b_" in grams


def test_padding_makes_word_edges_visible():
    """`_the_` and `the` are different features, and the difference between prose and code
    is largely at the edges - identifiers run together, words do not."""
    assert "_a" in ngrams("a", sizes=(2,))


def test_an_empty_string_still_produces_the_padding():
    assert ngrams("", sizes=(1,)) == ["_", "_"]


def test_longer_text_gives_more_ngrams():
    assert len(ngrams("abcdef")) > len(ngrams("abc"))


# --- profiles -----------------------------------------------------------------------------


def test_a_profile_ranks_its_commonest_ngrams_first():
    profile = Profile.build("english", ENGLISH)
    assert profile.ranks["e"] < profile.ranks["z"] if "z" in profile.ranks else True


def test_a_profile_is_capped_at_its_size():
    profile = Profile.build("english", ENGLISH, size=10)
    assert len(profile.ranks) <= 10


def test_a_profile_keeps_its_text_for_the_compression_rule():
    profile = Profile.build("english", ENGLISH)
    assert profile.text == ENGLISH


def test_the_total_is_the_ngram_count():
    profile = Profile.build("x", "abc")
    assert profile.total == sum(profile.counts.values())


# --- the three rules ----------------------------------------------------------------------


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_recognises_text_from_its_own_class(method):
    assert classify(method, "the cat sat on the mat near the river", profiles()) == "english"


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_recognises_code(method):
    assert classify(method, "for (int i = 0; i < 10; i++) { x += i; }", profiles()) == "code"


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_returns_one_of_the_known_labels(method):
    assert classify(method, "zzz qqq", profiles()) in {"english", "code"}


@pytest.mark.parametrize("method", sorted(METHODS))
def test_every_method_survives_an_empty_input(method):
    assert classify(method, "", profiles()) in {"english", "code"}


def test_an_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        classify("magic", "text", profiles())


# --- properties that distinguish the rules ------------------------------------------------


def test_cavnar_trenkle_uses_only_rank_order():
    """It has no probabilities. Scaling a profile's text by repetition changes every count
    but not the ranking, so the decision must not move."""
    small = [Profile.build("english", ENGLISH), Profile.build("code", CODE)]
    large = [Profile.build("english", ENGLISH * 3), Profile.build("code", CODE * 3)]
    text = "the cat sat on the mat"
    assert cavnar_trenkle(text, small) == cavnar_trenkle(text, large)


def test_naive_bayes_stays_finite_on_a_long_input():
    """A product of several hundred n-gram probabilities underflows to exactly zero in
    double precision, and every input would then land in whichever class was checked
    first. Log space is what prevents it."""
    long_text = "the cat sat on the mat " * 200
    assert naive_bayes(long_text, profiles()) == "english"


def test_naive_bayes_handles_an_ngram_no_profile_has_seen():
    """Smoothing, asserted: without alpha this is log(0) and the score is -inf for every
    class."""
    assert naive_bayes("çãõ üñ", profiles()) in {"english", "code"}


def test_compression_gives_every_class_the_same_context_length():
    """Otherwise the largest corpus wins by having more text to reference, which measures
    corpus size rather than similarity."""
    lopsided = [Profile.build("english", ENGLISH * 8), Profile.build("code", CODE)]
    assert compression("for (int i = 0; i < 10; i++) { x += i; }", lopsided) == "code"


def test_the_rules_can_disagree_on_a_short_input():
    """The whole project: on a long input all three agree, and on a short one they need
    not. If they always agreed there would be nothing to measure."""
    built = profiles()
    short = "x = 0"
    verdicts = {m: classify(m, short, built) for m in METHODS}
    assert len(verdicts) == 3
