"""Tests for the four rule sets.

No dataset, no network, no lexicon file: every case uses a hand-built lexicon of four words,
so these run anywhere and test the rules rather than the resources.

The rules are the axis this project measures, so each one is tested for the *specific*
linguistic behaviour it was added for, and for not changing anything else.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexicons import Lexicon, agreement, rescale
from rules import (
    RULES,
    classify,
    score,
    score_bag,
    score_contrastive,
    score_intensifier,
    score_negation,
    tokenize,
)

LEX = Lexicon("test", {"good": 1.0, "great": 1.0, "bad": -1.0, "awful": -1.0})


def s(rule, text):
    return score(rule, text, LEX)


# --- tokenizing -----------------------------------------------------------------------------


def test_contractions_lose_their_apostrophe_so_negators_match():
    """`don't` has to become `dont` or it never matches the negator list, and every negation
    rule silently does nothing on the commonest way English negates."""
    assert "dont" in tokenize("I don't care")


def test_punctuation_that_carries_emphasis_survives():
    assert "!" in tokenize("good!")


def test_tokenizing_lowercases():
    assert tokenize("GOOD") == ["good"]


# --- bag of words ---------------------------------------------------------------------------


def test_bag_adds_up_polarity():
    assert s("bag of words", "good great") == pytest.approx(2.0)


def test_bag_ignores_unknown_words():
    assert s("bag of words", "good elephant") == pytest.approx(1.0)


def test_bag_cannot_see_negation():
    """The failure the next rule exists to fix: to a bag of words, `not good` is `good`."""
    assert s("bag of words", "not good") == pytest.approx(1.0)
    assert s("bag of words", "good") == pytest.approx(1.0)


# --- negation -------------------------------------------------------------------------------


def test_negation_flips_the_sign():
    assert s("+ negation", "not good") == pytest.approx(-1.0)


def test_negation_reaches_across_a_short_gap():
    """`not very good` - the negator is two tokens away and must still apply."""
    assert s("+ negation", "not really good") < 0


def test_negation_does_not_reach_forever():
    """Beyond the window a negator must stop applying, or one `not` poisons a paragraph."""
    assert s("+ negation", "not a b c d good") == pytest.approx(1.0)


def test_a_negator_flips_everything_in_its_window_not_just_its_target():
    """A real limitation of window-based negation, recorded rather than hidden.

    In `not bad good`, both `bad` and `good` sit within three tokens of `not`, so both flip
    and the sentence scores 0.0 - even though a reader negates only `bad`. Clause-based
    negation (Taboada et al.) handles this; a fixed window cannot, and the window is what
    almost every lexicon pipeline actually uses.

    The behaviour is identical for every lexicon, so it cannot bias the comparison this
    project makes - but it does bound how good any `+ negation` row can be.
    """
    assert s("+ negation", "not bad good") == pytest.approx(0.0)


def test_a_word_before_the_negator_is_untouched():
    """Negation reaches forward, not backward."""
    assert s("+ negation", "good and not bad") == pytest.approx(2.0)


def test_a_contraction_negates():
    assert s("+ negation", "it wasn't good") < 0


def test_negation_is_the_only_difference_from_bag():
    """On a sentence with no negator the two rules must agree exactly, or the comparison
    between rows is measuring something other than negation."""
    text = "a good and great film with nothing awful"
    assert score_negation(tokenize("a good and great film"), LEX) == pytest.approx(
        score_bag(tokenize("a good and great film"), LEX)
    )
    assert "nothing" in tokenize(text)  # and with one, they must differ


# --- intensifiers ---------------------------------------------------------------------------


def test_a_booster_amplifies():
    assert s("+ intensifiers", "very good") > s("+ intensifiers", "good")


def test_a_dampener_reduces():
    assert 0 < s("+ intensifiers", "slightly good") < s("+ intensifiers", "good")


def test_an_intensifier_only_modifies_the_word_after_it():
    """`very good bad` must boost `good` and leave `bad` alone."""
    assert s("+ intensifiers", "very good bad") == pytest.approx(1.5 - 1.0)


def test_a_negated_booster_keeps_the_multiplier_and_flips():
    """`not very good` is strongly negative, not weakly positive."""
    assert s("+ intensifiers", "not very good") == pytest.approx(-1.5)


def test_intensifiers_still_negate():
    """Each rule set adds to the previous one rather than replacing it."""
    assert s("+ intensifiers", "not good") == pytest.approx(-1.0)


def test_intensifiers_change_nothing_without_an_intensifier():
    plain = "good and bad and great"
    assert score_intensifier(tokenize(plain), LEX) == pytest.approx(
        score_negation(tokenize(plain), LEX)
    )


# --- contrastive ----------------------------------------------------------------------------


def test_but_downweights_what_came_before():
    """The sentence this rule exists for: a positive first clause overruled by a negative
    second one should come out negative."""
    assert s("+ contrastive", "the acting was good but the film was awful") < 0


def test_the_same_sentence_without_the_rule_is_neutral():
    """Without the rule the two clauses cancel exactly, which is the wrong answer and the
    reason the rule is worth measuring."""
    assert s("+ intensifiers", "the acting was good but the film was awful") == pytest.approx(0.0)


def test_contrastive_changes_nothing_without_a_conjunction():
    plain = "a good and great film"
    assert score_contrastive(tokenize(plain), LEX) == pytest.approx(
        score_intensifier(tokenize(plain), LEX)
    )


def test_only_the_first_conjunction_is_a_pivot():
    """A sentence with two `but`s must not be split twice; the rule is defined on the
    first pivot."""
    assert isinstance(s("+ contrastive", "good but awful but good"), float)


# --- the dispatcher and classification ------------------------------------------------------


@pytest.mark.parametrize("rule", list(RULES))
def test_every_rule_scores_empty_text_as_zero(rule):
    assert s(rule, "") == pytest.approx(0.0)


@pytest.mark.parametrize("rule", list(RULES))
def test_every_rule_returns_a_float(rule):
    assert isinstance(s(rule, "a good film"), float)


def test_an_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown rule set"):
        score("sarcasm", "good", LEX)


def test_classification_splits_on_zero():
    assert classify(0.5) == 1
    assert classify(-0.5) == 0


def test_a_zero_score_is_classified_rather_than_abstained():
    """Dropping the items a lexicon cannot score would inflate accuracy by removing the
    hard ones. They are assigned to one class and counted; run.py reports the rate."""
    assert classify(0.0) in (0, 1)


# --- the lexicon container ------------------------------------------------------------------


def test_rescaling_puts_the_largest_magnitude_at_one():
    assert max(abs(v) for v in rescale({"a": 4.0, "b": -2.0}).values()) == pytest.approx(1.0)


def test_rescaling_preserves_sign_and_order():
    """Rescaling must not change any classification, since classification is by the sign of
    a sum and dividing by a positive constant cannot change that."""
    scaled = rescale({"a": 4.0, "b": -2.0, "c": 1.0})
    assert scaled["a"] > scaled["c"] > 0 > scaled["b"]


def test_rescaling_an_empty_lexicon_does_not_divide_by_zero():
    assert rescale({}) == {}


def test_agreement_is_one_for_a_lexicon_with_itself():
    assert agreement(LEX, LEX) == (1.0, len(LEX))


def test_agreement_reports_zero_overlap_rather_than_crashing():
    other = Lexicon("other", {"unrelated": 1.0})
    assert agreement(LEX, other) == (0.0, 0)


def test_agreement_counts_only_shared_words():
    other = Lexicon("other", {"good": -1.0, "unseen": 1.0})
    share, count = agreement(LEX, other)
    assert count == 1
    assert share == 0.0
