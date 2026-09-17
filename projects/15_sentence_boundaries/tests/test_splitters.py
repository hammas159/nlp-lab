"""Tests for the four splitters, the learned abbreviation list and the error taxonomy.

No dataset and no network. Every case is a sentence short enough to count boundaries in by
eye, because a splitter that is wrong in one construction still produces plausible output
everywhere else - which is exactly how two real bugs survived the first run of this project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from splitters import (
    COMMON_ABBREVIATIONS,
    abbreviation,
    classify_disagreement,
    conservative,
    learn_abbreviations,
    naive,
    trained,
)

# --- the end-of-text convention -----------------------------------------------------------


def test_the_end_of_the_text_is_not_a_boundary():
    """Regression test for the bug that cost fifteen points of precision.

    A sentence obviously ends at the end of the text, but the reference segmentation is a
    list of pieces and n pieces have n-1 joins. Emitting one anyway put a guaranteed false
    positive on every paragraph in the corpus.
    """
    assert naive("One sentence.") == []
    assert abbreviation("One sentence.") == []
    assert conservative("One sentence.") == []


def test_an_internal_boundary_is_still_found():
    assert len(naive("One sentence. Two sentences.")) == 1


def test_a_trailing_space_still_does_not_make_a_final_boundary():
    """`"One. "` has whitespace after the period, but nothing follows it."""
    assert naive("One sentence. ") == [13]


# --- naive --------------------------------------------------------------------------------


def test_naive_splits_on_each_terminator():
    assert len(naive("A. B! C? D")) == 3


def test_naive_needs_whitespace_after():
    """`3.5` is not two sentences, and the only thing stopping the naive splitter saying so
    is that no space follows the period."""
    assert naive("The value 3.5 is used here") == []


def test_naive_splits_on_a_known_abbreviation():
    """It has no list, so it must. This is the behaviour the next splitter exists to fix."""
    assert len(naive("Dr. Smith arrived late. He apologised")) == 2


# --- abbreviation list --------------------------------------------------------------------


def test_the_list_suppresses_a_known_abbreviation():
    assert len(abbreviation("Dr. Smith arrived late. He apologised")) == 1


def test_the_list_does_not_suppress_an_ordinary_word():
    assert len(abbreviation("He was late. He apologised")) == 1


def test_the_list_is_case_insensitive():
    assert abbreviation("DR. Smith arrived") == []


def test_an_abbreviation_not_on_the_list_still_splits():
    """The point of the list is that it is never complete."""
    assert len(abbreviation("The Rt. Hon. member spoke")) == 2


# --- learned abbreviations ----------------------------------------------------------------


def test_learning_finds_a_token_bound_to_its_period():
    texts = ["Dr. Smith saw Dr. Jones and Dr. Patel."] * 3
    learned = learn_abbreviations(texts, threshold=0.8, min_count=3)
    assert "dr" in learned


def test_learning_ignores_an_ordinary_word():
    texts = ["the cat sat on the mat and the dog ran."] * 5
    assert "the" not in learn_abbreviations(texts, threshold=0.8, min_count=3)


def test_learning_respects_the_minimum_count():
    """One occurrence is not evidence, however bound it looks."""
    assert "xyz" not in learn_abbreviations(["xyz. done"], threshold=0.5, min_count=4)


def test_a_learned_list_drives_the_trained_splitter():
    corpus = [
        "Prof. Yang spoke. The meeting was long and the meeting was useful.",
        "Prof. Yang left. The meeting continued and the meeting ended early.",
        "Prof. Yang returned. A meeting is a meeting after all of it.",
    ]
    learned = learn_abbreviations(corpus, threshold=0.8, min_count=3)
    assert "prof" in learned
    assert "meeting" not in learned
    assert len(trained("Prof. Yang spoke. He left", learned)) == 1


def test_learning_over_fits_a_small_corpus():
    """An honest property of the unsupervised approach, asserted rather than discovered
    later: on a corpus where an ordinary word happens to always end a sentence, that word
    is 'bound to its period' by the same evidence a real abbreviation is, and the splitter
    then refuses to split there.
    """
    degenerate = ["Prof. Yang spoke. Prof. Yang left. Prof. Yang."] * 3
    learned = learn_abbreviations(degenerate, threshold=0.8, min_count=3)
    assert "spoke" in learned
    assert trained("Prof. Yang spoke. He left", learned) == []


def test_learning_from_nothing_gives_an_empty_list():
    assert learn_abbreviations([]) == frozenset()


# --- conservative -------------------------------------------------------------------------


def test_conservative_requires_a_sentence_like_next_token():
    assert conservative("It cost 4. 50 dollars were left") != []


def test_conservative_rejects_a_lowercase_continuation():
    assert conservative("He is a Dr. of philosophy here") == []


def test_conservative_accepts_a_digit_start():
    assert len(conservative("That was 1999. 2000 was better")) == 1


def test_conservative_accepts_a_quote_start():
    assert len(conservative('He spoke. "Hello," she said')) == 1


# --- the taxonomy -------------------------------------------------------------------------


def test_end_of_text_is_named_rather_than_misfiled():
    """The second bug: `following[:1] in "\\"'(["` is True for the empty string, because the
    empty string is a substring of everything. Thousands of end-of-paragraph positions were
    filed under 'quote follows'."""
    text = "A sentence."
    assert classify_disagreement(text, len(text)) == "end of text"


def test_a_single_initial_is_named():
    text = "J. R. R. Tolkien wrote"
    assert classify_disagreement(text, 2) == "single initial"


def test_a_known_abbreviation_is_named():
    text = "Dr. Smith arrived"
    assert classify_disagreement(text, 3) == "known abbreviation"


def test_a_lowercase_continuation_is_named():
    text = "ended. and then"
    assert classify_disagreement(text, 6) == "lowercase follows"


def test_a_quote_continuation_is_named():
    text = 'ended. "Hello"'
    assert classify_disagreement(text, 6) == "quote or bracket follows"


def test_a_question_mark_is_named():
    text = "Really? Yes"
    assert classify_disagreement(text, 7) == "exclamation or question"


def test_an_unremarkable_boundary_falls_through_to_other():
    text = "ended. Next"
    assert classify_disagreement(text, 6) == "other"


# --- ordering between the splitters -------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith arrived late. He apologised",
        "It was 1999. Things changed",
        'He spoke. "Hello," she said',
    ],
)
def test_conservative_never_predicts_more_than_naive(text):
    """Each splitter is the previous one plus a restriction, so the counts must be ordered.
    If they were not, one of them would be adding boundaries rather than removing them."""
    assert len(conservative(text)) <= len(abbreviation(text)) <= len(naive(text))


def test_the_supplied_list_is_not_empty():
    assert len(COMMON_ABBREVIATIONS) > 10
