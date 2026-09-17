"""Tests for the two learned baselines.

No dataset and no network. Both classifiers are given a task with an obvious answer, so a
baseline that has quietly degenerated into predicting one class still fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from baselines import multinomial_nb, nearest_centroid, tokenize

SEPARABLE_TEXTS = [
    "malloc free buffer overflow strcpy",
    "malloc strcpy buffer unchecked length",
    "printf hello world return zero",
    "printf format string return value",
]
SEPARABLE_LABELS = np.array([True, True, False, False])


# --- tokenizer ----------------------------------------------------------------------------


def test_identifiers_survive_as_one_token():
    assert tokenize("int max_length = 42;") == ["int", "max_length", "=", "42", ";"]


def test_punctuation_is_kept():
    """Braces and semicolons carry control flow in C. Dropping them would make two
    functions with different structure look alike - the same reasoning as project 04."""
    assert tokenize("if (x) { y++; }") == ["if", "(", "x", ")", "{", "y", "+", "+", ";", "}"]


def test_case_is_preserved():
    assert tokenize("MyStruct myVar") == ["MyStruct", "myVar"]


def test_empty_text_gives_no_tokens():
    assert tokenize("") == []


# --- Naive Bayes --------------------------------------------------------------------------


def test_naive_bayes_separates_an_obvious_task():
    predictions = multinomial_nb(SEPARABLE_TEXTS, SEPARABLE_LABELS, SEPARABLE_TEXTS)
    assert predictions.tolist() == SEPARABLE_LABELS.tolist()


def test_naive_bayes_generalises_to_unseen_documents():
    test = ["malloc strcpy buffer", "printf return zero"]
    assert multinomial_nb(SEPARABLE_TEXTS, SEPARABLE_LABELS, test).tolist() == [True, False]


def test_naive_bayes_returns_one_prediction_per_document():
    out = multinomial_nb(SEPARABLE_TEXTS, SEPARABLE_LABELS, ["anything", "else", "here"])
    assert len(out) == 3


def test_naive_bayes_survives_a_document_of_unknown_words():
    """Every token out of vocabulary means the prior decides. It must return a label rather
    than a NaN."""
    out = multinomial_nb(SEPARABLE_TEXTS, SEPARABLE_LABELS, ["zzz qqq wwww"])
    assert out.tolist()[0] in (True, False)


def test_naive_bayes_handles_an_empty_vocabulary():
    """min_count is 2, so a training set with no repeated token leaves nothing to fit."""
    out = multinomial_nb(["alpha", "beta"], np.array([True, False]), ["gamma"])
    assert len(out) == 1


def test_log_space_survives_a_long_document():
    """A product of several thousand token probabilities underflows to exactly zero in
    double precision, and the classifier then returns the same class for everything. Doing
    it in log space is what stops that, so a long document is asserted here."""
    long_positive = " ".join(["malloc strcpy buffer overflow"] * 500)
    long_negative = " ".join(["printf return value zero"] * 500)
    out = multinomial_nb(SEPARABLE_TEXTS, SEPARABLE_LABELS, [long_positive, long_negative])
    assert out.tolist() == [True, False]


# --- nearest centroid ---------------------------------------------------------------------


def test_nearest_centroid_separates_an_obvious_task():
    predictions = nearest_centroid(SEPARABLE_TEXTS, SEPARABLE_LABELS, SEPARABLE_TEXTS)
    assert predictions.tolist() == SEPARABLE_LABELS.tolist()


def test_nearest_centroid_generalises_to_unseen_documents():
    test = ["buffer overflow strcpy malloc", "format string printf return"]
    assert nearest_centroid(SEPARABLE_TEXTS, SEPARABLE_LABELS, test).tolist() == [True, False]


def test_nearest_centroid_does_not_divide_by_zero_on_an_empty_document():
    out = nearest_centroid(SEPARABLE_TEXTS, SEPARABLE_LABELS, [""])
    assert len(out) == 1
    assert out.tolist()[0] in (True, False)


def test_both_baselines_beat_a_constant_prediction_here():
    """If either collapsed to one class it would score 0.5 on this balanced task, which is
    what a degenerate classifier looks like from the outside."""
    for fn in (multinomial_nb, nearest_centroid):
        predictions = fn(SEPARABLE_TEXTS, SEPARABLE_LABELS, SEPARABLE_TEXTS)
        assert len(set(predictions.tolist())) == 2
