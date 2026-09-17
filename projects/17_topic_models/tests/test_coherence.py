"""Tests for NPMI coherence.

No dataset and no network. Coherence is the number this project exists to distrust, so the
tests pin its *properties* - bounds, direction, and the two conventions that quietly change
it - rather than checking one hard-coded value.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from coherence import (
    document_frequencies,
    model_coherence,
    npmi_pair,
    topic_coherence,
)

VOCAB = {"cat": 0, "dog": 1, "quantum": 2, "spoon": 3}


def corpus():
    """`cat` and `dog` always co-occur; `quantum` never appears with either."""
    return [
        ["cat", "dog", "spoon"],
        ["cat", "dog"],
        ["cat", "dog", "cat"],
        ["quantum"],
        ["quantum", "spoon"],
    ]


def inputs():
    counts, presence = document_frequencies(corpus(), VOCAB)
    return VOCAB, counts, presence


# --- document frequencies -----------------------------------------------------------------


def test_frequencies_are_document_counts_not_token_counts():
    """`cat` appears four times but in three documents. Coherence asks about documents."""
    counts, _ = document_frequencies(corpus(), VOCAB)
    assert counts[VOCAB["cat"]] == 3.0


def test_presence_is_boolean():
    _, presence = document_frequencies(corpus(), VOCAB)
    assert presence.dtype == bool


def test_tokens_outside_the_vocabulary_are_ignored():
    counts, _ = document_frequencies([["cat", "zzz"]], VOCAB)
    assert counts.sum() == 1.0


# --- the NPMI statistic -------------------------------------------------------------------


def test_npmi_is_one_for_words_that_always_co_occur():
    """The upper bound, and the definition's sanity check. Two words in the same 2 of 5
    documents and nowhere else."""
    assert npmi_pair(2, 2, 2, 5) == pytest.approx(1.0, abs=1e-6)


def test_a_word_pair_in_every_document_scores_zero_rather_than_one():
    """A genuine 0/0. If both words are in all n documents, P(a,b) = P(a)P(b) = 1, so the
    ratio is 1 and its logarithm is 0 - but so is the denominator.

    The boundary matters because it is exactly the stopword case: function words appear
    everywhere, and a metric that scored them 1.0 would rank a topic of pure stopwords as
    perfectly coherent. Falling to 0 instead means "no information", which is correct.
    """
    assert npmi_pair(5, 5, 5, 5) == pytest.approx(0.0, abs=1e-3)


def test_npmi_is_minus_one_for_words_that_never_co_occur():
    assert npmi_pair(3, 2, 0, 5) == -1.0


def test_npmi_is_near_zero_for_independent_words():
    """Two words in half the documents each, co-occurring in a quarter, are independent."""
    assert npmi_pair(50, 50, 25, 100) == pytest.approx(0.0, abs=1e-6)


def test_npmi_stays_within_bounds():
    for a, b, ab in [(3, 2, 1), (5, 5, 5), (1, 1, 1), (9, 1, 1)]:
        assert -1.0 <= npmi_pair(a, b, ab, 10) <= 1.0


def test_a_missing_pair_scores_the_floor_rather_than_being_dropped():
    """Dropping undefined pairs would raise a bad topic's average by deleting its worst
    evidence. The floor keeps it honest."""
    vocabulary, counts, presence = inputs()
    never = topic_coherence(["quantum", "cat"], vocabulary, counts, presence)
    assert never == -1.0


# --- topic and model coherence ------------------------------------------------------------


def test_a_coherent_topic_beats_an_incoherent_one():
    """The one property the metric must have, or nothing else about it matters."""
    vocabulary, counts, presence = inputs()
    good = topic_coherence(["cat", "dog"], vocabulary, counts, presence)
    bad = topic_coherence(["quantum", "dog"], vocabulary, counts, presence)
    assert good > bad


def test_a_topic_with_one_known_word_is_not_scored():
    vocabulary, counts, presence = inputs()
    assert topic_coherence(["cat"], vocabulary, counts, presence) == 0.0
    assert topic_coherence(["zzz", "yyy"], vocabulary, counts, presence) == 0.0


def test_model_coherence_is_the_mean_over_topics():
    vocabulary, counts, presence = inputs()
    topics = [["cat", "dog"], ["quantum", "dog"]]
    expected = np.mean([topic_coherence(t, vocabulary, counts, presence) for t in topics])
    assert model_coherence(topics, vocabulary, counts, presence) == pytest.approx(expected)


def test_a_model_with_no_topics_scores_zero():
    vocabulary, counts, presence = inputs()
    assert model_coherence([], vocabulary, counts, presence) == 0.0


def test_word_order_within_a_topic_does_not_matter():
    """Coherence is over unordered pairs; a model that shuffled its top words would
    otherwise score differently for the same topic."""
    vocabulary, counts, presence = inputs()
    a = topic_coherence(["cat", "dog", "spoon"], vocabulary, counts, presence)
    b = topic_coherence(["spoon", "dog", "cat"], vocabulary, counts, presence)
    assert a == pytest.approx(b)
