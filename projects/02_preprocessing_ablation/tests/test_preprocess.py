"""Tests for the stemmer, the stopword list and the paired bootstrap.

No dataset and no network. The Porter cases are taken from the behaviour the 1980 paper
specifies, so a regression in the stemmer shows up as a failing assertion rather than as
a quietly different number in the results table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from preprocess import STOPWORDS, VARIANTS, make_tokenizer, porter_stem
from significance import paired_bootstrap

# --- Porter stemmer ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "word,expected",
    [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("cats", "cat"),
        ("feed", "feed"),
        ("agreed", "agre"),
        ("plastered", "plaster"),
        ("motoring", "motor"),
        ("sing", "sing"),
        ("hopping", "hop"),
        ("falling", "fall"),
        ("happy", "happi"),
        ("sky", "sky"),
    ],
)
def test_porter_matches_the_published_examples(word, expected):
    assert porter_stem(word) == expected


def test_stemmer_leaves_very_short_words_alone():
    for w in ("a", "an", "is", "be"):
        assert porter_stem(w) == w


def test_stemmer_is_idempotent_on_its_own_output():
    """Stemming a stem must not keep eroding it, or the index depends on how many times
    the pipeline ran."""
    for w in ("nationalization", "relational", "conditional", "running"):
        once = porter_stem(w)
        assert porter_stem(once) == once


def test_stemmer_collapses_an_inflected_family():
    stems = {porter_stem(w) for w in ("connect", "connected", "connecting", "connection")}
    assert len(stems) == 1


def test_stemmer_can_produce_short_stems():
    """The mechanism behind the headline: stemming creates tokens a length filter then
    deletes. If this stops being true, the interaction finding no longer holds."""
    assert len(porter_stem("aging")) <= 3


# --- stopwords ---------------------------------------------------------------------------


def test_stopword_list_contains_the_obvious_ones():
    assert {"the", "a", "of", "and", "is"} <= STOPWORDS


def test_stopword_list_also_removes_negations():
    """Worth knowing rather than assuming: the standard list drops `no` and `not`,
    which changes the meaning of a query."""
    assert "not" in STOPWORDS
    assert "no" in STOPWORDS


# --- tokenizer composition -----------------------------------------------------------------


def test_baseline_lowercases():
    assert make_tokenizer()("The CAT") == ["the", "cat"]


def test_case_sensitive_variant_keeps_case():
    assert make_tokenizer(lowercase=False)("The CAT") == ["The", "CAT"]


def test_stopword_removal_removes_them():
    assert make_tokenizer(remove_stopwords=True)("the cat and the hat") == ["cat", "hat"]


def test_stemming_applies_to_every_token():
    assert make_tokenizer(stem=True)("running cats") == ["run", "cat"]


def test_min_length_filters_after_stemming():
    """Order matters: filtering before stemming would remove different tokens."""
    out = make_tokenizer(stem=True, min_length=3)("aging is running")
    assert all(len(t) >= 3 for t in out)


def test_every_variant_produces_tokens():
    for label, kwargs in VARIANTS.items():
        out = make_tokenizer(**kwargs)("The running cats were connected to the machine")
        assert out, f"{label} produced no tokens"


def test_variants_are_distinct_configurations():
    seen = {tuple(sorted(v.items())) for v in VARIANTS.values()}
    assert len(seen) == len(VARIANTS), "two variants have identical settings"


# --- the paired bootstrap ---------------------------------------------------------------


def test_identical_scores_are_not_significant():
    a = np.array([1.0, 0.5, 0.0, 1.0] * 25)
    assert paired_bootstrap(a, a.copy())["significant"] is False


def test_a_large_consistent_gap_is_significant():
    rng = np.random.default_rng(1)
    base = rng.random(300)
    better = np.clip(base + 0.4, 0, 1)
    res = paired_bootstrap(better, base)
    assert res["significant"] is True
    assert res["mean_difference"] > 0


def test_a_tiny_noisy_gap_is_not_significant():
    """The point of the test: a small mean difference on a noisy sample must not be
    reported as a result."""
    rng = np.random.default_rng(2)
    base = rng.random(300)
    barely = base + rng.normal(0, 0.5, 300) + 0.005
    assert paired_bootstrap(barely, base)["significant"] is False


def test_confidence_interval_brackets_the_mean_difference():
    rng = np.random.default_rng(3)
    a, b = rng.random(200), rng.random(200)
    res = paired_bootstrap(a, b)
    assert res["ci_low"] <= res["mean_difference"] <= res["ci_high"]
