"""Tests for weighted BM25 scoring, the relevance model, and query expansion.

No dataset and no network. The corpora are a handful of short documents so the expected
ranking can be read off by eye, which is the only way to catch an expansion that is
plausible but weighted wrongly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.bm25 import BM25

from prf import expand, recall_at_k, relevance_model, score_weighted

DOCS = [
    "quantum entanglement physics experiment",
    "quantum computing qubit algorithm",
    "baking bread sourdough starter flour",
    "baking cake sugar butter flour",
    "football league season match goal",
]


def fitted() -> BM25:
    return BM25().fit(DOCS)


# --- weighted scoring ---------------------------------------------------------------------


def test_uniform_weights_reproduce_plain_bm25():
    """A weight of 1 on every query term is exactly what `BM25.score` already does. If the
    two disagreed, every expanded result would be incomparable with its baseline."""
    bm25 = fitted()
    plain = bm25.score("quantum algorithm")
    weighted = score_weighted(bm25, {"quantum": 1.0, "algorithm": 1.0})
    assert np.allclose(plain, weighted, atol=1e-5)


def test_weights_scale_a_term_contribution():
    bm25 = fitted()
    light = score_weighted(bm25, {"sourdough": 0.1})
    heavy = score_weighted(bm25, {"sourdough": 1.0})
    assert np.allclose(heavy, light * 10, atol=1e-4)


def test_a_zero_weight_contributes_nothing():
    bm25 = fitted()
    with_zero = score_weighted(bm25, {"quantum": 1.0, "sourdough": 0.0})
    without = score_weighted(bm25, {"quantum": 1.0})
    assert np.allclose(with_zero, without)


def test_unknown_terms_are_ignored():
    bm25 = fitted()
    assert np.allclose(
        score_weighted(bm25, {"quantum": 1.0, "zzzz": 5.0}),
        score_weighted(bm25, {"quantum": 1.0}),
    )


def test_an_empty_query_scores_nothing():
    assert np.allclose(score_weighted(fitted(), {}), 0.0)


# --- the relevance model ------------------------------------------------------------------


def test_the_model_harvests_terms_from_the_top_documents():
    bm25 = fitted()
    scores = bm25.score("sourdough")
    model = relevance_model(bm25, DOCS, scores, n_feedback=1, n_terms=10)
    assert "sourdough" in model
    assert "starter" in model
    assert "football" not in model


def test_more_feedback_documents_widen_the_model():
    bm25 = fitted()
    scores = bm25.score("baking")
    narrow = relevance_model(bm25, DOCS, scores, n_feedback=1, n_terms=20)
    wide = relevance_model(bm25, DOCS, scores, n_feedback=2, n_terms=20)
    assert len(wide) > len(narrow)


def test_the_term_budget_is_respected():
    bm25 = fitted()
    model = relevance_model(bm25, DOCS, bm25.score("baking flour"), n_feedback=4, n_terms=3)
    assert len(model) == 3


def test_feedback_documents_are_weighted_by_their_score():
    """A document the first stage was confident about should contribute more. If every
    feedback document counted equally, the tenth result would pull the query as hard as the
    first - which is the failure mode the interpolation is supposed to limit."""
    bm25 = fitted()
    scores = bm25.score("sourdough")
    model = relevance_model(bm25, DOCS, scores, n_feedback=2, n_terms=20)
    assert model["sourdough"] > model.get("cake", 0.0)


def test_a_query_matching_nothing_gives_an_empty_model():
    """All scores zero means there is no evidence to build a model from, and inventing one
    from an arbitrary top-k would be feedback from documents chosen by tie-breaking."""
    bm25 = fitted()
    assert relevance_model(bm25, DOCS, np.zeros(len(DOCS)), n_feedback=3) == {}


# --- expansion ----------------------------------------------------------------------------


def test_alpha_of_zero_leaves_the_query_alone():
    weights = expand("quantum algorithm", {"qubit": 1.0}, alpha=0.0)
    assert weights.get("qubit", 0.0) == 0.0
    assert weights["quantum"] > 0


def test_alpha_of_one_replaces_the_query():
    weights = expand("quantum algorithm", {"qubit": 1.0}, alpha=1.0)
    assert weights.get("quantum", 0.0) == 0.0
    assert weights["qubit"] == pytest.approx(1.0)


def test_both_sides_are_normalised_before_interpolation():
    """So alpha means the same thing whatever the query length or how peaked the model is.
    Without this, a two-word query and a ten-word query get different amounts of expansion
    from the same alpha."""
    short = expand("quantum", {"qubit": 1.0, "algorithm": 1.0}, alpha=0.5)
    long = expand(
        "quantum computing physics experiment", {"qubit": 1.0, "algorithm": 1.0}, alpha=0.5
    )
    assert sum(v for k, v in short.items() if k in {"qubit", "algorithm"}) == pytest.approx(0.5)
    assert sum(v for k, v in long.items() if k in {"qubit", "algorithm"}) == pytest.approx(0.5)


def test_the_original_terms_keep_the_rest_of_the_mass():
    weights = expand("quantum computing", {"qubit": 1.0}, alpha=0.3)
    original = weights["quantum"] + weights["computing"]
    assert original == pytest.approx(0.7)


def test_a_term_in_both_query_and_model_accumulates():
    """It should be weighted more, not overwritten by whichever was applied last."""
    weights = expand("quantum", {"quantum": 1.0}, alpha=0.5)
    assert weights["quantum"] == pytest.approx(1.0)


def test_an_empty_model_leaves_the_query_intact():
    weights = expand("quantum computing", {}, alpha=0.5)
    assert weights["quantum"] == pytest.approx(0.25)


# --- recall -------------------------------------------------------------------------------


def test_recall_finds_the_gold_document():
    scores = np.array([0.1, 0.9, 0.2])
    assert recall_at_k(scores, {1}, k=1) == pytest.approx(1.0)


def test_recall_is_partial_for_one_of_two():
    scores = np.array([0.1, 0.9, 0.2])
    assert recall_at_k(scores, {1, 2}, k=1) == pytest.approx(0.5)


def test_recall_with_no_gold_is_zero():
    assert recall_at_k(np.array([0.5, 0.1]), set(), k=1) == 0.0


def test_expansion_can_move_the_ranking():
    """The whole point of the project: adding terms changes which documents come back, and
    it can change them for the worse."""
    bm25 = fitted()
    before = np.argsort(-bm25.score("flour"))[0]
    after = np.argsort(-score_weighted(bm25, expand("flour", {"football": 5.0}, alpha=0.9)))[0]
    assert before != after
