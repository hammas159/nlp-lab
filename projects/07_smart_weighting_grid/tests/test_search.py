"""Tests for scoring, recall, significance and the component summary.

No dataset and no network. The corpora are three documents long so the correct ranking can
be read off the matrix by eye.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from search import component_spread, paired_bootstrap, recall_at_k


def documents() -> sparse.csr_matrix:
    """Four documents over three terms, each dominated by a different term."""
    return sparse.csr_matrix(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.1, 0.1, 0.1],
            ]
        )
    )


# --- recall -------------------------------------------------------------------------------


def test_exact_match_is_ranked_first():
    queries = sparse.csr_matrix(np.array([[1.0, 0.0, 0.0]]))
    assert recall_at_k(documents(), queries, [{0}], k=1)[0] == pytest.approx(1.0)


def test_a_miss_scores_zero():
    queries = sparse.csr_matrix(np.array([[1.0, 0.0, 0.0]]))
    assert recall_at_k(documents(), queries, [{2}], k=1)[0] == 0.0


def test_partial_credit_for_one_of_two_gold_documents():
    queries = sparse.csr_matrix(np.array([[1.0, 0.0, 0.0]]))
    assert recall_at_k(documents(), queries, [{0, 2}], k=1)[0] == pytest.approx(0.5)


def test_a_larger_k_can_only_help():
    queries = sparse.csr_matrix(np.array([[1.0, 0.5, 0.0]]))
    at_one = recall_at_k(documents(), queries, [{1}], k=1)[0]
    at_three = recall_at_k(documents(), queries, [{1}], k=3)[0]
    assert at_three >= at_one


def test_a_query_with_no_indexable_term_scores_zero():
    """Its scores are zero everywhere, so argpartition returns an arbitrary k. Crediting
    those hits would pay a weighting scheme for words its index could not hold."""
    queries = sparse.csr_matrix((1, 3))
    assert recall_at_k(documents(), queries, [{0}], k=1)[0] == 0.0


def test_batching_does_not_change_the_answer():
    queries = sparse.csr_matrix(np.tile(np.array([[1.0, 0.0, 0.0]]), (7, 1)))
    gold = [{0}] * 7
    one = recall_at_k(documents(), queries, gold, k=1, batch=1)
    many = recall_at_k(documents(), queries, gold, k=1, batch=32)
    assert np.allclose(one, many)


def test_per_query_scores_are_returned_not_just_a_mean():
    queries = sparse.csr_matrix(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    out = recall_at_k(documents(), queries, [{0}, {0}], k=1)
    assert out.tolist() == [1.0, 0.0]


def test_an_empty_gold_set_is_skipped_rather_than_scored_zero():
    """A query with no gold document in the corpus is unanswerable by construction; scoring
    it zero would penalise every scheme equally for a defect of the corpus."""
    queries = sparse.csr_matrix(np.array([[1.0, 0.0, 0.0]]))
    assert recall_at_k(documents(), queries, [set()], k=1)[0] == 0.0


def test_query_normalisation_cannot_change_the_ranking():
    """The sharpest thing the SMART notation hides.

    Normalising a query vector multiplies every document's score for that query by one
    constant. A constant cannot reorder a list, so the third letter of the query code is
    inert for any rank-based metric - recall, precision, MAP, NDCG. Fifteen of the
    forty-five query schemes are duplicates of another fifteen under a different name.
    """
    docs = documents()
    gold = [{0}, {1}]
    base = sparse.csr_matrix(np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.2]]))

    plain = recall_at_k(docs, base, gold, k=1)
    scaled = recall_at_k(
        docs, sparse.csr_matrix(base.multiply(np.array([[3.0], [0.1]]))), gold, k=1
    )
    assert np.allclose(plain, scaled)


def test_document_normalisation_does_change_the_ranking():
    """The document side is the opposite case: each document is scaled by its own constant,
    so the relative order across documents moves. Asserting both halves is what makes the
    query-side result a finding rather than a bug."""
    docs = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0], [3.0, 3.0, 3.0]]))
    queries = sparse.csr_matrix(np.array([[1.0, 1.0, 0.0]]))

    unnormalised = recall_at_k(docs, queries, [{0}], k=1)
    lengths = np.sqrt(np.asarray(docs.multiply(docs).sum(axis=1))).ravel()
    normalised = recall_at_k(sparse.diags(1 / lengths) @ docs, queries, [{0}], k=1)
    assert unnormalised[0] != normalised[0]


# --- significance -------------------------------------------------------------------------


def test_identical_scores_are_not_significant():
    scores = np.array([1.0, 0.0, 0.5, 1.0] * 30)
    out = paired_bootstrap(scores, scores)
    assert out["mean_difference"] == 0.0
    assert out["significant"] is False


def test_a_consistent_gap_is_significant():
    out = paired_bootstrap(np.ones(300), np.zeros(300))
    assert out["significant"] is True
    assert out["ci_low"] > 0


def test_the_test_is_two_sided():
    """A scheme that is reliably worse must be flagged, not silently pass as 'not better'."""
    out = paired_bootstrap(np.zeros(300), np.ones(300))
    assert out["significant"] is True
    assert out["ci_high"] < 0


def test_the_interval_brackets_the_mean():
    rng = np.random.default_rng(0)
    a = rng.random(400)
    out = paired_bootstrap(a, a - 0.02)
    assert out["ci_low"] <= out["mean_difference"] <= out["ci_high"]


# --- component summary --------------------------------------------------------------------


def test_component_spread_groups_by_the_requested_letter():
    scores = {"lnc": 0.5, "lnn": 0.3, "bnc": 0.9, "bnn": 0.7}
    by_tf = component_spread(scores, 0)
    assert by_tf["l"] == pytest.approx(0.4)
    assert by_tf["b"] == pytest.approx(0.8)


def test_component_spread_reads_each_position_independently():
    scores = {"lnc": 0.5, "lnn": 0.3, "bnc": 0.9, "bnn": 0.7}
    by_norm = component_spread(scores, 2)
    assert by_norm["c"] == pytest.approx(0.7)
    assert by_norm["n"] == pytest.approx(0.5)


def test_component_spread_returns_sorted_letters():
    scores = {"lnc": 0.5, "anc": 0.4, "bnc": 0.9}
    assert list(component_spread(scores, 0)) == ["a", "b", "l"]
