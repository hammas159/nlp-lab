"""Tests for the ceiling logic and the metrics.

No model and no network: the reranker is never loaded here. What is under test is the
arithmetic of the ceiling argument, which is where the project's claim lives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rerank import DEPTH, KS, Random, metrics

# --- metrics ------------------------------------------------------------------------------


def test_recall_counts_gold_in_the_top_k():
    order = np.array([5, 9, 1, 7, 3])
    out = metrics(order, {5, 3})
    assert out["recall@1"] == 0.5
    assert out["recall@5"] == 1.0


def test_mrr_uses_the_first_gold_hit():
    assert metrics(np.array([9, 8, 4]), {4})["mrr"] == 1 / 3


def test_metrics_include_the_depth_ceiling():
    out = metrics(np.arange(100), {0})
    assert f"recall@{DEPTH}" in out


def test_recall_is_monotonic():
    out = metrics(np.arange(100), {0, 60})
    values = [out[f"recall@{k}"] for k in KS] + [out[f"recall@{DEPTH}"]]
    assert values == sorted(values)


# --- the ceiling argument -------------------------------------------------------------------


def rerank(candidates: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """The same reordering the experiment performs, with the model's scores supplied."""
    return candidates[np.argsort(-scores)]


def test_a_perfect_reranker_cannot_exceed_the_first_stage_ceiling():
    """The central claim. Gold sits at position 60; the reranker only sees the top 50, so
    no amount of reranking quality can recover it."""
    order = np.arange(100)
    gold = {60}
    before = metrics(order, gold)
    candidates = order[:DEPTH]
    # An oracle reranker: perfect scores, gold first if present.
    scores = np.array([1.0 if i in gold else 0.0 for i in candidates])
    after = metrics(rerank(candidates, scores), gold)
    assert before[f"recall@{DEPTH}"] == 0.0
    assert after["recall@10"] == 0.0


def test_a_perfect_reranker_reaches_the_ceiling_when_gold_is_inside_it():
    order = np.arange(100)
    gold = {40}  # inside the top 50, but far outside the top 10
    assert metrics(order, gold)["recall@10"] == 0.0
    candidates = order[:DEPTH]
    scores = np.array([1.0 if i in gold else 0.0 for i in candidates])
    assert metrics(rerank(candidates, scores), gold)["recall@10"] == 1.0


def test_reranking_cannot_introduce_a_document_the_first_stage_missed():
    order = np.arange(DEPTH)
    reranked = rerank(order, np.random.default_rng(0).random(DEPTH))
    assert set(reranked) == set(order)


def test_conversion_is_bounded_by_one():
    """after_recall@10 / ceiling can never exceed 1 - a value above it means a bug."""
    order = np.arange(100)
    gold = {5, 40}
    ceiling = metrics(order, gold)[f"recall@{DEPTH}"]
    candidates = order[:DEPTH]
    scores = np.array([1.0 if i in gold else 0.0 for i in candidates])
    after = metrics(rerank(candidates, scores), gold)["recall@10"]
    assert after / ceiling <= 1.0


# --- the control --------------------------------------------------------------------------


def test_random_first_stage_is_deterministic_across_fits():
    """The control has to be reproducible or its ceiling is not a fixed number."""
    a, b = Random().fit(["x"] * 50), Random().fit(["x"] * 50)
    assert np.allclose(a.score("q"), b.score("q"))


def test_random_first_stage_returns_one_score_per_document():
    assert Random().fit(["x"] * 37).score("q").shape == (37,)


def test_random_first_stage_has_a_low_ceiling():
    """With 3,000 documents and 2 gold, a random top-50 should almost never contain gold.
    This is why it is the control: if the reranker rescued it, the first stage would not
    matter at all."""
    rng = np.random.default_rng(0)
    n_docs, trials, hits = 3000, 400, 0
    for _ in range(trials):
        gold = set(rng.choice(n_docs, 2, replace=False).tolist())
        order = rng.permutation(n_docs)
        hits += metrics(order, gold)[f"recall@{DEPTH}"]
    assert hits / trials < 0.1
