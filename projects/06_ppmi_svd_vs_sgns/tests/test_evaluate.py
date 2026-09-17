"""Tests for document pooling, recall@k, neighbour agreement and the paired bootstrap.

No dataset and no network. The retrieval tests build a tiny corpus whose right answer is
obvious, so a scorer that returns a plausible number for the wrong reason still fails.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluate import (
    Pooler,
    embed_documents,
    neighbour_agreement,
    paired_bootstrap,
    recall_at_k,
)
from factorize import Embedding


@dataclass(frozen=True)
class FakeQuery:
    question: str
    gold_titles: tuple[str, ...]


def toy_embedding() -> Embedding:
    """Three orthogonal axes, so 'cat'-ish and 'ship'-ish documents cannot be confused."""
    return Embedding(
        ["cat", "kitten", "ship", "boat", "filler"],
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


# --- pooling ------------------------------------------------------------------------------


def test_documents_are_unit_length():
    matrix = embed_documents(toy_embedding(), ["cat kitten", "ship boat"])
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_unknown_words_are_skipped_not_zeroed():
    """A document of one known word plus noise must embed as that word, not be dragged
    towards the origin by tokens the space has never seen."""
    embedding = toy_embedding()
    matrix = embed_documents(embedding, ["cat zzz qqq"])
    assert np.allclose(matrix[0], embedding["cat"] / np.linalg.norm(embedding["cat"]))


def test_a_document_with_no_known_words_is_the_zero_vector():
    matrix = embed_documents(toy_embedding(), ["zzz qqq"])
    assert np.allclose(matrix[0], 0.0)


# --- the pooler ---------------------------------------------------------------------------


def test_pooler_reproduces_the_naive_pooling():
    """The sparse matmul exists purely for speed. If it disagreed with the obvious
    implementation it would be a bug disguised as an optimisation."""
    embedding = toy_embedding()
    docs = ["cat kitten cat", "ship boat", "zzz", "filler cat"]
    pooler = Pooler(embedding.words, docs)

    naive = np.zeros((len(docs), embedding.dim))
    for i, doc in enumerate(docs):
        vectors = [embedding[t] for t in doc.split() if t in embedding]
        if vectors:
            naive[i] = np.mean(vectors, axis=0)
    norms = np.linalg.norm(naive, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    assert np.allclose(pooler.pool(embedding.vectors), naive / norms)


def test_pooler_counts_repeated_words_once_per_occurrence():
    """Mean pooling weights by term frequency, so a document saying 'cat' twice must lean
    towards 'cat' twice as hard - a set-based count would quietly change the metric."""
    embedding = toy_embedding()
    pooled = Pooler(embedding.words, ["cat cat ship"]).pool(embedding.vectors)
    expected = (2 * embedding["cat"] + embedding["ship"]) / 3
    assert np.allclose(pooled[0], expected / np.linalg.norm(expected))


def test_pooler_flags_texts_with_no_known_words():
    pooler = Pooler(toy_embedding().words, ["cat", "zzz qqq", "ship"])
    assert pooler.empty.tolist() == [False, True, False]


def test_pooler_is_reusable_across_embeddings():
    """The whole point: one tokenizing pass serves every embedding in the ladder."""
    words = toy_embedding().words
    pooler = Pooler(words, ["cat ship"])
    first = pooler.pool(np.eye(5)[:, :3])
    second = pooler.pool(np.ones((5, 3)))
    assert first.shape == second.shape == (1, 3)
    assert not np.allclose(first, second)


# --- recall -------------------------------------------------------------------------------


def _toy_corpus():
    doc_ids = ["cats", "ships", "noise1", "noise2", "noise3"]
    docs = ["cat kitten cat", "ship boat ship", "filler", "filler", "filler"]
    return doc_ids, docs


def test_perfect_retrieval_scores_one():
    doc_ids, docs = _toy_corpus()
    queries = [FakeQuery("cat", ("cats",)), FakeQuery("boat", ("ships",))]
    assert recall_at_k(toy_embedding(), doc_ids, docs, queries, k=1)["recall"] == pytest.approx(1.0)


def test_recall_counts_both_gold_documents():
    doc_ids = ["cats", "kittens", "ships"]
    docs = ["cat cat", "kitten kitten", "ship ship"]
    queries = [FakeQuery("cat kitten", ("cats", "kittens"))]
    out = recall_at_k(toy_embedding(), doc_ids, docs, queries, k=2)
    assert out["recall"] == pytest.approx(1.0)


def test_a_query_with_no_known_words_scores_zero():
    """Its vector is all zeros, so every document scores exactly 0 and argpartition returns
    an arbitrary k. Crediting those hits pays a method for its vocabulary gaps - and the
    first version of this scorer did, awarding a perfect score to a query it could not
    read."""
    doc_ids, docs = _toy_corpus()
    queries = [FakeQuery("zzz", ("cats",))]
    out = recall_at_k(toy_embedding(), doc_ids, docs, queries, k=1)
    assert out["recall"] == 0.0
    assert out["queries_with_no_known_words"] == 1


def test_per_query_scores_are_returned_for_significance_testing():
    doc_ids, docs = _toy_corpus()
    queries = [FakeQuery("cat", ("cats",)), FakeQuery("zzz", ("cats",))]
    out = recall_at_k(toy_embedding(), doc_ids, docs, queries, k=1)
    assert out["per_query"].tolist() == [1.0, 0.0]


def test_batching_does_not_change_the_answer():
    """The batched matmul exists for speed. If it changed the score it would be a bug that
    looked like a tuning parameter."""
    doc_ids, docs = _toy_corpus()
    queries = [FakeQuery("cat", ("cats",)), FakeQuery("boat", ("ships",))] * 9
    embedding = toy_embedding()
    one = recall_at_k(embedding, doc_ids, docs, queries, k=1, batch=1)
    many = recall_at_k(embedding, doc_ids, docs, queries, k=1, batch=64)
    assert one["recall"] == pytest.approx(many["recall"])


# --- neighbour agreement ------------------------------------------------------------------


def test_a_space_agrees_completely_with_itself():
    embedding = toy_embedding()
    out = neighbour_agreement(embedding, embedding, top_n=5, k=2)
    assert out["mean_jaccard"] == pytest.approx(1.0)
    assert out["share_with_no_overlap"] == 0.0


def test_unrelated_spaces_agree_little():
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(60)]
    a = Embedding(words, rng.random((60, 10)))
    b = Embedding(words, rng.random((60, 10)))
    assert neighbour_agreement(a, b, top_n=60, k=5)["mean_jaccard"] < 0.3


def test_only_shared_words_are_compared():
    left = toy_embedding()
    right = Embedding(["cat", "kitten", "ship"], np.eye(3))
    out = neighbour_agreement(left, right, top_n=5, k=1)
    assert out["words_compared"] == 3


def test_too_few_shared_words_is_an_error_not_a_number():
    left = toy_embedding()
    right = Embedding(["zzz"], np.ones((1, 3)))
    with pytest.raises(ValueError, match="too few"):
        neighbour_agreement(left, right, top_n=5, k=1)


# --- significance -------------------------------------------------------------------------


def test_an_identical_pair_is_not_significant():
    scores = np.array([1.0, 0.0, 1.0, 0.5] * 25)
    out = paired_bootstrap(scores, scores)
    assert out["mean_difference"] == 0.0
    assert out["significant"] is False


def test_a_large_consistent_difference_is_significant():
    a = np.ones(200)
    b = np.zeros(200)
    out = paired_bootstrap(a, b)
    assert out["mean_difference"] == pytest.approx(1.0)
    assert out["significant"] is True


def test_the_interval_brackets_the_difference():
    rng = np.random.default_rng(1)
    a = rng.random(300)
    b = a - 0.05
    out = paired_bootstrap(a, b)
    assert out["ci_low"] <= out["mean_difference"] <= out["ci_high"]
