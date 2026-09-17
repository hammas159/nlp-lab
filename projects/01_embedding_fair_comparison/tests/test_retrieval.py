"""Tests for the benchmark construction, the metrics, and BM25.

No dataset and no network: every case uses a hand-built corpus, so CI exercises the
comparison machinery rather than Hugging Face's availability.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import Query, coverage, tokenize

from evaluate import metrics
from retrievers import BM25, LSA, NomicEmbed, TfIdf, _unit

DOCS = [
    "the cat sat on the mat",
    "dogs are loyal animals and dogs bark",
    "quantum entanglement links two particles",
    "the mat was purchased in a shop",
    "particles in quantum mechanics behave strangely",
]


# --- tokenizer -------------------------------------------------------------------------


def test_tokenizer_lowercases_and_strips_punctuation():
    assert tokenize("The Cat, sat!") == ["the", "cat", "sat"]


def test_tokenizer_keeps_digits():
    assert tokenize("BERT v2 scored 93.5") == ["bert", "v2", "scored", "93", "5"]


def test_tokenizer_is_shared_by_every_lexical_method():
    """A comparison where one method gets better preprocessing measures preprocessing."""
    assert TfIdf().fit(DOCS).vec.tokenizer is tokenize
    assert LSA().fit(DOCS).vec.tokenizer is tokenize


# --- metrics ---------------------------------------------------------------------------


def test_perfect_ranking_scores_one():
    order = np.array([3, 7, 0, 1])
    out = metrics(order, {3, 7})
    assert out["recall@1"] == 0.5  # one of two gold in the top 1
    assert out["recall@5"] == 1.0
    assert out["mrr"] == 1.0


def test_gold_at_rank_four_gives_reciprocal_rank_one_quarter():
    order = np.array([9, 8, 7, 3])
    assert metrics(order, {3})["mrr"] == pytest.approx(0.25)


def test_missing_gold_scores_zero():
    order = np.array([9, 8, 7, 6])
    out = metrics(order, {1, 2})
    assert out["recall@20"] == 0.0
    assert out["mrr"] == 0.0


def test_recall_is_monotonic_in_k():
    order = np.arange(50)
    out = metrics(order, {0, 30})
    assert out["recall@1"] <= out["recall@5"] <= out["recall@10"] <= out["recall@20"]


# --- BM25 -------------------------------------------------------------------------------


def test_bm25_ranks_the_obvious_document_first():
    bm25 = BM25().fit(DOCS)
    assert int(np.argmax(bm25.score("quantum entanglement"))) == 2


def test_bm25_returns_one_score_per_document():
    assert BM25().fit(DOCS).score("cat").shape == (len(DOCS),)


def test_bm25_scores_zero_for_an_unseen_term():
    assert BM25().fit(DOCS).score("zzzzz").sum() == 0.0


def test_bm25_idf_is_never_negative():
    """A term appearing in almost every document must not push scores down."""
    bm25 = BM25().fit(DOCS + ["the"] * 20)
    assert min(bm25.idf.values()) >= 0.0


def test_bm25_saturates_term_frequency():
    """Ten occurrences must not score ten times one - that is the point of k1."""
    bm25 = BM25().fit(["dog", "dog dog dog dog dog dog dog dog dog dog"])
    scores = bm25.score("dog")
    assert scores[1] > scores[0]
    assert scores[1] < 10 * scores[0]


def test_bm25_penalises_length():
    """Same single match, longer document, lower score - that is b."""
    short = "needle"
    long = "needle " + " ".join(f"filler{i}" for i in range(200))
    scores = BM25().fit([short, long]).score("needle")
    assert scores[0] > scores[1]


# --- vector helpers ----------------------------------------------------------------------


def test_unit_rows_have_norm_one():
    m = _unit(np.array([[3.0, 4.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(m, axis=1), 1.0)


def test_unit_handles_a_zero_row_without_dividing_by_zero():
    m = _unit(np.zeros((1, 5)))
    assert np.isfinite(m).all()


# --- benchmark construction ---------------------------------------------------------------


def test_coverage_flags_a_gold_title_missing_from_the_corpus():
    """If a gold document is not in the corpus, recall is capped and the run is invalid."""
    q = [Query("1", "q", ("present", "absent"), "easy", "bridge")]
    stats = coverage(["present", "other"], q)
    assert stats["missing_gold_titles"] == 1


def test_coverage_reports_zero_when_every_gold_is_present():
    q = [Query("1", "q", ("a", "b"), "easy", "bridge")]
    assert coverage(["a", "b", "c"], q)["missing_gold_titles"] == 0


# --- the claim the repo makes ---------------------------------------------------------------


def test_lexical_methods_agree_on_an_unambiguous_query():
    """Sanity: on a query with exact term overlap, every lexical method finds it."""
    query = "quantum entanglement particles"
    for cls in (BM25, TfIdf, LSA):
        best = int(np.argmax(cls().fit(DOCS).score(query)))
        assert best in {2, 4}, f"{cls.name} ranked doc {best} first"


def test_lsa_clamps_components_to_the_vocabulary_size():
    """Regression: TruncatedSVD raises rather than clamping when n_components exceeds
    the number of terms, so LSA crashed on any corpus with a small vocabulary."""
    tiny = ["cat sat mat", "dog barked loudly"]
    lsa = LSA().fit(tiny)
    assert lsa.n_components < 300
    assert lsa.score("cat").shape == (2,)


# --- the pretrained arm's cache -------------------------------------------------------------
#
# nomic-embed is one HTTP round-trip per document, a few hundred milliseconds each, so a
# 3,000-document corpus is a twenty-minute silence with nothing written until the end. These
# pin the resume behaviour, with the network stubbed out.


class _StubNomic(NomicEmbed):
    """Counts calls instead of making them, so the cache logic is testable offline."""

    def __init__(self):
        self.calls = 0

    def _embed(self, text: str) -> np.ndarray:
        self.calls += 1
        return np.full(4, float(len(text)), dtype=np.float32)


def test_a_cached_document_is_not_embedded_again():
    cache: dict = {}
    _StubNomic().fit(DOCS, cache=cache)
    resumed = _StubNomic()
    resumed.fit(DOCS, cache=cache)
    assert resumed.calls == 0


def test_a_half_filled_cache_only_embeds_what_is_missing():
    """The point of checkpointing: an interrupted run resumes rather than restarts."""
    partial = {str(i): [float(len(d))] * 4 for i, d in enumerate(DOCS[:3])}
    resumed = _StubNomic()
    resumed.fit(DOCS, cache=partial)
    assert resumed.calls == len(DOCS) - 3


def test_the_cache_is_checkpointed_during_the_run():
    """Not only at the end - a crash at document 2,900 must not throw away 2,899 of them."""
    seen = []
    stub = _StubNomic()
    stub.CHECKPOINT_EVERY = 2
    stub.fit(DOCS, cache={}, checkpoint=lambda c: seen.append(len(c)))
    assert seen[0] == 2
    assert seen[-1] == len(DOCS)


def test_a_fully_cached_run_writes_no_checkpoint():
    """Nothing was computed, so there is nothing to save."""
    cache = {str(i): [float(len(d))] * 4 for i, d in enumerate(DOCS)}
    seen = []
    _StubNomic().fit(DOCS, cache=cache, checkpoint=lambda c: seen.append(len(c)))
    assert seen == []


def test_resuming_reproduces_the_uninterrupted_matrix():
    """A resumed run and a clean one must give the same vectors, or the cache is a bug."""
    clean = _StubNomic().fit(DOCS, cache={})
    partial = {str(i): [float(len(d))] * 4 for i, d in enumerate(DOCS[:2])}
    resumed = _StubNomic().fit(DOCS, cache=partial)
    assert np.allclose(clean.matrix, resumed.matrix)
