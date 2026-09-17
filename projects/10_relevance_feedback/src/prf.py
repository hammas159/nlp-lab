"""RM3 pseudo-relevance feedback over a BM25 first stage.

The idea is old and it works: run the query, assume the top few documents are relevant,
harvest terms from them, add those terms to the query, run it again. Reported as a mean
over a query set it reliably improves retrieval, which is why it appears in most
descriptions of a serious search pipeline.

The relevance model (Lavrenko & Croft, 2001), in the RM3 form that interpolates with the
original query:

    P(t | R) = sum over feedback documents d of  P(t | d) * P(d | q)
    q' = (1 - alpha) * q  +  alpha * P(t | R)

``P(d | q)`` is the first-stage score, normalised over the feedback set, so a document the
first stage was confident about contributes more terms. ``alpha`` is how far the query is
allowed to move.

Nothing here assumes the feedback documents are actually relevant, because nothing checks.
That is the whole mechanism, and it is also the whole risk: when the top-k are wrong, the
expansion is built from the wrong documents and the second run is worse than the first.
The literature calls that query drift; this project measures it per query instead of
averaging it away.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import tokenize
from shared.bm25 import BM25


def score_weighted(bm25: BM25, weights: dict[str, float]) -> np.ndarray:
    """BM25 over a *weighted* bag of terms.

    `shared.bm25.BM25.score` takes a string and gives every term weight 1, which is right
    for an unexpanded query and wrong for an expanded one - an expansion term harvested
    with weight 0.01 would otherwise count as much as a term the user typed.

    The saturation and length normalisation are unchanged; only the per-term contribution
    is scaled.
    """
    scores = np.zeros(bm25.n_docs, dtype=np.float32)
    for term, weight in weights.items():
        posting = bm25.postings.get(term)
        if not posting or weight == 0.0:
            continue
        idf = bm25.idf[term]
        for i, freq in posting.items():
            norm = 1 - bm25.b + bm25.b * bm25.lengths[i] / bm25.avg_len
            scores[i] += weight * idf * (freq * (bm25.k1 + 1)) / (freq + bm25.k1 * norm)
    return scores


def relevance_model(
    bm25: BM25,
    docs: list[str],
    initial_scores: np.ndarray,
    n_feedback: int = 10,
    n_terms: int = 20,
) -> dict[str, float]:
    """Top expansion terms and their weights, from the top ``n_feedback`` documents.

    Feedback documents are weighted by their first-stage score, softmax-free: the scores
    are simply normalised to sum to one over the feedback set. A softmax would introduce a
    temperature, which is another knob nobody reports.
    """
    top = np.argsort(-initial_scores)[:n_feedback]
    weights = initial_scores[top].astype(np.float64)
    if weights.sum() <= 0:
        return {}
    weights = weights / weights.sum()

    model: Counter = Counter()
    for rank, doc_index in enumerate(top):
        tokens = tokenize(docs[doc_index])
        if not tokens:
            continue
        counts = Counter(tokens)
        length = len(tokens)
        for term, count in counts.items():
            model[term] += weights[rank] * (count / length)

    return dict(model.most_common(n_terms))


def expand(original: str, model: dict[str, float], alpha: float = 0.5) -> dict[str, float]:
    """Interpolate the original query with the relevance model.

    The original query's terms are given uniform weight before interpolation, which is what
    an unexpanded BM25 query already assumes. Both sides are normalised first so ``alpha``
    means the same thing regardless of query length or how peaked the relevance model is.
    """
    query_terms = tokenize(original)
    weights: dict[str, float] = {}

    if query_terms:
        share = 1.0 / len(query_terms)
        for term in query_terms:
            weights[term] = weights.get(term, 0.0) + (1.0 - alpha) * share

    total = sum(model.values())
    if total > 0:
        for term, value in model.items():
            weights[term] = weights.get(term, 0.0) + alpha * (value / total)

    return weights


def recall_at_k(scores: np.ndarray, gold: set[int], k: int = 10) -> float:
    if not gold:
        return 0.0
    top = np.argsort(-scores)[:k]
    return len(gold & set(top.tolist())) / len(gold)
