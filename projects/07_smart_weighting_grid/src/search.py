"""Scoring and recall, given two already-weighted matrices.

The similarity is the inner product of the weighted query and weighted document vectors.
Cosine is not applied here: in SMART notation cosine *is* the ``c`` normalisation, applied
to each side by `smart.weight`. Normalising again in the scorer would silently convert
every scheme in the grid into its cosine variant and collapse the thing being measured.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def recall_at_k(
    documents: sparse.csr_matrix,
    queries: sparse.csr_matrix,
    gold: list[set[int]],
    k: int = 10,
    batch: int = 128,
) -> np.ndarray:
    """Per-query recall@k. Batched, because the full score matrix does not fit.

    1,000 queries against 66,581 documents is 66 million scores; at eight bytes each that
    is half a gigabyte for a matrix used once.
    """
    n_docs = documents.shape[0]
    kk = min(k, n_docs - 1)
    per_query = np.zeros(len(gold), dtype=np.float64)
    transposed = documents.T.tocsr()

    for start in range(0, queries.shape[0], batch):
        stop = min(start + batch, queries.shape[0])
        scores = np.asarray((queries[start:stop] @ transposed).todense())
        top = np.argpartition(-scores, kk, axis=1)[:, :kk]
        for offset, row in enumerate(top):
            j = start + offset
            if not gold[j]:
                continue
            # A query with no indexable term scores zero everywhere, and argpartition then
            # returns an arbitrary k. Crediting those would pay a scheme for the words its
            # index could not hold.
            if scores[offset].max() <= 0:
                continue
            per_query[j] = len(gold[j] & set(row.tolist())) / len(gold[j])

    return per_query


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = 2000, seed: int = 0) -> dict:
    """Resample queries, not scores - the unit of variation is the query."""
    rng = np.random.default_rng([seed, 0x5A17])
    diff = a - b
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "mean_difference": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "significant": bool(lo > 0 or hi < 0),
    }


def component_spread(scores: dict[str, float], position: int) -> dict[str, float]:
    """Mean recall grouped by one letter of the document code.

    Position 0 is the term-frequency component, 1 the document-frequency component, 2 the
    normalisation. The spread of these group means is how much that one decision moves the
    answer on average, which is a more honest summary than naming the single best scheme -
    the best scheme is one draw and the group mean is forty-five.
    """
    groups: dict[str, list[float]] = {}
    for code, value in scores.items():
        groups.setdefault(code[position], []).append(value)
    return {letter: float(np.mean(values)) for letter, values in sorted(groups.items())}
