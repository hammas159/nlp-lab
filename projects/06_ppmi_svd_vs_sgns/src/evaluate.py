"""Two measurements: a retrieval task, and how much two spaces agree with each other.

**Why retrieval rather than word-similarity benchmarks.** WordSim-353, SimLex-999 and the
Google analogy set are the usual intrinsic evaluations, and none of them is in the local
cache; downloading them would make this the one project in the lab that needs a dataset.
The lab already has a task with exact ground truth - HotpotQA, two gold paragraphs per
question - and project 01 already scores embeddings on it by mean pooling. Using the same
task and the same pooling means the numbers here sit directly beside that table.

The cost is real and worth naming: retrieval recall is a coarser instrument than a
similarity correlation, and it cannot see the analogy result, which is precisely where Levy
& Goldberg report that SGNS stays ahead of SVD. Nothing here tests analogies.

**Neighbour agreement** needs no gold data at all. If SGNS really is factorising a shifted
PMI matrix, then an explicit factorisation of that matrix should put the same words next to
each other. Overlap of the top-10 neighbour sets measures that directly, and unlike a task
score it cannot be matched by two spaces that are both mediocre in different ways.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import tokenize

from factorize import Embedding


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_documents(embedding: Embedding, docs: list[str]) -> np.ndarray:
    """Mean of the word vectors present, unit-normalised.

    Deliberately the crudest pooling there is, and identical to project 01's, because the
    question is what the *matrix* buys, not what a better pooling strategy buys.

    Convenience wrapper. Anything scoring more than one embedding should build a `Pooler`
    once instead - see its docstring for why.
    """
    return Pooler(embedding.words, docs).pool(embedding.vectors)


class Pooler:
    """Term counts for a fixed text set, so mean pooling becomes one sparse matmul.

    The obvious implementation re-tokenizes every document for every embedding. This
    project scores eight embeddings over 66,581 paragraphs, so that is eight passes of
    Python-level tokenizing and list building to produce eight matrices whose *only*
    difference is the word vectors they look up.

    Counting once gives a sparse document-by-term matrix ``C``. Mean pooling is then

        rows of (C @ W) divided by the row sums of C

    which BLAS does in a fraction of a second. Every embedding must share one vocabulary
    for this to hold, which the run already requires for the comparison to be fair.
    """

    def __init__(self, words: list[str], texts: list[str]) -> None:
        index = {w: i for i, w in enumerate(words)}
        rows, cols, data = [], [], []
        for i, text in enumerate(texts):
            counted = Counter(t for t in tokenize(text) if t in index)
            for word, n in counted.items():
                rows.append(i)
                cols.append(index[word])
                data.append(float(n))
        self.counts = sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(texts), len(words)), dtype=np.float64
        )
        self.lengths = np.asarray(self.counts.sum(axis=1)).ravel()
        #: Texts containing no in-vocabulary word at all. Their pooled vector is zero, and
        #: the caller must not credit them - see `recall_at_k`.
        self.empty = self.lengths == 0

    def pool(self, vectors: np.ndarray) -> np.ndarray:
        divisor = np.where(self.lengths == 0, 1.0, self.lengths)[:, None]
        return _unit((self.counts @ vectors) / divisor)


def recall_at_k(
    embedding: Embedding,
    doc_ids: list[str],
    docs: list[str],
    queries,
    k: int = 10,
    batch: int = 256,
    pooled: tuple[Pooler, Pooler] | None = None,
) -> dict:
    """Mean recall@k over queries, plus the per-query vector for significance testing.

    Scored in batches. One query at a time against 66,581 documents is a matrix-vector
    product per query and spends the run in Python overhead; one batch at a time is a
    matrix-matrix product that BLAS handles, and scoring every query at once would need a
    two-gigabyte score matrix.

    ``pooled`` accepts the ``(documents, queries)`` poolers so a caller scoring several
    embeddings over one corpus pays for tokenizing it once.
    """
    doc_pooler, query_pooler = pooled or (
        Pooler(embedding.words, docs),
        Pooler(embedding.words, [q.question for q in queries]),
    )
    matrix = doc_pooler.pool(embedding.vectors)
    query_vectors = query_pooler.pool(embedding.vectors)
    answerable = ~query_pooler.empty
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}

    per_query = np.zeros(len(queries), dtype=np.float64)
    kk = min(k, matrix.shape[0] - 1)
    for start in range(0, len(queries), batch):
        stop = min(start + batch, len(queries))
        scores = query_vectors[start:stop] @ matrix.T
        top = np.argpartition(-scores, kk, axis=1)[:, :kk]
        for offset, row in enumerate(top):
            j = start + offset
            # A query with no in-vocabulary words has an all-zero vector, so every document
            # scores exactly 0 and argpartition returns an arbitrary k. Crediting those hits
            # would pay a method for its vocabulary gaps, and the smaller vocabulary would
            # score better the more often it failed.
            if not answerable[j]:
                continue
            gold = {title_to_idx[t] for t in queries[j].gold_titles if t in title_to_idx}
            if gold:
                per_query[j] = len(gold & set(row.tolist())) / len(gold)

    return {
        "recall": float(per_query.mean()),
        "per_query": per_query,
        "queries_with_no_known_words": int((~answerable).sum()),
    }


def neighbour_agreement(left: Embedding, right: Embedding, top_n: int = 2_000, k: int = 10) -> dict:
    """Mean Jaccard overlap of the top-k neighbour sets, over words both spaces contain.

    ``top_n`` restricts to the most frequent shared words. Rare words have unstable
    neighbourhoods in any space, so including them would measure sampling noise in both and
    report it as disagreement.
    """
    shared = [w for w in left.words[:top_n] if w in right]
    if len(shared) < 2:
        raise ValueError("the two spaces share too few words to compare")

    scores = []
    for space in (left, right):
        sub = _unit(np.vstack([space[w] for w in shared]))
        sim = sub @ sub.T
        np.fill_diagonal(sim, -np.inf)
        top = np.argpartition(-sim, k, axis=1)[:, :k]
        scores.append(top)

    overlaps = [
        len(set(a.tolist()) & set(b.tolist())) / len(set(a.tolist()) | set(b.tolist()))
        for a, b in zip(scores[0], scores[1])
    ]
    return {
        "words_compared": len(shared),
        "k": k,
        "mean_jaccard": float(np.mean(overlaps)),
        "median_jaccard": float(np.median(overlaps)),
        "share_with_no_overlap": float(np.mean([o == 0 for o in overlaps])),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = 2000, seed: int = 0) -> dict:
    """Resample queries, not scores. Copied in spirit from project 02, which exists because
    a table of numbers two points apart invites a ranking the sample size may not support.
    """
    rng = np.random.default_rng([seed, 0xB007])
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
