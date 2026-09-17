"""The word-context count matrix, with word2vec's preprocessing available as options.

Levy, Goldberg & Dagan (2015) argue that most of what separates a neural word embedding
from a counting model is not the objective but a handful of preprocessing and weighting
decisions that arrived bundled with the neural implementations and were never applied to
the counting ones. Two of those decisions live here - the dynamic context window and the
subsampling of frequent words - and both are switches rather than defaults, so the ladder
in `run.py` can turn them on one at a time and attribute the change.

Everything is built from counts. There is no model, no objective and no gradient in this
file or in the two that follow it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import sparse

#: word2vec's subsampling threshold. Mikolov's paper gives P(discard) = 1 - sqrt(t/f); the
#: released C code, and gensim after it, use the form below. The SGNS baseline in this
#: project is gensim, so the counting side uses gensim's formula - otherwise the comparison
#: would include a difference neither paper is about.
SUBSAMPLE_T = 1e-5

#: Second entropy word for the subsampling mask, so its uniform stream cannot coincide with
#: another generator seeded the same way. Project 05 lost an afternoon to exactly that.
SUBSAMPLE_STREAM = 0x50B5A


@dataclass(frozen=True)
class Vocabulary:
    words: list[str]
    index: dict[str, int]
    counts: np.ndarray

    def __len__(self) -> int:
        return len(self.words)


def build_vocabulary(tokens: list[str], min_count: int = 2, max_size: int = 20_000) -> Vocabulary:
    """Most frequent types first, after a minimum-count cut.

    ``max_size`` is a tractability bound, not a modelling claim: the co-occurrence matrix is
    V x V, and the corpus here has 160,768 types. It is reported in the results rather than
    hidden, because a vocabulary cut changes what the tail of the PMI matrix contains.
    """
    counter = Counter(tokens)
    kept = [(w, c) for w, c in counter.most_common() if c >= min_count][:max_size]
    words = [w for w, _ in kept]
    return Vocabulary(
        words=words,
        index={w: i for i, w in enumerate(words)},
        counts=np.array([c for _, c in kept], dtype=np.int64),
    )


def to_ids(tokens: list[str], vocab: Vocabulary) -> np.ndarray:
    """Token stream as vocabulary indices, with out-of-vocabulary positions removed.

    Removed, not replaced by a placeholder: an UNK symbol would become one of the most
    frequent types in the corpus and would co-occur with everything, which is a large
    artefact in a PMI matrix.
    """
    index = vocab.index
    return np.array([index[t] for t in tokens if t in index], dtype=np.int32)


def subsample(
    ids: np.ndarray, vocab: Vocabulary, t: float = SUBSAMPLE_T, seed: int = 0
) -> np.ndarray:
    """Drop frequent tokens with word2vec's keep probability.

        keep(w) = (sqrt(f/t) + 1) * t/f

    This is not a speed trick. Dropping a frequent word also *widens* the context window
    around the words that survive, because the window is applied after the deletion - so
    subsampling changes which pairs are counted, not merely how many.
    """
    freq = vocab.counts / vocab.counts.sum()
    ratio = freq / t
    keep_prob = np.minimum(1.0, (np.sqrt(ratio) + 1.0) * (1.0 / ratio))
    rng = np.random.default_rng([seed, SUBSAMPLE_STREAM])
    return ids[rng.random(len(ids)) < keep_prob[ids]]


def cooccurrence(
    ids: np.ndarray, size: int, window: int = 5, dynamic: bool = True
) -> sparse.csr_matrix:
    """Symmetric word-context counts over a window.

    ``dynamic`` reproduces word2vec's window sampling. word2vec draws an actual window size
    uniformly from 1..L for every token, so a context at distance d is counted with
    probability (L - d + 1)/L. Using that expectation directly is the deterministic
    equivalent, and it means near contexts are weighted more heavily than far ones - which
    a fixed window does not do, and which is one of the differences usually attributed to
    the neural objective.

    Built one offset at a time. The alternative - one array of every pair in the corpus -
    is about a gigabyte at this corpus size for no benefit, because the CSR conversion sums
    duplicates per offset just as well.
    """
    total = sparse.csr_matrix((size, size), dtype=np.float64)
    for d in range(1, window + 1):
        weight = (window - d + 1) / window if dynamic else 1.0
        left, right = ids[:-d], ids[d:]
        data = np.full(len(left), weight, dtype=np.float64)
        # Both directions, so the matrix is symmetric and #(w) equals #(c).
        forward = sparse.coo_matrix((data, (left, right)), shape=(size, size))
        backward = sparse.coo_matrix((data, (right, left)), shape=(size, size))
        total = total + forward.tocsr() + backward.tocsr()
    return total


def build(
    tokens: list[str],
    min_count: int = 2,
    max_size: int = 20_000,
    window: int = 5,
    dynamic: bool = True,
    do_subsample: bool = True,
    seed: int = 0,
) -> tuple[sparse.csr_matrix, Vocabulary, dict]:
    """Vocabulary, subsampling and counting in the order word2vec applies them."""
    vocab = build_vocabulary(tokens, min_count=min_count, max_size=max_size)
    ids = to_ids(tokens, vocab)
    kept = subsample(ids, vocab, seed=seed) if do_subsample else ids
    matrix = cooccurrence(kept, len(vocab), window=window, dynamic=dynamic)
    stats = {
        "types_in_corpus": len(set(tokens)),
        "vocabulary": len(vocab),
        "tokens_in_vocabulary": len(ids),
        "tokens_after_subsampling": len(kept),
        "nonzero_cells": int(matrix.nnz),
        "window": window,
        "dynamic_window": dynamic,
        "subsampled": do_subsample,
        "min_count": min_count,
    }
    return matrix, vocab, stats
