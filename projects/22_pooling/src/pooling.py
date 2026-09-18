"""Five ways to turn a transformer's token vectors into one sentence vector.

A bi-encoder produces one vector per *token*. Turning that into one vector per *sentence*
needs a pooling step, and the choice is usually presented as a free hyperparameter - mean is
the default, `[CLS]` is "what BERT does", max is occasionally tried.

It is not free. **A sentence encoder is trained end to end with one specific pooling**, and
the training signal shapes the token vectors to suit it. `bge-small-en-v1.5` ships
`pooling_mode_cls_token: true`; `all-MiniLM-L6-v2` ships `pooling_mode_mean_tokens: true`.
Swapping the pooling at inference does not compare two pooling operations - it asks what
happens when you read the vectors in a way the model was never optimised to be read.

``cls``       the first token. What the model is trained to put the summary in, if it was
              trained that way.
``mean``      masked mean over real tokens. The default nearly everywhere.
``max``       masked max per dimension. Keeps the strongest signal and discards the rest.
``last``      the final real token, ignoring padding. What a decoder-style model would use.
``idf_mean``  mean weighted by inverse document frequency, so common subwords contribute
              less. The cheapest stand-in for "attention-weighted", and the only option here
              that uses corpus statistics rather than the model alone.

Padding is masked everywhere. A mean that averages over pad positions silently scores short
documents differently from long ones, which looks like a pooling effect and is not.
"""

from __future__ import annotations

import numpy as np

NEGATIVE = -1e9


def cls(hidden: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """The first token of every sequence."""
    return hidden[:, 0, :]


def mean(hidden: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Masked mean. The denominator counts real tokens only."""
    m = mask[:, :, None].astype(np.float64)
    return (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1.0)


def maximum(hidden: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Masked max.

    Padding is pushed to a large negative value rather than zero: with zeros, a dimension
    whose real values are all negative would take 0 from a pad position, so the result would
    depend on how much padding the batch happened to have.
    """
    m = mask[:, :, None].astype(bool)
    return np.where(m, hidden, NEGATIVE).max(axis=1)


def last(hidden: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """The final real token of each sequence."""
    lengths = mask.sum(axis=1).astype(int)
    index = np.maximum(lengths - 1, 0)
    return hidden[np.arange(hidden.shape[0]), index, :]


def idf_mean(hidden: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Mean weighted by per-token weights, normally inverse document frequency.

    Falls back to the plain mean when no weights are supplied, so the function is total and
    a missing weight table cannot silently produce a different pooling under this name.
    """
    if weights is None:
        return mean(hidden, mask)
    w = (weights * mask)[:, :, None].astype(np.float64)
    return (hidden * w).sum(axis=1) / np.maximum(w.sum(axis=1), 1e-9)


POOLINGS = {
    "cls": cls,
    "mean": mean,
    "max": maximum,
    "last": last,
    "idf_mean": idf_mean,
}


def pool(name: str, hidden: np.ndarray, mask: np.ndarray, weights: np.ndarray | None = None):
    try:
        return POOLINGS[name](hidden, mask, weights)
    except KeyError:
        raise ValueError(f"unknown pooling {name!r}; expected one of {sorted(POOLINGS)}") from None


def unit(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot product is a cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


def token_idf(token_ids: list[list[int]], vocabulary_size: int) -> np.ndarray:
    """Inverse document frequency per token id, over the corpus being encoded."""
    document_count = np.zeros(vocabulary_size, dtype=np.float64)
    for ids in token_ids:
        for i in set(ids):
            document_count[i] += 1.0
    n = max(1, len(token_ids))
    return np.log((n + 1.0) / (document_count + 1.0)) + 1.0
