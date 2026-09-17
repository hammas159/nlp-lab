"""Two cheap learned baselines, so "gzip beats X" has an X that was actually tried.

Both are implemented here rather than imported, for the same reason the rest of this lab
implements its baselines: a comparison against a method nobody configured is not a
comparison. Neither needs a dependency beyond numpy.

`multinomial_nb` is the textbook Naive Bayes over token counts with Laplace smoothing.
`nearest_centroid` is the simplest possible vector-space classifier: one averaged TF-IDF
vector per class, cosine to each.

Both train in under a second on a thousand documents, which matters to the argument. The
headline claim for compression-based classification is that it is free of training; the
thing it is free of costs about a second.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|[^\s\w]")


def tokenize(text: str) -> list[str]:
    """Identifiers, numbers and punctuation, case preserved.

    Punctuation is kept because in C the braces and semicolons carry control flow - the
    same reasoning as project 04's tokenizer, and the same tokenizer shape.
    """
    return TOKEN.findall(text)


def _vocabulary(texts: list[str], min_count: int = 2) -> dict[str, int]:
    counter = Counter(t for text in texts for t in tokenize(text))
    return {w: i for i, (w, c) in enumerate(counter.most_common()) if c >= min_count}


def _count_matrix(texts: list[str], vocabulary: dict[str, int]) -> np.ndarray:
    out = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for i, text in enumerate(texts):
        for token, n in Counter(tokenize(text)).items():
            j = vocabulary.get(token)
            if j is not None:
                out[i, j] = n
    return out


def multinomial_nb(
    train_texts: list[str], train_labels: np.ndarray, test_texts: list[str], alpha: float = 1.0
) -> np.ndarray:
    """Multinomial Naive Bayes with Laplace smoothing, in log space.

    Log space is not an optimisation. A product of several thousand token probabilities
    underflows to exactly zero in double precision, and the classifier then returns the
    same class for every document.
    """
    vocabulary = _vocabulary(train_texts)
    if not vocabulary:
        return np.zeros(len(test_texts), dtype=train_labels.dtype)

    train = _count_matrix(train_texts, vocabulary)
    test = _count_matrix(test_texts, vocabulary)
    classes = np.unique(train_labels)

    log_prior = np.empty(len(classes))
    log_likelihood = np.empty((len(classes), len(vocabulary)))
    for c, label in enumerate(classes):
        rows = train[train_labels == label]
        log_prior[c] = math.log(len(rows) / len(train))
        totals = rows.sum(axis=0) + alpha
        log_likelihood[c] = np.log(totals / totals.sum())

    scores = test @ log_likelihood.T + log_prior
    return classes[np.argmax(scores, axis=1)]


def nearest_centroid(
    train_texts: list[str], train_labels: np.ndarray, test_texts: list[str]
) -> np.ndarray:
    """One TF-IDF centroid per class, cosine similarity to each."""
    vocabulary = _vocabulary(train_texts)
    if not vocabulary:
        return np.zeros(len(test_texts), dtype=train_labels.dtype)

    train = _count_matrix(train_texts, vocabulary)
    test = _count_matrix(test_texts, vocabulary)

    document_frequency = (train > 0).sum(axis=0)
    idf = np.log(len(train) / np.maximum(document_frequency, 1))
    train = np.log1p(train) * idf
    test = np.log1p(test) * idf

    classes = np.unique(train_labels)
    centroids = np.vstack([train[train_labels == label].mean(axis=0) for label in classes])

    def unit(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    return classes[np.argmax(unit(test) @ unit(centroids).T, axis=1)]
