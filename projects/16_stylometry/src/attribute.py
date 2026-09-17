"""Two attributors: Burrows' Delta, and multinomial Naive Bayes.

Burrows' Delta (1987) is the standard stylometric method and is worth implementing rather
than importing, because what it does is unusual and easy to misdescribe. It does not model
the text. It **z-scores each feature across the corpus**, then compares a document to an
author's centroid by mean absolute difference in z units.

That normalisation is the whole method. It puts a rare feature and a common one on the same
scale, so a habit that shows up in a tenth of a percent of tokens counts as much as one that
shows up in five percent. That is exactly what you want if you believe style lives in small
consistent preferences - and exactly what makes Delta sensitive to any feature that happens
to correlate with the author, including topic.

Naive Bayes is included as the contrast: it weights by evidence rather than by z-score, so
it leans on frequent discriminative features. Running both over the same four views shows
whether a conclusion depends on the attributor or only on what it was shown.
"""

from __future__ import annotations

import math

import numpy as np


def frequency_matrix(documents: list[list[str]], vocabulary: dict[str, int]) -> np.ndarray:
    """Relative frequencies, one row per document.

    Relative, not raw: a 400-line function and a 20-line one must be comparable, and Delta
    is defined over frequencies.
    """
    out = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for i, tokens in enumerate(documents):
        if not tokens:
            continue
        for token in tokens:
            j = vocabulary.get(token)
            if j is not None:
                out[i, j] += 1.0
        total = out[i].sum()
        if total:
            out[i] /= total
    return out


class BurrowsDelta:
    """Nearest author-centroid in z-scored frequency space, by mean absolute difference."""

    def fit(self, matrix: np.ndarray, labels: np.ndarray) -> BurrowsDelta:
        self.mean = matrix.mean(axis=0)
        self.sd = matrix.std(axis=0)
        # A feature with no variance carries no information and would divide by zero. Set
        # its scale to 1 so it contributes exactly nothing rather than infinity.
        self.sd[self.sd == 0] = 1.0
        z = (matrix - self.mean) / self.sd
        self.classes = np.unique(labels)
        self.centroids = np.vstack([z[labels == c].mean(axis=0) for c in self.classes])
        return self

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        z = (matrix - self.mean) / self.sd
        # Mean absolute difference in z units - the Delta statistic itself.
        distances = np.abs(z[:, None, :] - self.centroids[None, :, :]).mean(axis=2)
        return self.classes[np.argmin(distances, axis=1)]


class NaiveBayes:
    """Multinomial Naive Bayes over counts, in log space."""

    def fit(self, counts: np.ndarray, labels: np.ndarray, alpha: float = 1.0) -> NaiveBayes:
        self.classes = np.unique(labels)
        self.log_prior = np.empty(len(self.classes))
        self.log_likelihood = np.empty((len(self.classes), counts.shape[1]))
        for i, label in enumerate(self.classes):
            rows = counts[labels == label]
            self.log_prior[i] = math.log(len(rows) / len(counts))
            totals = rows.sum(axis=0) + alpha
            self.log_likelihood[i] = np.log(totals / totals.sum())
        return self

    def predict(self, counts: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(counts @ self.log_likelihood.T + self.log_prior, axis=1)]


def count_matrix(documents: list[list[str]], vocabulary: dict[str, int]) -> np.ndarray:
    out = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for i, tokens in enumerate(documents):
        for token in tokens:
            j = vocabulary.get(token)
            if j is not None:
                out[i, j] += 1.0
    return out


def accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float((predicted == truth).mean())


def majority_baseline(train: np.ndarray, truth: np.ndarray) -> float:
    """Always answer with the commonest training class.

    Devign's provenance split is about 64/36, so this floor is high. Any attribution
    accuracy has to be read against it rather than against 50%.
    """
    values, counts = np.unique(train, return_counts=True)
    return float((truth == values[np.argmax(counts)]).mean())
