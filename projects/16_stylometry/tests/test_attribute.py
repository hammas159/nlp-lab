"""Tests for Burrows' Delta and the Naive Bayes contrast.

No dataset and no network. Delta's defining behaviour is the z-scoring, so the tests check
that directly rather than only checking that it classifies something correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from attribute import (
    BurrowsDelta,
    NaiveBayes,
    accuracy,
    count_matrix,
    frequency_matrix,
    majority_baseline,
)

VOCAB = {"a": 0, "b": 1, "c": 2}


def separable():
    """Two classes with clearly different feature profiles."""
    docs = [
        ["a", "a", "a", "b"],
        ["a", "a", "a", "a"],
        ["a", "a", "b", "b"],
        ["c", "c", "c", "b"],
        ["c", "c", "c", "c"],
        ["c", "c", "b", "b"],
    ]
    labels = np.array([0, 0, 0, 1, 1, 1])
    return docs, labels


# --- matrices -----------------------------------------------------------------------------


def test_counts_are_raw_counts():
    matrix = count_matrix([["a", "a", "b"]], VOCAB)
    assert matrix[0].tolist() == [2.0, 1.0, 0.0]


def test_frequencies_sum_to_one():
    matrix = frequency_matrix([["a", "a", "b"]], VOCAB)
    assert matrix[0].sum() == pytest.approx(1.0)


def test_frequencies_make_documents_comparable():
    """A long document and a short one with the same profile must be identical. Delta is
    defined over frequencies for exactly this reason."""
    short = frequency_matrix([["a", "b"]], VOCAB)
    long = frequency_matrix([["a"] * 50 + ["b"] * 50], VOCAB)
    assert np.allclose(short, long)


def test_an_empty_document_does_not_divide_by_zero():
    matrix = frequency_matrix([[], ["a"]], VOCAB)
    assert np.isfinite(matrix).all()


def test_tokens_outside_the_vocabulary_are_ignored():
    matrix = count_matrix([["a", "zzz"]], VOCAB)
    assert matrix[0].sum() == 1.0


# --- Burrows' Delta -----------------------------------------------------------------------


def test_delta_separates_two_profiles():
    docs, labels = separable()
    matrix = frequency_matrix(docs, VOCAB)
    model = BurrowsDelta().fit(matrix, labels)
    assert accuracy(model.predict(matrix), labels) == pytest.approx(1.0)


def test_delta_z_scores_the_features():
    """The defining step. After fitting, the stored mean and scale are the corpus
    statistics, and a feature is compared in units of its own variation."""
    docs, labels = separable()
    matrix = frequency_matrix(docs, VOCAB)
    model = BurrowsDelta().fit(matrix, labels)
    assert np.allclose(model.mean, matrix.mean(axis=0))
    assert np.all(model.sd > 0)


def test_a_constant_feature_does_not_divide_by_zero():
    """A feature with no variance carries no information; its scale is set to 1 so it
    contributes nothing rather than infinity."""
    matrix = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    labels = np.array([0, 0, 1])
    model = BurrowsDelta().fit(matrix, labels)
    assert np.isfinite(model.centroids).all()
    assert np.isfinite(model.predict(matrix)).all()


def test_delta_weights_a_rare_feature_as_heavily_as_a_common_one():
    """The property that makes Delta a stylometric method - and the property that makes it
    sensitive to anything correlated with the author, including topic.

    Feature 0 varies between 0.90 and 0.91; feature 1 between 0.09 and 0.10. In raw terms
    they move by the same tiny amount, but each is scaled by its own standard deviation, so
    both count equally.
    """
    matrix = np.array([[0.90, 0.10], [0.91, 0.09], [0.90, 0.09], [0.91, 0.10]])
    labels = np.array([0, 1, 0, 1])
    model = BurrowsDelta().fit(matrix, labels)
    scaled = (matrix - model.mean) / model.sd
    assert abs(scaled[:, 0].std() - scaled[:, 1].std()) < 1e-9


def test_delta_returns_one_label_per_document():
    docs, labels = separable()
    matrix = frequency_matrix(docs, VOCAB)
    assert len(BurrowsDelta().fit(matrix, labels).predict(matrix)) == len(docs)


# --- Naive Bayes --------------------------------------------------------------------------


def test_naive_bayes_separates_two_profiles():
    docs, labels = separable()
    matrix = count_matrix(docs, VOCAB)
    model = NaiveBayes().fit(matrix, labels)
    assert accuracy(model.predict(matrix), labels) == pytest.approx(1.0)


def test_naive_bayes_survives_an_unseen_profile():
    docs, labels = separable()
    matrix = count_matrix(docs, VOCAB)
    model = NaiveBayes().fit(matrix, labels)
    assert model.predict(np.zeros((1, len(VOCAB))))[0] in (0, 1)


def test_naive_bayes_stays_finite_on_a_long_document():
    """A product of thousands of feature probabilities underflows to zero in double
    precision; log space is what prevents every document landing in one class."""
    docs, labels = separable()
    model = NaiveBayes().fit(count_matrix(docs, VOCAB), labels)
    long = count_matrix([["a"] * 5_000], VOCAB)
    assert model.predict(long)[0] == 0


# --- the floor ----------------------------------------------------------------------------


def test_the_majority_floor_uses_the_training_distribution():
    train = np.array([0, 0, 0, 1])
    truth = np.array([0, 0, 1, 1])
    assert majority_baseline(train, truth) == pytest.approx(0.5)


def test_the_floor_is_high_on_an_imbalanced_task():
    """Devign's provenance split is about 64/36, so any attribution accuracy has to be read
    against this rather than against 50%."""
    train = np.array([0] * 64 + [1] * 36)
    truth = np.array([0] * 64 + [1] * 36)
    assert majority_baseline(train, truth) == pytest.approx(0.64)


def test_accuracy_is_the_share_correct():
    assert accuracy(np.array([1, 0, 1]), np.array([1, 0, 0])) == pytest.approx(2 / 3)
