"""Tests for k-means, silhouette, the Rand index and purity.

No dataset and no network. Every test builds a configuration whose correct answer is
obvious by construction, because a clustering that is subtly wrong still returns labels and
a plausible score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cluster import adjusted_rand, kmeans, purity, silhouette, unit


def three_blobs(per_cluster: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three tight, well-separated groups on three orthogonal axes."""
    rng = np.random.default_rng(seed)
    centres = np.eye(3) * 5.0
    data = np.vstack([c + rng.normal(0, 0.1, (per_cluster, 3)) for c in centres])
    truth = np.repeat(np.arange(3), per_cluster)
    return data, truth


# --- k-means ------------------------------------------------------------------------------


def test_it_recovers_obvious_clusters():
    data, truth = three_blobs()
    result = kmeans(unit(data), 3, spherical=True, seed=0)
    assert adjusted_rand(result.labels, truth) > 0.99


def test_it_returns_one_label_per_point():
    data, _ = three_blobs()
    result = kmeans(unit(data), 3)
    assert len(result.labels) == len(data)


def test_it_uses_every_cluster_it_was_given():
    data, _ = three_blobs()
    result = kmeans(unit(data), 3, seed=0)
    assert len(np.unique(result.labels)) == 3


def test_it_converges_on_easy_data():
    data, _ = three_blobs()
    assert kmeans(unit(data), 3, seed=0).converged


def test_it_is_deterministic_given_a_seed():
    data, _ = three_blobs()
    a = kmeans(unit(data), 3, seed=4)
    b = kmeans(unit(data), 3, seed=4)
    assert a.labels.tolist() == b.labels.tolist()


def test_k_of_one_puts_everything_together():
    data, _ = three_blobs()
    assert len(np.unique(kmeans(unit(data), 1).labels)) == 1


def test_spherical_centroids_are_unit_length():
    """That is the definition of the algorithm, and the step the Euclidean variant skips."""
    data, _ = three_blobs()
    result = kmeans(unit(data), 3, spherical=True, seed=0)
    assert np.allclose(np.linalg.norm(result.centroids, axis=1), 1.0)


def test_euclidean_centroids_are_not_unit_length():
    """The mean of a set of unit vectors is not itself a unit vector. This is the whole
    difference between the two implementations, and it is invisible in the labels."""
    data, _ = three_blobs()
    result = kmeans(unit(data), 3, spherical=False, seed=0)
    assert not np.allclose(np.linalg.norm(result.centroids, axis=1), 1.0)


def test_the_two_variants_can_disagree():
    """On well-separated blobs they agree. On anything harder they need not, which is why
    'we used cosine k-means' does not identify an algorithm."""
    rng = np.random.default_rng(1)
    data = unit(rng.normal(size=(120, 8)))
    a = kmeans(data, 4, spherical=True, seed=0)
    b = kmeans(data, 4, spherical=False, seed=0)
    assert adjusted_rand(a.labels, b.labels) < 1.0


# --- silhouette ---------------------------------------------------------------------------


def test_silhouette_is_high_for_a_correct_clustering():
    data, truth = three_blobs()
    assert silhouette(unit(data), truth) > 0.7


def test_silhouette_is_low_for_a_random_clustering():
    data, _ = three_blobs()
    rng = np.random.default_rng(0)
    assert silhouette(unit(data), rng.integers(0, 3, size=len(data))) < 0.2


def test_silhouette_needs_at_least_two_clusters():
    data, _ = three_blobs()
    assert np.isnan(silhouette(unit(data), np.zeros(len(data), dtype=np.int64)))


def test_silhouette_rejects_an_unknown_metric():
    data, truth = three_blobs()
    with pytest.raises(ValueError, match="unknown metric"):
        silhouette(data, truth, metric="manhattan")


def test_a_singleton_cluster_scores_zero_rather_than_nan():
    """A point alone in its cluster has no within-cluster distance to average. Returning
    NaN would poison the mean silhouette for the whole run."""
    data = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    labels = np.array([0, 0, 1])
    assert np.isfinite(silhouette(data, labels))


# --- agreement measures -------------------------------------------------------------------


def test_a_perfect_clustering_has_a_rand_index_of_one():
    truth = np.repeat(np.arange(3), 10)
    assert adjusted_rand(truth, truth) == pytest.approx(1.0)


def test_relabelling_does_not_change_the_rand_index():
    """Cluster ids are arbitrary; a measure that depended on them would be measuring the
    label assignment order."""
    truth = np.repeat(np.arange(3), 10)
    permuted = np.where(truth == 0, 2, np.where(truth == 2, 0, truth))
    assert adjusted_rand(permuted, truth) == pytest.approx(1.0)


def test_a_random_clustering_has_a_rand_index_near_zero():
    rng = np.random.default_rng(0)
    truth = np.repeat(np.arange(3), 40)
    assert abs(adjusted_rand(rng.permutation(truth), truth)) < 0.1


def test_putting_everything_in_one_cluster_scores_zero():
    """Chance correction is what makes this zero rather than merely low. The uncorrected
    Rand index rewards it."""
    truth = np.repeat(np.arange(2), 30)
    assert adjusted_rand(np.zeros(len(truth), dtype=np.int64), truth) == pytest.approx(0.0)


def test_purity_is_one_for_a_perfect_clustering():
    truth = np.repeat(np.arange(3), 10)
    assert purity(truth, truth) == pytest.approx(1.0)


def test_purity_is_one_when_every_point_is_its_own_cluster():
    """The reason purity cannot be used to choose k: it is maximised by the most useless
    clustering available."""
    truth = np.repeat(np.arange(3), 10)
    assert purity(np.arange(len(truth)), truth) == pytest.approx(1.0)


def test_purity_of_one_cluster_is_the_majority_share():
    truth = np.array([0, 0, 0, 1])
    assert purity(np.zeros(4, dtype=np.int64), truth) == pytest.approx(0.75)


# --- normalisation ------------------------------------------------------------------------


def test_unit_makes_rows_unit_length():
    rng = np.random.default_rng(0)
    assert np.allclose(np.linalg.norm(unit(rng.normal(size=(10, 5))), axis=1), 1.0)


def test_unit_leaves_a_zero_row_alone():
    out = unit(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.isfinite(out).all()
    assert np.allclose(out[1], [0.6, 0.8])
