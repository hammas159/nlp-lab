"""k-means, two ways, and the metrics used to choose k.

Text is clustered with cosine similarity because document length should not decide
membership - a long document and a short one about the same subject point the same way and
sit far apart in Euclidean distance. The standard way to get cosine behaviour from k-means
is **spherical k-means**: normalise every vector to unit length, assign by maximum dot
product, and re-normalise the centroids after each update.

The part that goes wrong is the last step. Normalising the input and then running ordinary
Euclidean k-means is *almost* spherical k-means - on the unit sphere, minimising squared
Euclidean distance and maximising dot product give the same assignment - but the centroid of
a set of unit vectors is not itself a unit vector. Skip the re-normalisation and the
centroids drift inward, the effective decision boundary changes, and the algorithm is no
longer the one named in the write-up.

Both are implemented here so the difference is a measurement rather than a warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Clustering:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    iterations: int
    converged: bool


def unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _initial_centroids(data: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding.

    Uniform seeding on text routinely picks two points from the same dense region and
    leaves a whole topic without a seed, so the run-to-run variance swamps whatever is
    being compared.
    """
    n = len(data)
    centroids = [data[rng.integers(n)]]
    for _ in range(1, k):
        distances = np.min(
            ((data[:, None, :] - np.array(centroids)[None, :, :]) ** 2).sum(axis=2), axis=1
        )
        total = distances.sum()
        if total <= 0:
            centroids.append(data[rng.integers(n)])
            continue
        centroids.append(data[rng.choice(n, p=distances / total)])
    return np.array(centroids)


def kmeans(
    data: np.ndarray,
    k: int,
    spherical: bool = True,
    max_iter: int = 100,
    tol: float = 1e-6,
    seed: int = 0,
) -> Clustering:
    """Lloyd's algorithm. ``spherical=True`` re-normalises centroids each iteration.

    With ``spherical=False`` this is ordinary Euclidean k-means - which, on unit-normalised
    input, is what a great deal of code calling itself cosine k-means actually runs.
    """
    rng = np.random.default_rng([seed, 0xC1051])
    centroids = _initial_centroids(data, k, rng)
    labels = np.zeros(len(data), dtype=np.int64)
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        if spherical:
            centroids = unit(centroids)
            assignments = np.argmax(data @ centroids.T, axis=1)
        else:
            distances = ((data[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            assignments = np.argmin(distances, axis=1)

        moved = float((assignments != labels).mean())
        labels = assignments

        new_centroids = np.zeros_like(centroids)
        for cluster in range(k):
            members = data[labels == cluster]
            # An empty cluster has no centroid. Re-seeding it at a random point is what
            # scikit-learn does; leaving it in place silently reduces k.
            new_centroids[cluster] = (
                members.mean(axis=0) if len(members) else data[rng.integers(len(data))]
            )

        shift = float(np.abs(new_centroids - centroids).max())
        centroids = new_centroids
        if moved == 0.0 or shift < tol:
            converged = True
            break

    if spherical:
        centroids = unit(centroids)
        inertia = float(-np.sum(data * centroids[labels]))
    else:
        inertia = float(((data - centroids[labels]) ** 2).sum())

    return Clustering(labels, centroids, inertia, iterations, converged)


def silhouette(data: np.ndarray, labels: np.ndarray, metric: str = "cosine") -> float:
    """Mean silhouette coefficient.

    The internal metric people use to choose k when they have no labels. It rewards
    clusters that are compact and far apart, which is not the same as clusters that
    correspond to anything.
    """
    unique = np.unique(labels)
    if len(unique) < 2:
        return float("nan")

    if metric == "cosine":
        normalised = unit(data)
        distances = 1.0 - normalised @ normalised.T
        np.clip(distances, 0.0, 2.0, out=distances)
    elif metric == "euclidean":
        distances = np.sqrt(
            np.maximum(
                (
                    (data**2).sum(axis=1)[:, None]
                    + (data**2).sum(axis=1)[None, :]
                    - 2 * data @ data.T
                ),
                0.0,
            )
        )
    else:
        raise ValueError(f"unknown metric {metric!r}; expected 'cosine' or 'euclidean'")

    scores = np.zeros(len(data))
    for i in range(len(data)):
        own = labels[i]
        same = labels == own
        same[i] = False
        if not same.any():
            scores[i] = 0.0
            continue
        a = distances[i, same].mean()
        b = min(distances[i, labels == other].mean() for other in unique if other != own)
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def adjusted_rand(labels: np.ndarray, truth: np.ndarray) -> float:
    """Adjusted Rand index against known classes.

    Chance-corrected, so a clustering that happens to split a balanced corpus in half does
    not score well for doing nothing.
    """
    from math import comb

    left = np.unique(labels)
    right = np.unique(truth)
    table = np.array([[int(((labels == i) & (truth == j)).sum()) for j in right] for i in left])

    sum_cells = sum(comb(int(n), 2) for n in table.ravel())
    sum_rows = sum(comb(int(n), 2) for n in table.sum(axis=1))
    sum_cols = sum(comb(int(n), 2) for n in table.sum(axis=0))
    total = comb(len(labels), 2)

    expected = sum_rows * sum_cols / total if total else 0.0
    maximum = (sum_rows + sum_cols) / 2
    return float((sum_cells - expected) / (maximum - expected)) if maximum != expected else 0.0


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Share of points in the majority class of their own cluster.

    Reported beside the Rand index because purity rises monotonically with k and reaches 1
    when every point is its own cluster - it is the metric that makes over-clustering look
    like progress.
    """
    total = 0
    for cluster in np.unique(labels):
        members = truth[labels == cluster]
        if len(members):
            total += int(np.bincount(members).max())
    return total / len(labels)
