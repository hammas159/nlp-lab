"""Tests for the truncated SVD, the eigenvalue weighting and the Embedding container.

No dataset and no network. The SVD is checked against a matrix built from known factors, so
"it returned vectors" is not mistaken for "it returned the right vectors".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from factorize import Embedding, arpack_svd, factorize, randomized_svd, truncated_svd

WORDS = [f"w{i}" for i in range(40)]


def low_rank(rank: int = 5, size: int = 40, seed: int = 0) -> sparse.csr_matrix:
    """A matrix that genuinely has the rank we will ask the SVD to find."""
    rng = np.random.default_rng(seed)
    left = rng.random((size, rank))
    right = rng.random((rank, size))
    return sparse.csr_matrix(left @ right)


# --- the decomposition --------------------------------------------------------------------


def test_singular_values_come_back_descending():
    """Regression guard. `svds` returns them ascending, so `s[:d]` on the raw output keeps
    the *least* important components - a mistake that produces vectors, not an error."""
    _, s, _ = truncated_svd(low_rank(), dim=10)
    assert np.all(np.diff(s) <= 0)


def test_it_recovers_the_rank_that_is_there():
    """A rank-5 matrix has five non-negligible singular values and the rest near zero."""
    _, s, _ = truncated_svd(low_rank(rank=5), dim=12)
    assert s[4] > 1e-6
    assert s[5] < s[0] * 1e-6


def test_reconstruction_is_close_for_a_full_rank_request():
    matrix = low_rank(rank=4, size=30)
    u, s, vt = truncated_svd(matrix, dim=8)
    approx = (u * s) @ vt
    assert np.allclose(approx, matrix.toarray(), atol=1e-8)


def test_dimension_is_clamped_to_what_the_matrix_allows():
    """`svds` raises rather than clamping when asked for more components than the matrix
    has. A small vocabulary is a legitimate input."""
    embedding = factorize(low_rank(size=6), WORDS[:6], dim=300)
    assert embedding.dim <= 5


# --- the randomized approximation ---------------------------------------------------------


def test_randomized_matches_arpack_on_singular_values():
    """The approximation is checked against the exact decomposition, not against itself.
    A randomized method that converged to the wrong subspace would still return vectors."""
    matrix = low_rank(rank=8, size=60)
    _, exact, _ = arpack_svd(matrix, dim=8)
    _, approx, _ = randomized_svd(matrix, dim=8)
    assert np.allclose(exact, approx, rtol=1e-6)


def test_randomized_recovers_an_exactly_low_rank_matrix():
    matrix = low_rank(rank=5, size=50)
    u, s, vt = randomized_svd(matrix, dim=5)
    assert np.allclose((u * s) @ vt, matrix.toarray(), atol=1e-8)


def test_randomized_spans_the_same_subspace_as_arpack():
    """Singular vectors are only defined up to sign, so the subspaces are compared rather
    than the vectors: the absolute cosine between matched components must be one."""
    matrix = low_rank(rank=6, size=60)
    exact_u, _, _ = arpack_svd(matrix, dim=6)
    approx_u, _, _ = randomized_svd(matrix, dim=6)
    alignment = np.abs(np.sum(exact_u * approx_u, axis=0))
    assert np.allclose(alignment, 1.0, atol=1e-5)


def test_power_iterations_help_on_a_slowly_decaying_spectrum():
    """A PMI matrix does not have a sharp rank cutoff. With no power iterations the random
    subspace is contaminated by the tail, and the recovered singular values are too small."""
    rng = np.random.default_rng(3)
    size = 80
    spectrum = np.diag(1.0 / np.arange(1, size + 1))  # slow decay, no cutoff
    left, _ = np.linalg.qr(rng.standard_normal((size, size)))
    right, _ = np.linalg.qr(rng.standard_normal((size, size)))
    matrix = sparse.csr_matrix(left @ spectrum @ right.T)

    _, exact, _ = arpack_svd(matrix, dim=10)
    _, none, _ = randomized_svd(matrix, dim=10, n_iter=0)
    _, powered, _ = randomized_svd(matrix, dim=10, n_iter=4)
    assert np.abs(powered - exact).sum() < np.abs(none - exact).sum()


def test_randomized_is_the_default():
    """ARPACK had not produced one factorisation in twelve minutes at the size this project
    runs at, so the default matters."""
    matrix = low_rank(rank=6, size=50)
    assert np.allclose(truncated_svd(matrix, dim=6)[1], randomized_svd(matrix, dim=6)[1])
    assert np.allclose(truncated_svd(matrix, dim=6, exact=True)[1], arpack_svd(matrix, dim=6)[1])


def test_randomized_is_deterministic_given_a_seed():
    matrix = low_rank(rank=6, size=50)
    assert np.allclose(
        randomized_svd(matrix, dim=6, seed=2)[1], randomized_svd(matrix, dim=6, seed=2)[1]
    )


# --- eigenvalue weighting -----------------------------------------------------------------


def test_eigenvalue_weight_changes_the_vectors():
    matrix = low_rank()
    a = factorize(matrix, WORDS, dim=6, eigenvalue_weight=1.0).vectors
    b = factorize(matrix, WORDS, dim=6, eigenvalue_weight=0.5).vectors
    assert not np.allclose(a, b)


def test_weight_of_zero_discards_the_singular_values():
    """p = 0 leaves U untouched: the singular values contribute nothing at all.

    Note that this does *not* make the word vectors unit length. U's columns are
    orthonormal; its rows are a truncation of an orthonormal basis and have norms well
    below 1. Normalising is a separate decision, made in `evaluate.py`.
    """
    matrix = low_rank()
    u, _, _ = truncated_svd(matrix, dim=6)
    assert np.allclose(factorize(matrix, WORDS, dim=6, eigenvalue_weight=0.0).vectors, u)


def test_weight_of_one_reproduces_u_times_s():
    matrix = low_rank()
    u, s, _ = truncated_svd(matrix, dim=6)
    assert np.allclose(factorize(matrix, WORDS, dim=6, eigenvalue_weight=1.0).vectors, u * s)


def test_adding_context_vectors_changes_the_result():
    matrix = low_rank()
    without = factorize(matrix, WORDS, dim=6, add_context=False).vectors
    with_context = factorize(matrix, WORDS, dim=6, add_context=True).vectors
    assert not np.allclose(without, with_context)


# --- the container ------------------------------------------------------------------------


def test_lookup_by_word():
    embedding = factorize(low_rank(), WORDS, dim=4)
    assert "w0" in embedding
    assert "absent" not in embedding
    assert embedding["w3"].shape == (4,)


def test_normalised_gives_unit_vectors():
    normalised = factorize(low_rank(), WORDS, dim=4).normalised()
    assert np.allclose(np.linalg.norm(normalised.vectors, axis=1), 1.0)


def test_normalising_a_zero_vector_does_not_divide_by_zero():
    embedding = Embedding(["a", "b"], np.array([[0.0, 0.0], [3.0, 4.0]]))
    normalised = embedding.normalised()
    assert np.isfinite(normalised.vectors).all()
    assert np.allclose(normalised["b"], [0.6, 0.8])


def test_restrict_keeps_the_requested_order():
    embedding = factorize(low_rank(), WORDS, dim=4)
    restricted = embedding.restrict_to(["w5", "w1", "absent", "w9"])
    assert restricted.words == ["w5", "w1", "w9"]
    assert np.allclose(restricted["w1"], embedding["w1"])


def test_restrict_to_nothing_is_empty_rather_than_an_error():
    restricted = factorize(low_rank(), WORDS, dim=4).restrict_to(["nope"])
    assert restricted.words == []


def test_restriction_is_what_makes_the_comparison_fair():
    """Two methods with different vocabularies produce a retrieval score that is partly a
    coverage score. Restricting both to the intersection is the only way the number is
    about the method."""
    left = factorize(low_rank(), WORDS, dim=4)
    right = factorize(low_rank(seed=1), WORDS[:20], dim=4)
    shared = left.restrict_to(right.words)
    assert set(shared.words) == set(right.words)
