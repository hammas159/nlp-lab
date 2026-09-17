"""Tests for PMI, the negative-sampling shift and context distribution smoothing.

PMI values are checked against arithmetic done by hand on tiny matrices. A PMI
implementation that is off by a normalising constant still produces a plausible-looking
matrix and a plausible-looking embedding, so "it ran" is not evidence of anything.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pmi import ALPHA_NONE, ALPHA_SGNS, pmi_matrix, shift_of, sparsity


def dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense())


# --- the arithmetic -----------------------------------------------------------------------


def test_pmi_of_an_independent_pair_is_zero():
    """If P(w,c) = P(w)P(c) the PMI is log 1 = 0 - and the positive form then stores
    nothing, which is why an independent pair costs no memory."""
    counts = sparse.csr_matrix(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert pmi_matrix(counts, positive=False).nnz == 0


def test_pmi_matches_hand_arithmetic():
    counts = sparse.csr_matrix(np.array([[3.0, 1.0], [1.0, 5.0]]))
    total = 10.0
    rows = np.array([4.0, 6.0])
    cols = np.array([4.0, 6.0])
    expected = math.log((3.0 / total) / ((rows[0] / total) * (cols[0] / total)))
    assert dense(pmi_matrix(counts, positive=False))[0, 0] == pytest.approx(expected)


def test_positive_clipping_removes_negative_cells():
    counts = sparse.csr_matrix(np.array([[3.0, 1.0], [1.0, 5.0]]))
    plain = dense(pmi_matrix(counts, positive=False))
    clipped = dense(pmi_matrix(counts, positive=True))
    assert (plain < 0).any()
    assert (clipped >= 0).all()


def test_shift_subtracts_log_k():
    counts = sparse.csr_matrix(np.array([[3.0, 1.0], [1.0, 5.0]]))
    unshifted = dense(pmi_matrix(counts, k=1, positive=False))
    shifted = dense(pmi_matrix(counts, k=5, positive=False))
    assert (unshifted - shifted) == pytest.approx(np.full((2, 2), math.log(5)))


def test_shift_of_one_is_no_shift():
    assert shift_of(1) == 0.0
    assert shift_of(5) == pytest.approx(math.log(5))


def test_a_larger_shift_makes_the_positive_matrix_sparser():
    """The shift is what SGNS's negative-sampling count k does to the matrix it factorises.
    More negatives means a higher bar for a cell to survive the positive clipping."""
    rng = np.random.default_rng(0)
    dense_counts = rng.integers(1, 40, size=(30, 30)).astype(float)
    counts = sparse.csr_matrix(dense_counts)
    assert pmi_matrix(counts, k=10).nnz < pmi_matrix(counts, k=1).nnz


# --- context distribution smoothing -------------------------------------------------------


def test_smoothing_changes_the_matrix():
    rng = np.random.default_rng(1)
    counts = sparse.csr_matrix(rng.integers(1, 50, size=(20, 20)).astype(float))
    plain = dense(pmi_matrix(counts, alpha=ALPHA_NONE, positive=False))
    smoothed = dense(pmi_matrix(counts, alpha=ALPHA_SGNS, positive=False))
    assert not np.allclose(plain, smoothed)


def test_smoothing_is_less_generous_to_rare_contexts():
    """Raising the context distribution to 0.75 flattens it: rare contexts get a larger
    denominator than their raw probability, so their PMI falls. That is the whole reason
    SGNS's negative sampling distribution matters to a counting model."""
    counts = np.ones((3, 3))
    counts[:, 0] = 100.0  # one very frequent context, two rare ones
    matrix = sparse.csr_matrix(counts)
    plain = dense(pmi_matrix(matrix, alpha=ALPHA_NONE, positive=False))
    smoothed = dense(pmi_matrix(matrix, alpha=ALPHA_SGNS, positive=False))
    assert smoothed[1, 2] < plain[1, 2]


def test_alpha_of_one_is_no_smoothing():
    rng = np.random.default_rng(2)
    counts = sparse.csr_matrix(rng.integers(1, 30, size=(10, 10)).astype(float))
    a = dense(pmi_matrix(counts, alpha=1.0, positive=False))
    b = dense(pmi_matrix(counts, positive=False))
    assert np.allclose(a, b)


# --- shape and edge cases -----------------------------------------------------------------


def test_zero_cells_stay_zero():
    """A cell with no observed co-occurrence has PMI of negative infinity. It must not
    become a stored nonzero, or the matrix stops being sparse and the SVD stops fitting."""
    counts = sparse.csr_matrix(np.array([[5.0, 0.0], [0.0, 5.0]]))
    out = pmi_matrix(counts)
    assert out.nnz <= 2
    assert np.isfinite(out.data).all()


def test_output_shape_matches_input():
    counts = sparse.csr_matrix(np.ones((7, 7)) + np.eye(7) * 3)
    assert pmi_matrix(counts).shape == (7, 7)


def test_empty_matrix_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        pmi_matrix(sparse.csr_matrix((4, 4)))


def test_no_infinities_or_nans_survive():
    rng = np.random.default_rng(3)
    dense_counts = rng.integers(0, 5, size=(40, 40)).astype(float)
    out = pmi_matrix(sparse.csr_matrix(dense_counts))
    assert np.isfinite(out.data).all()


def test_sparsity_is_a_fraction():
    counts = sparse.csr_matrix(np.eye(10) * 5 + np.ones((10, 10)))
    value = sparsity(pmi_matrix(counts))
    assert 0.0 <= value <= 1.0
