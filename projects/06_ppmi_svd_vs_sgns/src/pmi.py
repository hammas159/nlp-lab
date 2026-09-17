"""PMI and the two weightings that turn it into the matrix SGNS factorises.

Levy & Goldberg (2014) showed that skip-gram with negative sampling, at its optimum, is
factorising a word-context matrix whose cells are

    M[w, c] = PMI(w, c) - log k

with ``k`` the number of negative samples. The neural model is not learning a different
kind of representation; it is computing a shifted PMI matrix and factorising it implicitly.
That makes ``k`` - which reads like an optimisation detail of negative sampling - a term in
the objective, and it is available here as a parameter of a counting model.

Because a sparse matrix cannot hold the log of a zero cell, the usable form is the positive
one, SPPMI:

    SPPMI_k[w, c] = max( PMI(w, c) - log k, 0 )

Clipping at zero is not a convenience. PMI is most negative and least reliable exactly
where the counts are smallest, so the clipped cells are the ones a count matrix knows least
about, and keeping them would give the factorisation its noisiest entries to fit.

The second weighting is context distribution smoothing. SGNS draws its negative samples
from the unigram distribution raised to the power 0.75, so the denominator it is implicitly
dividing by is not P(c) but

    P_a(c) = #(c)^a / sum_c' #(c')^a          with a = 0.75

which flattens the context distribution and makes PMI less generous to rare contexts.
Levy, Goldberg & Dagan (2015) found this to be one of the largest single transfers from the
neural side to the counting side.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

ALPHA_NONE = 1.0
ALPHA_SGNS = 0.75


def pmi_matrix(
    counts: sparse.csr_matrix,
    alpha: float = ALPHA_NONE,
    k: int = 1,
    positive: bool = True,
) -> sparse.csr_matrix:
    """Shifted, smoothed, optionally positive PMI over a co-occurrence count matrix.

    ``alpha=1`` and ``k=1`` give plain PPMI. ``alpha=0.75`` and ``k>1`` give the matrix
    Levy & Goldberg's derivation says SGNS is factorising.

    Only stored nonzeros are transformed. Every other cell has a count of zero, so its PMI
    is negative infinity and its positive part is zero - which is what a sparse matrix
    already represents. Densifying to compute logs of zeros would need 3.2 GB at this
    vocabulary size to produce a matrix of zeros.
    """
    counts = counts.tocsr().astype(np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("co-occurrence matrix is empty")

    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    col_sums = np.asarray(counts.sum(axis=0)).ravel()

    smoothed = col_sums**alpha
    smoothed_total = smoothed.sum()

    coo = counts.tocoo()
    p_wc = coo.data / total
    p_w = row_sums[coo.row] / total
    p_c = smoothed[coo.col] / smoothed_total

    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(p_wc / (p_w * p_c))
    values -= np.log(k)
    values[~np.isfinite(values)] = 0.0

    if positive:
        values = np.maximum(values, 0.0)

    out = sparse.coo_matrix((values, (coo.row, coo.col)), shape=counts.shape).tocsr()
    out.eliminate_zeros()
    return out


def shift_of(k: int) -> float:
    """``log k``, the constant SGNS subtracts. k = 1 is no shift."""
    return float(np.log(k))


def sparsity(matrix: sparse.csr_matrix) -> float:
    rows, cols = matrix.shape
    return 1.0 - matrix.nnz / (rows * cols)
