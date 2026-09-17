"""Truncated SVD of the PMI matrix, with the two read-out choices that change the answer.

Given ``M ~ U S V^T``, the word vectors are

    W = U_d * S_d ** p

``p`` is the eigenvalue weighting. The textbook choice is ``p = 1``, which is the
least-squares-optimal reconstruction of M. Levy, Goldberg & Dagan (2015) report that
``p = 0.5`` - the symmetric split of the singular values between the word and context
sides - works better for similarity, and that ``p = 0`` (discarding the singular values
entirely) is sometimes better still. None of these is more principled than the others once
the goal is a downstream task rather than reconstructing M, which is the point: a knob with
no default is a knob that has to be reported.

The second choice is whether to use ``W`` or ``W + C``. SGNS learns two matrices and
discards the context one; adding it instead is free, and is one of the transfers that
closes part of the gap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds


@dataclass
class Embedding:
    """A word matrix with its vocabulary. The only interface `evaluate.py` needs.

    Both the SVD side and the gensim side are converted to this, so the evaluation code
    cannot accidentally treat them differently.
    """

    words: list[str]
    vectors: np.ndarray
    label: str = ""

    def __post_init__(self) -> None:
        self.index = {w: i for i, w in enumerate(self.words)}

    def __contains__(self, word: str) -> bool:
        return word in self.index

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.index[word]]

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def normalised(self) -> Embedding:
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return Embedding(self.words, self.vectors / norms, self.label)

    def restrict_to(self, words: list[str]) -> Embedding:
        """Keep only ``words``, in their given order.

        The counting side caps its vocabulary to keep a V x V matrix tractable; gensim caps
        its own by ``min_count`` and arrives at a different set. Comparing them as trained
        would let one method embed words the other cannot, and a retrieval score would then
        be partly a coverage score. Both sides are restricted to the intersection.
        """
        kept = [w for w in words if w in self.index]
        rows = np.array([self.index[w] for w in kept], dtype=np.int64)
        return Embedding(kept, self.vectors[rows], self.label)


def arpack_svd(
    matrix: sparse.csr_matrix, dim: int = 300, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``svds`` with singular values returned in descending order.

    ``svds`` returns them ascending, which silently makes ``s[:d]`` the *least* important
    components if the caller assumes otherwise.

    Exact, and the reference the randomized version is checked against - but ARPACK is a
    Krylov method whose cost grows badly with the number of components requested. At 300
    components on a 20,000-square PMI matrix it had not produced a single factorisation
    after twelve minutes, which is why it is not the default.
    """
    dim = min(dim, min(matrix.shape) - 1)
    u, s, vt = svds(matrix.astype(np.float64), k=dim, random_state=seed)
    order = np.argsort(-s)
    return u[:, order], s[order], vt[order, :]


def randomized_svd(
    matrix: sparse.csr_matrix,
    dim: int = 300,
    n_oversamples: int = 10,
    n_iter: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko, Martinsson & Tropp's randomized range finder, implemented rather than imported.

    Project the matrix onto a random subspace slightly larger than the rank wanted,
    orthonormalise, and take an exact SVD of the small projected matrix. Cost is a handful
    of sparse matrix products instead of a Krylov iteration per component, which is the
    difference between a minute and never finishing at this size.

    ``n_iter`` power iterations sharpen the separation between the components being kept
    and the ones being discarded. A PMI matrix has a slowly decaying spectrum, so with
    ``n_iter=0`` the subspace is contaminated by the tail; each iteration re-projects
    through ``M M^T`` and raises the singular values to a higher power, which pushes the
    tail down. Orthonormalising between iterations is not optional - without it the
    repeated products lose all precision to the leading component.

    `test_randomized_matches_arpack` checks it against the exact decomposition, which is
    the only way to know an approximation is approximating the right thing.
    """
    matrix = matrix.astype(np.float64)
    rows, cols = matrix.shape
    dim = min(dim, min(rows, cols) - 1)
    size = min(dim + n_oversamples, min(rows, cols))

    rng = np.random.default_rng([seed, 0x57D])
    basis, _ = np.linalg.qr(matrix @ rng.standard_normal((cols, size)))
    for _ in range(n_iter):
        projected, _ = np.linalg.qr(matrix.T @ basis)
        basis, _ = np.linalg.qr(matrix @ projected)

    small = basis.T @ matrix
    u_small, s, vt = np.linalg.svd(small, full_matrices=False)
    return (basis @ u_small)[:, :dim], s[:dim], vt[:dim, :]


def truncated_svd(
    matrix: sparse.csr_matrix, dim: int = 300, seed: int = 0, exact: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top ``dim`` singular triplets, descending. Randomized unless ``exact`` is asked for."""
    if exact:
        return arpack_svd(matrix, dim=dim, seed=seed)
    return randomized_svd(matrix, dim=dim, seed=seed)


def factorize(
    matrix: sparse.csr_matrix,
    words: list[str],
    dim: int = 300,
    eigenvalue_weight: float = 0.5,
    add_context: bool = False,
    seed: int = 0,
    label: str = "",
) -> Embedding:
    u, s, vt = truncated_svd(matrix, dim=dim, seed=seed)
    scale = s**eigenvalue_weight
    vectors = u * scale
    if add_context:
        vectors = vectors + (vt.T * scale)
    return Embedding(words, vectors, label=label)
