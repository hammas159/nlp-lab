"""MinHash and LSH, implemented rather than imported.

Exact-match-on-a-normal-form finds only what normalisation happens to collapse. Two
functions differing by a single statement hash differently and are invisible to it - which
is the limitation stated in the devign-leakage repository, and the one this measures.

MinHash estimates Jaccard similarity between shingle sets in fixed space, and LSH makes
finding the similar pairs sub-quadratic. Both are short enough to write out, and writing
them out means the approximation can be checked against exact Jaccard rather than trusted.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np

# 2**31 - 1, a Mersenne prime. Deliberately 31 bits, not 61: the hash family computes
# (a*x + b) mod p, and with 61-bit values that product overflows int64 silently.
# At 31 bits the product is at most ~2**62 and stays exact.
MERSENNE = (1 << 31) - 1

TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|[^\s\w]")


def tokenize_code(text: str) -> list[str]:
    """Identifiers, numbers and single punctuation characters.

    Punctuation is kept: in C, braces and semicolons carry structure, and dropping them
    would make two functions with different control flow look alike.
    """
    return TOKEN.findall(text or "")


def shingles(tokens: list[str], k: int = 5) -> set[int]:
    """Hashed k-grams of tokens.

    k=5 is the usual choice for code: long enough that a shared 5-token sequence means
    something, short enough that a one-statement edit does not destroy every shingle.
    """
    if len(tokens) < k:
        joined = " ".join(tokens)
        return {_hash(joined)} if joined else set()
    return {_hash(" ".join(tokens[i : i + k])) for i in range(len(tokens) - k + 1)}


def _hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % MERSENNE


def jaccard(a: set[int], b: set[int]) -> float:
    """Exact Jaccard - the ground truth MinHash is approximating."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class MinHash:
    """Signatures via random linear hash functions (a*x + b) mod prime."""

    def __init__(self, n_perm: int = 128, seed: int = 0):
        self.n_perm = n_perm
        rng = np.random.default_rng(seed)
        # Coefficients as int64; a must be non-zero or the permutation is constant.
        self.a = rng.integers(1, MERSENNE, size=n_perm, dtype=np.int64)
        self.b = rng.integers(0, MERSENNE, size=n_perm, dtype=np.int64)

    def signature(self, shingle_set: set[int]) -> np.ndarray:
        if not shingle_set:
            return np.full(self.n_perm, MERSENNE, dtype=np.int64)
        values = np.fromiter(shingle_set, dtype=np.int64, count=len(shingle_set))
        # (n_shingles, n_perm) then min over shingles. Python ints keep this exact.
        hashed = (np.outer(values, self.a) + self.b) % MERSENNE
        return hashed.min(axis=0)

    @staticmethod
    def similarity(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Fraction of agreeing positions - an unbiased estimate of Jaccard."""
        return float((sig_a == sig_b).mean())


class LSH:
    """Banded locality-sensitive hashing over MinHash signatures.

    A pair becomes a candidate if it collides in *any* band. With b bands of r rows the
    probability of becoming a candidate is 1 - (1 - s^r)^b, which is a steep S-curve
    around s = (1/b)^(1/r) - the effective threshold.
    """

    def __init__(self, bands: int = 32, rows: int = 4):
        self.bands, self.rows = bands, rows
        self.n_perm = bands * rows
        self.buckets: list[dict[bytes, list[int]]] = [defaultdict(list) for _ in range(bands)]

    @property
    def threshold(self) -> float:
        return (1 / self.bands) ** (1 / self.rows)

    def add(self, index: int, signature: np.ndarray) -> None:
        for band in range(self.bands):
            chunk = signature[band * self.rows : (band + 1) * self.rows]
            self.buckets[band][chunk.tobytes()].append(index)

    def candidate_pairs(self) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for band in self.buckets:
            for members in band.values():
                if len(members) < 2:
                    continue
                # A bucket with very many members is almost always a degenerate case
                # (empty or near-empty documents); it would dominate the pair count.
                if len(members) > 200:
                    continue
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        pairs.add((members[i], members[j]))
        return pairs
