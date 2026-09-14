"""Tests for shingling, MinHash and LSH.

No dataset and no network. The MinHash tests check the estimator against exact Jaccard on
constructed sets, which is the only way to know the approximation is working rather than
merely running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from minhash import LSH, MERSENNE, MinHash, jaccard, shingles, tokenize_code

# --- tokenizer ---------------------------------------------------------------------------


def test_tokenizer_keeps_punctuation():
    """Braces and semicolons carry structure in C; dropping them would make two functions
    with different control flow look alike."""
    assert tokenize_code("if (x) { y++; }") == ["if", "(", "x", ")", "{", "y", "+", "+", ";", "}"]


def test_tokenizer_splits_identifiers_from_numbers():
    assert tokenize_code("int x = 42;") == ["int", "x", "=", "42", ";"]


def test_tokenizer_handles_empty_input():
    assert tokenize_code("") == []
    assert tokenize_code(None) == []


# --- shingles ----------------------------------------------------------------------------


def test_shingle_count_is_n_minus_k_plus_one():
    tokens = [str(i) for i in range(20)]
    assert len(shingles(tokens, k=5)) == 20 - 5 + 1


def test_short_input_yields_one_shingle():
    assert len(shingles(["a", "b"], k=5)) == 1


def test_empty_input_yields_no_shingles():
    assert shingles([], k=5) == set()


def test_identical_token_streams_share_every_shingle():
    a = shingles(tokenize_code("int f(void) { return 1; }"))
    assert a == shingles(tokenize_code("int f(void) { return 1; }"))


def test_hashes_fit_the_prime_field():
    """Regression: 64-bit digests overflowed int64 in the signature computation. Values
    must stay below the 31-bit prime for (a*x + b) to be exact."""
    s = shingles(tokenize_code("int main(void) { return 0; }"))
    assert all(0 <= v < MERSENNE for v in s)


# --- exact Jaccard --------------------------------------------------------------------------


def test_jaccard_of_identical_sets_is_one():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_of_disjoint_sets_is_zero():
    assert jaccard({1, 2}, {3, 4}) == 0.0


def test_jaccard_of_two_empty_sets_is_one():
    assert jaccard(set(), set()) == 1.0


def test_jaccard_of_one_empty_set_is_zero():
    assert jaccard({1}, set()) == 0.0


def test_jaccard_half_overlap():
    assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)


# --- MinHash ---------------------------------------------------------------------------------


def test_signature_length_matches_permutations():
    assert MinHash(n_perm=64).signature({1, 2, 3}).shape == (64,)


def test_identical_sets_give_identical_signatures():
    m = MinHash(n_perm=64)
    assert np.array_equal(m.signature({1, 2, 3}), m.signature({1, 2, 3}))


def test_similarity_of_identical_sets_is_one():
    m = MinHash(n_perm=64)
    sig = m.signature({7, 8, 9})
    assert MinHash.similarity(sig, sig) == 1.0


def test_disjoint_sets_estimate_near_zero():
    m = MinHash(n_perm=128)
    a = m.signature(set(range(500)))
    b = m.signature(set(range(10_000, 10_500)))
    assert MinHash.similarity(a, b) < 0.05


def test_estimate_tracks_exact_jaccard():
    """The estimator must be close to the truth, not merely consistent with itself.

    128 permutations gives a standard error of sqrt(s(1-s)/n); the tolerance below is
    generous enough for that and tight enough to catch a broken hash family.
    """
    m = MinHash(n_perm=256)
    universe = list(range(1000))
    a = set(universe[:600])
    b = set(universe[300:900])  # |intersection| = 300, |union| = 900 -> 1/3
    true = jaccard(a, b)
    est = MinHash.similarity(m.signature(a), m.signature(b))
    assert abs(true - est) < 0.08, f"true {true:.3f}, estimated {est:.3f}"


def test_signatures_are_reproducible_across_instances():
    assert np.array_equal(
        MinHash(n_perm=32, seed=7).signature({1, 5, 9}),
        MinHash(n_perm=32, seed=7).signature({1, 5, 9}),
    )


# --- LSH ---------------------------------------------------------------------------------------


def test_lsh_threshold_formula():
    lsh = LSH(bands=32, rows=4)
    assert lsh.threshold == pytest.approx((1 / 32) ** (1 / 4))


def test_identical_signatures_always_collide():
    m = MinHash(n_perm=128)
    sig = m.signature(set(range(200)))
    lsh = LSH(bands=32, rows=4)
    lsh.add(0, sig)
    lsh.add(1, sig)
    assert (0, 1) in lsh.candidate_pairs()


def test_dissimilar_signatures_usually_do_not_collide():
    m = MinHash(n_perm=128)
    lsh = LSH(bands=32, rows=4)
    lsh.add(0, m.signature(set(range(400))))
    lsh.add(1, m.signature(set(range(50_000, 50_400))))
    assert lsh.candidate_pairs() == set()


def test_lsh_never_reports_a_pair_with_itself():
    m = MinHash(n_perm=128)
    lsh = LSH(bands=32, rows=4)
    lsh.add(0, m.signature({1, 2, 3}))
    assert all(i != j for i, j in lsh.candidate_pairs())


def test_lsh_is_a_filter_not_a_verifier():
    """LSH returns *candidates*. Some will be below the threshold, which is why the
    pipeline verifies every candidate with exact Jaccard rather than trusting the bucket."""
    m = MinHash(n_perm=128)
    lsh = LSH(bands=32, rows=4)
    a, b = set(range(500)), set(range(250, 750))  # Jaccard = 1/3, below 0.8
    lsh.add(0, m.signature(a))
    lsh.add(1, m.signature(b))
    for i, j in lsh.candidate_pairs():
        assert jaccard(a, b) < 0.8, "this candidate should not have passed verification"
        assert (i, j) == (0, 1)
