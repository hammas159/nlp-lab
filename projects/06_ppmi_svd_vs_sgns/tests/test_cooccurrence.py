"""Tests for vocabulary construction, subsampling and the co-occurrence count matrix.

No dataset and no network. The counts are checked against numbers worked out by hand on
sentences short enough to verify by eye, because a co-occurrence matrix that is merely
plausible is indistinguishable from one that is off by an index.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cooccurrence import (
    build,
    build_vocabulary,
    cooccurrence,
    subsample,
    to_ids,
)

# --- vocabulary ---------------------------------------------------------------------------


def test_vocabulary_is_ordered_by_frequency():
    vocab = build_vocabulary(["a"] * 5 + ["b"] * 3 + ["c"] * 9, min_count=1)
    assert vocab.words == ["c", "a", "b"]
    assert vocab.counts.tolist() == [9, 5, 3]


def test_min_count_removes_rare_types():
    vocab = build_vocabulary(["a"] * 5 + ["rare"], min_count=2)
    assert vocab.words == ["a"]


def test_max_size_truncates_after_the_count_cut():
    vocab = build_vocabulary([f"w{i % 50}" for i in range(500)], min_count=1, max_size=10)
    assert len(vocab) == 10


def test_index_agrees_with_words():
    vocab = build_vocabulary(["a", "b", "a", "c"], min_count=1)
    for i, word in enumerate(vocab.words):
        assert vocab.index[word] == i


# --- id mapping ---------------------------------------------------------------------------


def test_out_of_vocabulary_tokens_are_dropped_not_replaced():
    """An UNK placeholder would become one of the most frequent types in the corpus and
    would co-occur with everything, which is a large artefact in a PMI matrix."""
    vocab = build_vocabulary(["a"] * 5 + ["b"] * 5, min_count=2)
    ids = to_ids(["a", "zzz", "b", "qqq"], vocab)
    assert len(ids) == 2
    assert [vocab.words[i] for i in ids] == ["a", "b"]


def test_ids_are_empty_when_nothing_is_in_vocabulary():
    vocab = build_vocabulary(["a"] * 5, min_count=2)
    assert len(to_ids(["zzz"], vocab)) == 0


# --- subsampling --------------------------------------------------------------------------


def test_subsampling_removes_frequent_tokens_and_keeps_rare_ones():
    tokens = ["the"] * 100_000 + [f"rare{i}" for i in range(2_000)]
    vocab = build_vocabulary(tokens, min_count=1)
    ids = to_ids(tokens, vocab)
    kept = subsample(ids, vocab, seed=0)

    before = int((ids == vocab.index["the"]).sum())
    after = int((kept == vocab.index["the"]).sum())
    assert after < before / 2
    # A type at frequency far below the threshold should survive essentially intact.
    rare_before = int((ids == vocab.index["rare0"]).sum())
    rare_after = int((kept == vocab.index["rare0"]).sum())
    assert rare_after == rare_before


def test_subsampling_is_deterministic_given_a_seed():
    tokens = ["the"] * 20_000 + ["cat"] * 500
    vocab = build_vocabulary(tokens, min_count=1)
    ids = to_ids(tokens, vocab)
    assert np.array_equal(subsample(ids, vocab, seed=3), subsample(ids, vocab, seed=3))


def test_a_type_rarer_than_the_threshold_is_always_kept():
    """word2vec's keep probability exceeds 1 when f < t, so the clamp is what stops the
    formula being used outside its domain. 'rare' here is one token in 200,001, which is
    below the 1e-5 threshold; 'the' is far above it and must be thinned.
    """
    tokens = ["the"] * 200_000 + ["rare"]
    vocab = build_vocabulary(tokens, min_count=1)
    ids = to_ids(tokens, vocab)
    kept = subsample(ids, vocab, seed=0)
    assert int((kept == vocab.index["rare"]).sum()) == 1
    assert int((kept == vocab.index["the"]).sum()) < 200_000


# --- the matrix ---------------------------------------------------------------------------


def test_fixed_window_counts_are_exact():
    """`a b c` with window 1: a-b twice (once each direction), b-c twice. Nothing else."""
    ids = np.array([0, 1, 2], dtype=np.int32)
    matrix = cooccurrence(ids, size=3, window=1, dynamic=False).toarray()
    assert matrix[0, 1] == 1 and matrix[1, 0] == 1
    assert matrix[1, 2] == 1 and matrix[2, 1] == 1
    assert matrix[0, 2] == 0
    assert matrix.sum() == 4


def test_the_matrix_is_symmetric():
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 20, size=5_000).astype(np.int32)
    matrix = cooccurrence(ids, size=20, window=4, dynamic=True)
    assert (abs(matrix - matrix.T)).nnz == 0


def test_dynamic_window_downweights_distant_contexts():
    """word2vec samples an actual window uniformly from 1..L, so a context at distance d is
    counted with probability (L - d + 1)/L. A fixed window counts both equally, and that
    difference is routinely attributed to the neural objective instead."""
    ids = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    dynamic = cooccurrence(ids, size=5, window=3, dynamic=True).toarray()
    fixed = cooccurrence(ids, size=5, window=3, dynamic=False).toarray()

    assert dynamic[0, 1] > dynamic[0, 2] > dynamic[0, 3]
    assert fixed[0, 1] == fixed[0, 2] == fixed[0, 3]


def test_window_larger_than_the_text_does_not_crash():
    ids = np.array([0, 1], dtype=np.int32)
    matrix = cooccurrence(ids, size=2, window=10, dynamic=True)
    assert matrix.nnz == 2


def test_build_reports_what_it_did():
    tokens = ("the cat sat on the mat . the dog sat on the log " * 200).split()
    _, vocab, stats = build(tokens, min_count=2, max_size=50, window=3, do_subsample=False)
    assert stats["vocabulary"] == len(vocab)
    assert stats["tokens_after_subsampling"] == stats["tokens_in_vocabulary"]
    assert stats["window"] == 3
    assert stats["subsampled"] is False


def test_subsampling_actually_shortens_the_stream():
    tokens = ("the cat sat on the mat " * 3_000).split()
    _, _, without = build(tokens, min_count=1, max_size=50, do_subsample=False)
    _, _, with_sub = build(tokens, min_count=1, max_size=50, do_subsample=True)
    assert with_sub["tokens_after_subsampling"] < without["tokens_after_subsampling"]


def test_empty_stream_gives_an_empty_matrix():
    vocab = build_vocabulary(["a"] * 3, min_count=1)
    matrix = cooccurrence(to_ids([], vocab), size=len(vocab), window=2)
    assert matrix.nnz == 0


def test_a_window_of_zero_counts_nothing():
    ids = np.array([0, 1, 2], dtype=np.int32)
    assert cooccurrence(ids, size=3, window=0).nnz == 0


def test_vocabulary_is_unaffected_by_subsampling():
    """Vocabulary is built before subsampling, so every rung of the ladder shares one
    vocabulary and the recall differences cannot be coverage differences."""
    tokens = ("the cat sat on the mat " * 2_000).split()
    _, a, _ = build(tokens, min_count=1, max_size=50, do_subsample=False)
    _, b, _ = build(tokens, min_count=1, max_size=50, do_subsample=True)
    assert a.words == b.words


@pytest.mark.parametrize("dynamic", [True, False])
def test_counts_are_non_negative(dynamic):
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 10, size=1_000).astype(np.int32)
    matrix = cooccurrence(ids, size=10, window=3, dynamic=dynamic)
    assert matrix.data.min() > 0
