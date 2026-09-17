"""Tests for bigram counting and the contingency tables it produces.

No dataset and no network. The tables are checked against counts done by hand on sentences
short enough to enumerate, because a marginal that is off by one at the edges of the stream
produces tables that still sum correctly and scores that are quietly wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from bigrams import count, from_documents


def cell(data, pair: tuple[str, str]) -> tuple[float, float, float, float]:
    i = data.pairs.index(pair)
    return data.a[i], data.b[i], data.c[i], data.d[i]


# --- counting -----------------------------------------------------------------------------


def test_adjacent_pairs_are_counted():
    data = count(["a", "b", "c"])
    assert set(data.pairs) == {("a", "b"), ("b", "c")}
    assert data.total == 2


def test_a_repeated_pair_is_counted_once_per_occurrence():
    data = count(["a", "b", "a", "b"])
    a, _, _, _ = cell(data, ("a", "b"))
    assert a == 2


def test_the_table_sums_to_the_bigram_total():
    data = count(["a", "b", "c", "a", "b"])
    for i in range(len(data)):
        assert data.a[i] + data.b[i] + data.c[i] + data.d[i] == data.total


def test_marginals_use_first_and_second_positions_separately():
    """The subtle one. In `a b a`, the token `a` appears twice, but only once as a *first*
    element of a bigram and once as a *second*. Using one unigram count for both marginals
    is wrong at the edges of the stream and invisible in the output.
    """
    data = count(["a", "b", "a"])
    a, b, c, d = cell(data, ("a", "b"))
    assert a == 1  # "a b" once
    assert b == 0  # "a" is a first element once, all of it in this pair
    assert c == 0  # "b" is a second element once, all of it in this pair
    assert d == 1  # the other bigram, "b a"


def test_a_single_token_produces_no_bigrams():
    assert len(count(["a"])) == 0


def test_hapax_share_is_reported():
    data = count(["a", "b", "a", "b", "c", "d"])
    assert data.stats["hapax_bigrams"] == 3  # b-a, b-c, c-d
    assert data.stats["bigram_types"] == 4


# --- document boundaries ------------------------------------------------------------------


def test_documents_do_not_bleed_into_each_other():
    """Concatenating the corpus into one stream invents a bigram at every document
    boundary out of two words that never appeared together - and every one of them is a
    hapax pair, which is exactly the population PMI ranks highest."""
    data = from_documents(["alpha beta", "gamma delta"])
    assert ("beta", "gamma") not in data.pairs
    assert set(data.pairs) == {("alpha", "beta"), ("gamma", "delta")}


def test_a_boundary_bigram_would_have_been_created_by_concatenation():
    """The counterfactual, asserted so the guard above cannot be removed silently."""
    joined = count(["alpha", "beta", "gamma", "delta"])
    assert ("beta", "gamma") in joined.pairs


def test_documents_shorter_than_two_tokens_are_skipped():
    data = from_documents(["alpha", "", "beta gamma"])
    assert set(data.pairs) == {("beta", "gamma")}


def test_counts_accumulate_across_documents():
    data = from_documents(["alpha beta", "alpha beta"])
    a, _, _, _ = cell(data, ("alpha", "beta"))
    assert a == 2


def test_document_tables_also_sum_to_the_total():
    data = from_documents(["the cat sat on the mat", "the dog sat on the log"])
    for i in range(len(data)):
        assert data.a[i] + data.b[i] + data.c[i] + data.d[i] == pytest.approx(data.total)


# --- the cutoff ---------------------------------------------------------------------------


def test_filtering_keeps_only_frequent_enough_pairs():
    data = from_documents(["a b a b a b", "c d"])
    filtered = data.filter_by_count(2)
    assert ("c", "d") not in filtered.pairs
    assert ("a", "b") in filtered.pairs


def test_filtering_keeps_the_arrays_aligned():
    data = from_documents(["a b a b a b", "c d", "e f"])
    filtered = data.filter_by_count(2)
    assert len(filtered.pairs) == len(filtered.a) == len(filtered.b)
    assert len(filtered.c) == len(filtered.d) == len(filtered.pairs)


def test_filtering_does_not_change_the_corpus_total():
    """The marginals are properties of the whole corpus. Recomputing them after a cutoff
    would make every score depend on the cutoff twice - once through what is scored, and
    once through what it is scored against."""
    data = from_documents(["a b a b", "c d"])
    assert data.filter_by_count(2).total == data.total


def test_filtering_records_what_it_did():
    filtered = from_documents(["a b a b", "c d"]).filter_by_count(2)
    assert filtered.stats["min_count"] == 2
    assert filtered.stats["bigram_types_kept"] == len(filtered)


def test_a_cutoff_above_everything_leaves_nothing():
    assert len(from_documents(["a b", "c d"]).filter_by_count(99)) == 0


def test_hapax_pairs_dominate_a_realistic_corpus():
    """The fact the whole project turns on: most bigram types occur once, and that is the
    population PMI is most enthusiastic about."""
    docs = [f"word{i} word{i + 1} the cat" for i in range(200)]
    data = from_documents(docs)
    assert data.stats["hapax_share"] > 0.5


def test_arrays_are_float_so_the_measures_do_not_integer_divide():
    data = from_documents(["a b c"])
    for arr in (data.a, data.b, data.c, data.d):
        assert arr.dtype == np.float64
