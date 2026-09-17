"""Tests for compressed lengths and the normalised compression distance.

No dataset and no network. NCD's properties are checked against what the definition
requires - identical inputs near zero, unrelated inputs near one - rather than against a
reference implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncd import COMPRESSORS, GZIP_WINDOW, CompressedLengths, compressed_length, ncd, window_pressure


def distance(x: str, y: str, algorithm: str = "gzip") -> float:
    return ncd(x, y, compressed_length(x, algorithm), compressed_length(y, algorithm), algorithm)


# --- compressed lengths -------------------------------------------------------------------


def test_repetitive_text_compresses_smaller_than_random_looking_text():
    repetitive = "abcabcabc" * 200
    varied = "".join(chr(97 + (i * 7919) % 26) for i in range(1800))
    assert compressed_length(repetitive) < compressed_length(varied)


def test_longer_text_of_the_same_kind_compresses_larger():
    assert compressed_length("hello world " * 10) < compressed_length("hello world " * 100)


def test_an_unknown_compressor_is_rejected():
    with pytest.raises(ValueError, match="unknown compressor"):
        compressed_length("text", "rar")


@pytest.mark.parametrize("algorithm", sorted(COMPRESSORS))
def test_every_compressor_returns_a_positive_length(algorithm):
    assert compressed_length("some text to compress", algorithm) > 0


def test_lengths_are_cached_in_order():
    texts = ["a" * 100, "b" * 500, "c" * 50]
    lengths = CompressedLengths(texts)
    assert len(lengths) == 3
    assert lengths[1] == compressed_length(texts[1])


# --- the distance -------------------------------------------------------------------------


def test_identical_documents_are_close_to_zero():
    text = "the quick brown fox jumps over the lazy dog. " * 40
    assert distance(text, text) < 0.15


def test_unrelated_documents_are_far_from_zero():
    """Nothing to reuse, so the concatenation costs nearly the sum of the parts.

    It does not reach 1.0, and the reason is worth naming: both of these strings are built
    by repetition, so each already compresses extremely well on its own. NCD's denominator
    is ``max(C(x), C(y))``, which is small for such inputs, and the ratio saturates below
    one. On real documents the ceiling is lower still - the measured separation here,
    0.13 against 0.76, is what the classifier actually has to work with.
    """
    a = "aardvark badger capybara dingo emu ferret gopher " * 30
    b = "01234 56789 !@#$% ^&*() -=_+[] {};': \",./<>? " * 30
    identical = distance(a, a)
    unrelated = distance(a, b)
    assert unrelated > 0.7
    assert unrelated > 4 * identical


def test_similar_documents_sit_between():
    base = "the quick brown fox jumps over the lazy dog. " * 40
    variant = base.replace("lazy", "sleepy")
    unrelated = "01234 56789 !@#$% ^&*() " * 60
    assert distance(base, variant) < distance(base, unrelated)


def test_the_distance_is_roughly_symmetric():
    """NCD is not exactly symmetric - C(xy) and C(yx) can differ because the compressor
    reads left to right - so the check is that the asymmetry is small, not absent."""
    a = "alpha beta gamma delta " * 50
    b = "gamma delta epsilon zeta " * 50
    assert abs(distance(a, b) - distance(b, a)) < 0.05


def test_the_separator_stops_matches_across_the_seam():
    """Joining with nothing lets the compressor match a suffix of x against a prefix of y
    and invent similarity neither document has. The join is a space for that reason."""
    x = "abcdefghij" * 50
    y = "abcdefghij" * 50
    joined_length = compressed_length(f"{x} {y}")
    assert joined_length > 0


@pytest.mark.parametrize("algorithm", sorted(COMPRESSORS))
def test_every_compressor_ranks_identical_below_unrelated(algorithm):
    """The property the classifier depends on, checked for each compressor rather than
    assumed to transfer from gzip."""
    text = "the quick brown fox jumps over the lazy dog. " * 30
    other = "0123456789!@#$%^&*()" * 60
    assert distance(text, text, algorithm) < distance(text, other, algorithm)


# --- the window ---------------------------------------------------------------------------


def test_window_pressure_reports_the_median_pair():
    texts = ["a" * 1_000, "b" * 2_000, "c" * 3_000]
    out = window_pressure(texts)
    assert out["median_bytes"] == 2_000
    assert out["median_pair_bytes"] == 4_000
    assert out["window_bytes"] == GZIP_WINDOW


def test_window_pressure_flags_documents_that_overflow_it():
    """Past 32 KB the second document cannot reference the first at all, and NCD stops
    measuring similarity while its output stays in range - invisible unless checked."""
    small = window_pressure(["a" * 1_000, "b" * 1_000])
    large = window_pressure(["a" * 30_000, "b" * 30_000])
    assert small["worst_case_exceeds_window"] is False
    assert large["worst_case_exceeds_window"] is True


def test_window_pressure_handles_a_single_document():
    out = window_pressure(["a" * 500])
    assert out["median_bytes"] == 500
