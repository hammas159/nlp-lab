"""Tests for the Urdu tokenizer and the title-to-body benchmark.

No corpus and no network. The most important test is the first one: the lab's shared English
tokenizer returns an **empty list** for Urdu rather than raising, so a project that reused it
by habit would have computed every number on empty documents and reported them without a
single error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parents[2]))

from shared.benchmark import tokenize as english_tokenize

import urdu as U

URDU = "اردو ویکیپیڈیا ایک آزاد دائرۃ المعارف ہے"
MIXED = "اردو Wikipedia 2023 میں"


# --- the failure this project had to avoid ---------------------------------------------------


def test_the_english_tokenizer_silently_returns_nothing_for_urdu():
    """It matches `[a-z0-9]+`, so Urdu text yields zero tokens - and no exception.

    Reusing it here would have produced an index of empty documents, a BM25 that scored
    everything at zero, and a results table full of plausible-looking numbers. The lab's
    other projects all share that tokenizer; this is the one where sharing it breaks.
    """
    assert english_tokenize(URDU) == []


def test_the_urdu_tokenizer_returns_words():
    assert len(U.tokenize(URDU)) == 7


# --- the tokenizer ----------------------------------------------------------------------------


def test_perso_arabic_runs_are_kept_whole():
    assert "اردو" in U.tokenize(URDU)


def test_latin_and_digits_survive():
    """Urdu Wikipedia is full of names, dates and units in Latin script and ASCII digits.
    Dropping them would make the corpus easier than it is."""
    tokens = U.tokenize(MIXED)
    assert "wikipedia" in tokens
    assert "2023" in tokens


def test_latin_is_lowercased():
    assert U.tokenize("Wikipedia WIKIPEDIA") == ["wikipedia", "wikipedia"]


def test_punctuation_is_dropped():
    assert U.tokenize("اردو، ویکیپیڈیا۔") == ["اردو", "ویکیپیڈیا"]


def test_empty_text_gives_no_tokens():
    assert U.tokenize("") == []


def test_whitespace_only_gives_no_tokens():
    assert U.tokenize("   \n\t ") == []


def test_urdu_digits_are_kept():
    """Urdu uses its own digit codepoints (U+06F0-U+06F9), which fall inside the Arabic
    block and must not be silently dropped."""
    assert U.tokenize("۱۲۳") == ["۱۲۳"]


# --- the benchmark construction ----------------------------------------------------------------


def test_the_title_is_removed_from_the_body():
    """The query is the title. Leaving it in the body lets every retriever win by matching
    the query against a copy of itself, and every tokenizer would score near 1.0 - hiding
    exactly the differences this project measures.
    """
    body = U.strip_title("اردو ویکیپیڈیا ایک آزاد دائرۃ المعارف ہے", "اردو ویکیپیڈیا")
    assert "اردو ویکیپیڈیا" not in body


def test_stripping_leaves_the_rest_of_the_body():
    body = U.strip_title("اردو ویکیپیڈیا ایک آزاد", "اردو ویکیپیڈیا")
    assert "آزاد" in body


def test_stripping_an_empty_title_is_a_no_op():
    assert U.strip_title("کچھ متن", "") == "کچھ متن"


def test_stripping_a_title_that_is_absent_changes_nothing():
    assert U.strip_title("کچھ متن", "غائب") == "کچھ متن"


def test_a_missing_corpus_raises_with_the_fetch_command(tmp_path, monkeypatch):
    monkeypatch.setenv("URDU_ARTICLES", str(tmp_path / "nothing.parquet"))
    with pytest.raises(LookupError, match="fetch.py"):
        U.build(limit=5)


def test_an_explicit_corpus_path_is_used(tmp_path, monkeypatch):
    """The override exists so a test can point at its own corpus instead of whatever the
    machine happens to have - the mistake project 20 shipped to CI."""
    target = tmp_path / "articles.parquet"
    target.write_bytes(b"not really parquet")
    monkeypatch.setenv("URDU_ARTICLES", str(target))
    assert U.find_parquet() == target
