"""Tests for tokenization and the Corpus container.

No dataset and no network: the loaders are not exercised here, only the two tokenizers and
the truncation that every cross-register comparison depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corpora import Corpus, code_tokenize, tokenize

# --- the shared tokenizer -----------------------------------------------------------------


def test_word_tokenizer_case_folds():
    assert tokenize("The THE the") == ["the", "the", "the"]


def test_word_tokenizer_keeps_underscores():
    """Identifiers are one token, not two. `max_length` in code must not become `max` and
    `length`, or the code corpora would be handed a vocabulary the language does not have."""
    assert tokenize("max_length = 512") == ["max_length", "512"]


def test_word_tokenizer_drops_punctuation():
    assert tokenize("if (x) { y++; }") == ["if", "x", "y"]


def test_word_tokenizer_on_empty_input():
    assert tokenize("") == []


def test_the_same_tokenizer_applies_to_prose_and_code():
    """The cardinal rule of this lab: a comparison where one side gets better preprocessing
    is measuring the preprocessing. Both registers go through one function."""
    assert tokenize("The cat sat.") == ["the", "cat", "sat"]
    assert tokenize("int main(void)") == ["int", "main", "void"]


# --- the code-aware tokenizer (sensitivity only) ------------------------------------------


def test_code_tokenizer_keeps_structure():
    assert code_tokenize("if (x) { y; }") == ["if", "(", "x", ")", "{", "y", ";", "}"]


def test_code_tokenizer_preserves_case():
    assert code_tokenize("MyClass myVar") == ["MyClass", "myVar"]


def test_code_tokenizer_separates_numbers_from_identifiers():
    assert code_tokenize("x42 = 42") == ["x42", "=", "42"]


def test_the_two_tokenizers_disagree_on_the_same_text():
    """They must, or section 7 would have nothing to measure."""
    text = "for (int i = 0; i < n; i++) { total += a[i]; }"
    assert len(code_tokenize(text)) > len(tokenize(text))


# --- the container ------------------------------------------------------------------------


def test_corpus_reports_length_and_types():
    corpus = Corpus("x", "test", ["a", "b", "a", "c"])
    assert len(corpus) == 4
    assert corpus.types == 3


def test_truncate_takes_a_prefix_not_a_sample():
    """Heaps' law is measured over a prefix of the stream. Sampling instead would destroy
    the ordering the growth curve is made of."""
    corpus = Corpus("x", "test", list("abcdefghij"))
    assert corpus.truncate(4).tokens == list("abcd")


def test_truncate_beyond_the_end_is_the_whole_corpus():
    corpus = Corpus("x", "test", list("abc"))
    assert len(corpus.truncate(100)) == 3


def test_truncate_keeps_the_labels():
    corpus = Corpus("HotpotQA", "English prose", list("abcdef"))
    clipped = corpus.truncate(2)
    assert clipped.name == "HotpotQA"
    assert clipped.register == "English prose"


def test_type_token_ratio_falls_with_length():
    """The reason a type-token ratio cannot be compared across corpora of different sizes,
    and the reason every comparison here is made at a matched token count."""
    tokens = [f"w{i % 100}" for i in range(10_000)]
    short = Corpus("x", "test", tokens[:200])
    long = Corpus("x", "test", tokens)
    assert short.types / len(short) > long.types / len(long)


def test_empty_corpus_has_no_types():
    assert Corpus("x", "test", []).types == 0
    with pytest.raises(ZeroDivisionError):
        _ = Corpus("x", "test", []).types / len(Corpus("x", "test", []))
