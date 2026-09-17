"""Tests for tokenisation and the TF-IDF matrix.

No dataset and no network: `load_devign` is not exercised, only the vectoriser every result
depends on.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import tfidf, tokenize

DOCS = [
    "int main ( void ) { return 0 ; }",
    "int main ( void ) { return 1 ; }",
    "void helper ( char * buffer ) { free ( buffer ) ; }",
    "void helper ( char * buffer ) { malloc ( buffer ) ; }",
]


# --- tokenizer ----------------------------------------------------------------------------


def test_identifiers_are_one_token():
    assert tokenize("max_length = 512") == ["max_length", "=", "512"]


def test_punctuation_is_kept():
    assert tokenize("if (x) {") == ["if", "(", "x", ")", "{"]


def test_case_is_preserved():
    assert tokenize("MyStruct myVar") == ["MyStruct", "myVar"]


def test_empty_text_gives_nothing():
    assert tokenize("") == []


# --- the matrix ---------------------------------------------------------------------------


def test_rows_are_unit_length():
    """The `c` in `ltc`. Every downstream step here assumes it - spherical k-means assigns
    by dot product, and the cosine silhouette normalises again for safety."""
    matrix, _ = tfidf(DOCS, min_df=1)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_shape_matches_documents_and_vocabulary():
    matrix, vocabulary = tfidf(DOCS, min_df=1)
    assert matrix.shape == (len(DOCS), len(vocabulary))


def test_rare_terms_are_pruned():
    _, vocabulary = tfidf(DOCS + ["singleton_token_here"], min_df=2)
    assert "singleton_token_here" not in vocabulary


def test_the_feature_budget_is_respected():
    docs = [" ".join(f"tok{i}_{j}" for j in range(50)) for i in range(20)]
    _, vocabulary = tfidf(docs, min_df=1, max_features=25)
    assert len(vocabulary) <= 25


def test_a_term_in_every_document_gets_no_weight():
    """idf of log(N/N) is zero, so a term everything shares cannot separate anything. If it
    carried weight, document length would drive the clustering."""
    matrix, vocabulary = tfidf(DOCS, min_df=1)
    if "(" in vocabulary:
        column = matrix[:, vocabulary.index("(")]
        assert np.allclose(column, 0.0)


def test_term_frequency_is_logarithmic():
    """`1 + log tf`, so a token appearing four times counts about 2.4x one appearing once,
    not 4x.

    The comparison is *within* one row. Comparing the same term across two rows would say
    nothing, because cosine normalisation rescales each row by its own length - two
    documents whose only non-zero feature is the same term both normalise to exactly 1.
    """
    docs = ["alpha alpha alpha alpha gamma", "delta", "epsilon"]
    matrix, vocabulary = tfidf(docs, min_df=1)
    alpha = matrix[0, vocabulary.index("alpha")]
    gamma = matrix[0, vocabulary.index("gamma")]
    # alpha and gamma each appear in exactly one document, so their idf is identical and
    # the ratio is purely the term-frequency transform.
    assert alpha / gamma == pytest.approx(1 + math.log(4))


def test_documents_sharing_vocabulary_are_close():
    matrix, _ = tfidf(DOCS, min_df=1)
    same_family = float(matrix[0] @ matrix[1])
    different = float(matrix[0] @ matrix[2])
    assert same_family > different


def test_an_empty_document_does_not_divide_by_zero():
    matrix, _ = tfidf(["alpha beta", "", "beta gamma"], min_df=1)
    assert np.isfinite(matrix).all()


def test_a_corpus_with_no_surviving_vocabulary_is_empty_not_an_error():
    matrix, vocabulary = tfidf(["alpha", "beta"], min_df=5)
    assert vocabulary == []
    assert matrix.shape[1] == 0


@pytest.mark.parametrize("min_df", [1, 2, 3])
def test_every_pruning_level_keeps_rows_finite(min_df):
    matrix, _ = tfidf(DOCS, min_df=min_df)
    assert np.isfinite(matrix).all()
