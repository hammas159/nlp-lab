"""Tests for the SMART weighting components, against the definitions rather than against
each other.

Every formula here is short enough to evaluate by hand, and a weighting scheme that is
subtly wrong still produces a ranked list and a plausible recall figure. So each variant is
checked against arithmetic written out in the test, not against another variant.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smart import (
    DF_CODES,
    NORM_CODES,
    TF_CODES,
    all_schemes,
    document_frequency,
    normalise,
    term_frequency,
    weight,
)


def counts() -> sparse.csr_matrix:
    """Three documents, four terms. Worked through by hand in the assertions below.

    d0: t0 x3, t1 x1
    d1: t1 x2, t2 x4
    d2: t0 x1, t3 x1
    """
    return sparse.csr_matrix(
        np.array(
            [
                [3.0, 1.0, 0.0, 0.0],
                [0.0, 2.0, 4.0, 0.0],
                [1.0, 0.0, 0.0, 1.0],
            ]
        )
    )


def dense(matrix) -> np.ndarray:
    return np.asarray(matrix.todense())


# --- term frequency -----------------------------------------------------------------------


def test_natural_is_the_raw_count():
    assert np.allclose(dense(term_frequency(counts(), "n")), dense(counts()))


def test_boolean_flattens_every_count_to_one():
    out = dense(term_frequency(counts(), "b"))
    assert set(np.unique(out).tolist()) == {0.0, 1.0}
    assert out[0, 0] == 1.0 and out[1, 2] == 1.0


def test_logarithm_matches_the_definition():
    out = dense(term_frequency(counts(), "l"))
    assert out[0, 0] == pytest.approx(1 + math.log(3))
    assert out[1, 2] == pytest.approx(1 + math.log(4))


def test_a_count_of_one_is_unchanged_by_the_logarithm():
    """1 + log 1 = 1. A variant that shifted this would change every singleton in the
    collection, which is most of it."""
    assert dense(term_frequency(counts(), "l"))[0, 1] == pytest.approx(1.0)


def test_augmented_uses_the_row_maximum():
    out = dense(term_frequency(counts(), "a"))
    assert out[0, 0] == pytest.approx(0.5 + 0.5 * 3 / 3)  # d0 max is 3
    assert out[0, 1] == pytest.approx(0.5 + 0.5 * 1 / 3)
    assert out[1, 2] == pytest.approx(0.5 + 0.5 * 4 / 4)


def test_augmented_never_leaves_the_half_to_one_band():
    out = term_frequency(counts(), "a")
    assert out.data.min() >= 0.5 - 1e-12
    assert out.data.max() <= 1.0 + 1e-12


def test_log_average_divides_by_the_average_of_present_terms_only():
    """The average is over the terms the document actually contains. Averaging over the
    whole vocabulary would make it nearly zero for every document and the denominator
    meaningless."""
    out = dense(term_frequency(counts(), "L"))
    d1_average = (2 + 4) / 2
    assert out[1, 2] == pytest.approx((1 + math.log(4)) / (1 + math.log(d1_average)))


def test_an_unknown_term_frequency_code_is_rejected():
    with pytest.raises(ValueError, match="unknown term-frequency code"):
        term_frequency(counts(), "z")


@pytest.mark.parametrize("code", TF_CODES)
def test_every_tf_variant_preserves_sparsity(code):
    """A variant that made absent terms nonzero would densify a 66,581 x 160,743 matrix."""
    assert term_frequency(counts(), code).nnz == counts().nnz


# --- document frequency -------------------------------------------------------------------


def test_no_document_frequency_component_is_all_ones():
    assert np.allclose(document_frequency(counts(), "n"), np.ones(4))


def test_idf_matches_the_definition():
    idf = document_frequency(counts(), "t")
    # t0 appears in 2 of 3 documents, t2 in 1 of 3.
    assert idf[0] == pytest.approx(math.log(3 / 2))
    assert idf[2] == pytest.approx(math.log(3 / 1))


def test_idf_is_larger_for_rarer_terms():
    idf = document_frequency(counts(), "t")
    assert idf[2] > idf[0]


def test_probabilistic_idf_is_clamped_at_zero():
    """Robertson's form goes negative for a term in more than half the collection. Without
    the clamp those terms would actively penalise the documents containing them."""
    idf = document_frequency(counts(), "p")
    assert (idf >= 0).all()
    assert idf[0] == 0.0  # t0 is in 2 of 3 documents, so log((3-2)/2) < 0


def test_an_unknown_document_frequency_code_is_rejected():
    with pytest.raises(ValueError, match="unknown document-frequency code"):
        document_frequency(counts(), "z")


# --- normalisation ------------------------------------------------------------------------


def test_no_normalisation_leaves_the_matrix_alone():
    assert np.allclose(dense(normalise(counts(), "n")), dense(counts()))


def test_cosine_gives_unit_rows():
    out = dense(normalise(counts(), "c"))
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_cosine_removes_the_length_advantage():
    """Two documents with the same term proportions must score identically however long
    they are. That is the entire purpose of the c component."""
    short = sparse.csr_matrix(np.array([[1.0, 2.0]]))
    long = sparse.csr_matrix(np.array([[10.0, 20.0]]))
    assert np.allclose(dense(normalise(short, "c")), dense(normalise(long, "c")))


def test_pivoted_normalisation_uses_the_count_of_distinct_terms():
    out = dense(normalise(counts(), "u"))
    unique = np.array([2.0, 2.0, 2.0])
    pivot = unique.mean()
    expected = counts().toarray()[0] / ((1 - 0.2) * pivot + 0.2 * unique[0])
    assert np.allclose(out[0], expected)


def test_pivoted_normalisation_is_gentler_than_cosine_on_long_documents():
    """Pivoted length normalisation exists because cosine over-penalises long documents.
    A document with many distinct terms should keep more of its weight under u than c."""
    matrix = sparse.csr_matrix(np.array([[1.0] * 10 + [0.0] * 10, [1.0] + [0.0] * 19]))
    cosine = dense(normalise(matrix, "c"))
    pivoted = dense(normalise(matrix, "u"))
    assert pivoted[0].sum() / pivoted[1].sum() > cosine[0].sum() / cosine[1].sum()


def test_a_zero_row_does_not_divide_by_zero():
    matrix = sparse.csr_matrix(np.array([[0.0, 0.0], [1.0, 1.0]]))
    for code in NORM_CODES:
        assert np.isfinite(dense(normalise(matrix, code))).all()


def test_an_unknown_normalisation_code_is_rejected():
    with pytest.raises(ValueError, match="unknown normalisation code"):
        normalise(counts(), "z")


# --- the three-letter codes ---------------------------------------------------------------


def test_the_grid_is_forty_five_schemes():
    schemes = all_schemes()
    assert len(schemes) == len(TF_CODES) * len(DF_CODES) * len(NORM_CODES) == 45
    assert len(set(schemes)) == 45


def test_the_textbook_scheme_is_in_the_grid():
    assert "lnc" in all_schemes()
    assert "ltc" in all_schemes()


def test_a_code_must_be_three_letters():
    with pytest.raises(ValueError, match="three letters"):
        weight(counts(), "lc")


def test_nnn_is_the_raw_count_matrix():
    """The identity element of the grid: no term weighting, no collection factor, no
    normalisation."""
    assert np.allclose(dense(weight(counts(), "nnn")), dense(counts()))


def test_queries_are_weighted_with_document_collection_statistics():
    """An idf computed over the queries is estimated from a handful of short texts and
    leaks the query distribution into the weighting. `weight` takes the document idf."""
    documents = counts()
    idfs = {code: document_frequency(documents, code) for code in DF_CODES}
    queries = sparse.csr_matrix(np.array([[1.0, 0.0, 1.0, 0.0]]))

    with_document_idf = dense(weight(queries, "ltn", idf=idfs))
    with_query_idf = dense(weight(queries, "ltn"))
    assert not np.allclose(with_document_idf, with_query_idf)
    assert with_document_idf[0, 2] == pytest.approx(1.0 * math.log(3 / 1))


@pytest.mark.parametrize("scheme", all_schemes())
def test_every_scheme_produces_finite_weights(scheme):
    out = weight(counts(), scheme)
    assert np.isfinite(out.data).all()
    assert out.shape == counts().shape
