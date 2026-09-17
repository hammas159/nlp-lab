"""Tests for the three topic models behind one interface.

No dataset and no network. The point of the shared interface is that the comparison varies
the model and nothing else, so these check that the interface really is shared - same
shapes, same contract - and that each model's defining property survives the wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from models import LDA, LSA, MODELS, NMF, unit

DOCS = [
    "cat dog pet animal fur",
    "dog cat animal pet paw",
    "quantum particle physics entangled",
    "physics quantum wave particle",
    "cat pet fur animal paw",
    "wave physics entangled quantum",
]


def matrix(counts: bool = False):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    cls = CountVectorizer if counts else TfidfVectorizer
    vectorizer = cls(min_df=1)
    return vectorizer.fit_transform(DOCS), list(vectorizer.get_feature_names_out())


# --- the shared interface -----------------------------------------------------------------


@pytest.mark.parametrize("model_cls", MODELS)
def test_every_model_exposes_the_same_shapes(model_cls):
    data, terms = matrix(counts=model_cls.needs == "counts")
    model = model_cls(3, seed=0).fit(data)
    assert model.components.shape == (3, len(terms))
    assert model.transform(data).shape == (len(DOCS), 3)


@pytest.mark.parametrize("model_cls", MODELS)
def test_every_model_returns_the_requested_number_of_top_words(model_cls):
    data, terms = matrix(counts=model_cls.needs == "counts")
    topics = model_cls(3, seed=0).fit(data).top_words(terms, n=4)
    assert len(topics) == 3
    assert all(len(t) == 4 for t in topics)


@pytest.mark.parametrize("model_cls", MODELS)
def test_every_model_declares_the_representation_it_needs(model_cls):
    assert model_cls.needs in {"tfidf", "counts"}


@pytest.mark.parametrize("model_cls", MODELS)
def test_asking_for_more_topics_than_the_data_supports_does_not_raise(model_cls):
    """TruncatedSVD raises rather than clamping when n_components exceeds the rank, which
    crashed project 01 on a small corpus. Every model here clamps instead."""
    data, _ = matrix(counts=model_cls.needs == "counts")
    model = model_cls(500, seed=0).fit(data)
    assert model.components.shape[0] < 500


@pytest.mark.parametrize("model_cls", MODELS)
def test_every_model_separates_two_obvious_themes(model_cls):
    """The corpus is three pet documents and three physics documents. A model that cannot
    tell them apart is not fit to be compared on anything subtler."""
    data, _ = matrix(counts=model_cls.needs == "counts")
    vectors = unit(model_cls(2, seed=0).fit(data).transform(data))
    pets, physics = [0, 1, 4], [2, 3, 5]
    within = np.mean([vectors[i] @ vectors[j] for i in pets for j in pets if i != j])
    across = np.mean([vectors[i] @ vectors[j] for i in pets for j in physics])
    assert within > across


# --- each model's defining property -------------------------------------------------------


def test_nmf_components_are_non_negative():
    """The whole difference between NMF and LSA. If this fails the two are the same model."""
    data, _ = matrix()
    assert (NMF(3, seed=0).fit(data).components >= 0).all()


def test_lsa_components_are_not_all_non_negative():
    """LSA's freedom to go negative is what lets it mean 'these words, not those' - and
    what makes its topics harder to read as word lists."""
    data, _ = matrix()
    assert (LSA(3, seed=0).fit(data).components < 0).any()


def test_lda_document_topics_are_a_distribution():
    """LDA is the only one of the three that is probabilistic; its document vectors are a
    mixture over topics and must sum to one."""
    data, _ = matrix(counts=True)
    vectors = LDA(3, seed=0).fit(data).transform(data)
    assert np.allclose(vectors.sum(axis=1), 1.0)


def test_lda_is_reproducible_at_a_fixed_seed():
    """LDA is the only stochastic model here, so a comparison against it is only meaningful
    if the seed is pinned."""
    data, _ = matrix(counts=True)
    a = LDA(3, seed=7).fit(data).components
    b = LDA(3, seed=7).fit(data).components
    assert np.allclose(a, b)


# --- the cosine helper --------------------------------------------------------------------


def test_unit_rows_have_norm_one():
    assert np.allclose(np.linalg.norm(unit(np.array([[3.0, 4.0], [1.0, 0.0]])), axis=1), 1.0)


def test_unit_survives_an_all_zero_row():
    """A document with no in-vocabulary terms produces a zero vector; normalising it must
    not divide by zero and silently return NaN scores."""
    assert np.isfinite(unit(np.array([[0.0, 0.0], [1.0, 1.0]]))).all()


def test_unit_handles_negative_entries():
    """LSA vectors are signed, so the helper cannot assume non-negativity."""
    assert np.allclose(np.linalg.norm(unit(np.array([[-3.0, 4.0]])), axis=1), 1.0)
