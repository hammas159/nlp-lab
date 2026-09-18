"""Tests for the five pooling functions.

No model, no dataset, no network: pooling takes an array of token vectors and a mask, which
is all these tests supply. That is the whole point - the pooling step is arithmetic, and it
can be wrong in ways that look like a result.

The masking tests matter most. A pooling that averages over padding scores a short document
differently depending on what else was in its batch, which is not a property of the pooling
and would show up as one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pooling import POOLINGS, pool, token_idf, unit

# Two sequences of three tokens in two dimensions. The second is padded to length three.
HIDDEN = np.array(
    [
        [[1.0, 0.0], [3.0, 2.0], [5.0, -4.0]],
        [[2.0, 2.0], [4.0, 6.0], [99.0, 99.0]],
    ]
)
MASK = np.array([[1, 1, 1], [1, 1, 0]])


# --- each pooling does what its name says ----------------------------------------------------


def test_cls_takes_the_first_token():
    assert pool("cls", HIDDEN, MASK).tolist() == [[1.0, 0.0], [2.0, 2.0]]


def test_mean_averages_real_tokens_only():
    """The second sequence must average two tokens, not three. Including the pad would drag
    it towards 99."""
    out = pool("mean", HIDDEN, MASK)
    assert out[1].tolist() == [3.0, 4.0]


def test_max_takes_the_largest_real_value():
    out = pool("max", HIDDEN, MASK)
    assert out[1].tolist() == [4.0, 6.0]


def test_last_takes_the_final_real_token():
    """Not the final *position* - the final unmasked one. Taking position -1 would return
    the padding vector for every sequence shorter than the batch maximum."""
    out = pool("last", HIDDEN, MASK)
    assert out[0].tolist() == [5.0, -4.0]
    assert out[1].tolist() == [4.0, 6.0]


# --- masking, which is where the silent bugs live --------------------------------------------


@pytest.mark.parametrize("name", sorted(POOLINGS))
def test_padding_never_reaches_the_output(name):
    """The strongest form: the pad vector is 99, far outside the real range. If any pooling
    lets it through, the result moves visibly."""
    out = pool(name, HIDDEN, MASK)
    assert np.all(np.abs(out[1]) < 50)


@pytest.mark.parametrize("name", sorted(POOLINGS))
def test_extra_padding_does_not_change_the_answer(name):
    """A sequence must pool identically whether or not other batch members were longer.
    Otherwise the score depends on batch composition, which no one would think to check."""
    padded = np.concatenate([HIDDEN, np.full((2, 2, 2), 7.0)], axis=1)
    mask = np.concatenate([MASK, np.zeros((2, 2), dtype=int)], axis=1)
    assert np.allclose(pool(name, HIDDEN, MASK), pool(name, padded, mask))


def test_max_uses_a_negative_sentinel_not_zero():
    """With a zero sentinel, a dimension whose real values are all negative would take 0
    from a pad position - so an all-negative feature would read as 0 exactly when the batch
    happened to contain padding."""
    hidden = np.array([[[-5.0], [-3.0], [0.0]]])
    mask = np.array([[1, 1, 0]])
    assert pool("max", hidden, mask).tolist() == [[-3.0]]


# --- idf weighting ---------------------------------------------------------------------------


def test_idf_mean_weights_tokens():
    weights = np.array([[1.0, 3.0, 0.0], [1.0, 1.0, 0.0]])
    out = pool("idf_mean", HIDDEN, MASK, weights)
    expected = (1.0 * np.array([1.0, 0.0]) + 3.0 * np.array([3.0, 2.0])) / 4.0
    assert np.allclose(out[0], expected)


def test_idf_mean_without_weights_is_the_plain_mean():
    """A missing weight table must not silently produce a different pooling under this
    name."""
    assert np.allclose(pool("idf_mean", HIDDEN, MASK, None), pool("mean", HIDDEN, MASK))


def test_idf_mean_ignores_masked_positions_even_if_weighted():
    weights = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 50.0]])
    out = pool("idf_mean", HIDDEN, MASK, weights)
    assert np.all(np.abs(out[1]) < 50)


def test_idf_is_higher_for_rarer_tokens():
    """The property the weighting depends on."""
    idf = token_idf([[1, 2], [1, 3], [1, 4]], vocabulary_size=5)
    assert idf[2] > idf[1]


def test_idf_of_an_unseen_token_is_the_maximum():
    idf = token_idf([[1], [1]], vocabulary_size=4)
    assert idf[3] >= idf[1]


def test_a_zero_weight_row_does_not_divide_by_zero():
    out = pool("idf_mean", HIDDEN, MASK, np.zeros_like(MASK, dtype=float))
    assert np.isfinite(out).all()


# --- the dispatcher and the cosine helper -----------------------------------------------------


@pytest.mark.parametrize("name", sorted(POOLINGS))
def test_every_pooling_returns_one_vector_per_sequence(name):
    out = pool(name, HIDDEN, MASK)
    assert out.shape == (HIDDEN.shape[0], HIDDEN.shape[2])


@pytest.mark.parametrize("name", sorted(POOLINGS))
def test_every_pooling_is_finite(name):
    assert np.isfinite(pool(name, HIDDEN, MASK)).all()


def test_an_unknown_pooling_is_rejected():
    with pytest.raises(ValueError, match="unknown pooling"):
        pool("attention", HIDDEN, MASK)


def test_a_single_token_sequence_pools_to_that_token():
    """Every pooling must agree here - there is only one vector to choose."""
    hidden = np.array([[[2.0, 5.0]]])
    mask = np.array([[1]])
    for name in POOLINGS:
        assert pool(name, hidden, mask).tolist() == [[2.0, 5.0]]


def test_unit_rows_have_norm_one():
    assert np.allclose(np.linalg.norm(unit(np.array([[3.0, 4.0]])), axis=1), 1.0)


def test_unit_survives_an_all_zero_row():
    assert np.isfinite(unit(np.array([[0.0, 0.0], [1.0, 1.0]]))).all()


# --- the trap that "last" hides ---------------------------------------------------------------


def test_last_returns_the_final_real_position_which_is_sep_on_an_encoder():
    """`last` means the last *unmasked position*, and on a BERT-style encoder that position
    holds `[SEP]`, not the final content word.

    The sequence is `[CLS] w1 w2 [SEP]` plus padding, so "last-token pooling" reads a
    special token. On `bge-small-en-v1.5` that token's vector is numerically identical to
    `[CLS]` - cosine 1.000000, largest absolute difference 5e-5 across 384 dimensions - so
    `last` is not a fifth pooling at all there. It is `cls` under another name.

    Nothing about the arithmetic is wrong; the name is what misleads. A decoder-style model
    with no suffix token would give `last` a genuinely different meaning.
    """
    # positions:      [CLS]      w1         w2        [SEP]      pad
    hidden = np.array([[[0.0], [1.0], [2.0], [9.0], [-99.0]]])
    mask = np.array([[1, 1, 1, 1, 0]])
    assert pool("last", hidden, mask).tolist() == [[9.0]]  # the [SEP] slot, not w2


def test_last_and_cls_coincide_when_the_boundary_vectors_match():
    """The degenerate case the model above actually exhibits: if the first and final real
    positions hold the same vector, two 'different' poolings return the same thing and a
    results table shows them tied for reasons that have nothing to do with pooling."""
    hidden = np.array([[[4.0, 1.0], [0.0, 0.0], [4.0, 1.0]]])
    mask = np.array([[1, 1, 1]])
    assert pool("last", hidden, mask).tolist() == pool("cls", hidden, mask).tolist()
