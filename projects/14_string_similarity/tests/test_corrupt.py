"""Tests for the two error models.

No dataset and no network. The models are the independent variable of the whole project, so
what matters is that each one produces the *kind* of corruption it claims and not the other
kind — otherwise the comparison measures nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corrupt import MODELS, corrupt, phonetic, typing


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng([seed, 0xC0FFEE])


# --- typing -------------------------------------------------------------------------------


def test_typing_changes_the_word():
    changed = sum(typing("computer", rng(s)) != "computer" for s in range(20))
    assert changed >= 18


def test_typing_makes_a_small_change():
    """One edit, so the result stays within one character of the original length."""
    for seed in range(30):
        out = typing("information", rng(seed))
        assert abs(len(out) - len("information")) <= 1


def test_typing_is_deterministic_given_a_seed():
    assert typing("example", rng(3)) == typing("example", rng(3))


def test_typing_leaves_very_short_words_alone():
    """A one-character word has no adjacent pair to transpose and nothing to delete without
    emptying it."""
    assert typing("a", rng(0)) == "a"


def test_more_edits_change_more():
    one = typing("encyclopedia", rng(5), edits=1)
    many = typing("encyclopedia", rng(5), edits=4)
    assert one != many


# --- phonetic -----------------------------------------------------------------------------


def test_phonetic_rewrites_a_known_digraph():
    outputs = {phonetic("photograph", rng(s)) for s in range(30)}
    assert any("f" in out for out in outputs)


def test_phonetic_preserves_length_roughly():
    for seed in range(20):
        out = phonetic("telephone", rng(seed))
        assert abs(len(out) - len("telephone")) <= 2


def test_phonetic_leaves_a_word_with_no_applicable_rule_alone():
    """Returning the word unchanged is the honest behaviour - corrupting it with a rule
    that does not fit would make the two models corrupt the same words, which is the thing
    being varied."""
    assert phonetic("bldg", rng(0)) == "bldg"


def test_phonetic_is_deterministic_given_a_seed():
    assert phonetic("september", rng(7)) == phonetic("september", rng(7))


def test_phonetic_never_returns_an_empty_string():
    """Several rules delete characters; applied to a short word they could empty it, and an
    empty query would be scored against every candidate equally."""
    for seed in range(30):
        assert phonetic("ee", rng(seed), edits=3) != ""


# --- the two models are different ---------------------------------------------------------


def test_the_two_models_produce_different_corruptions():
    """If they agreed there would be nothing to compare."""
    word = "philosophy"
    typed = {typing(word, rng(s)) for s in range(25)}
    sounded = {phonetic(word, rng(s)) for s in range(25)}
    assert typed != sounded


def test_phonetic_corruptions_still_sound_alike():
    """The defining property: `ph` becomes `f`, which a phonetic code should map to the
    same class, while a keyboard slip would not."""
    outputs = {phonetic("phone", rng(s)) for s in range(30)}
    assert any(out in {"fone", "phon", "phne"} or "f" in out for out in outputs)


# --- the dispatcher -----------------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(MODELS))
def test_every_model_returns_a_string(model):
    assert isinstance(corrupt(model, "example", rng(0)), str)


@pytest.mark.parametrize("model", sorted(MODELS))
def test_every_model_handles_a_short_word(model):
    assert isinstance(corrupt(model, "ab", rng(0)), str)


def test_an_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown error model"):
        corrupt("cosmic_ray", "example", rng(0))
