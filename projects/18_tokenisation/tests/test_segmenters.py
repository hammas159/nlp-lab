"""Tests for the three subword algorithms and the intrinsic measurements.

No dataset and no network: the tokenizers are trained here, on a hand-built corpus, in
milliseconds. That is the point of subword training - it needs no pretrained artefact.

The comparison in this project is only meaningful if the three tokenizers really are
configured identically, so several of these pin exactly that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from segmenters import (
    TRAINERS,
    fertility,
    intact_rate,
    segmentation_agreement,
    strip_marker,
    train,
    unknown_rate,
)

CORPUS = [
    "the cat sat on the mat and the cat was happy",
    "unhappiness and happiness are both about being happy",
    "running runner runs ran and running again",
    "the quantum particle was entangled with another particle",
    "microbiology microbiologist microbiological studies",
] * 20

VOCAB = 300
NAMES = sorted(TRAINERS)


@pytest.fixture(scope="module")
def segmenters():
    return {name: train(name, CORPUS, VOCAB) for name in NAMES}


# --- the shared contract -------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_every_tokenizer_produces_pieces(name, segmenters):
    assert len(segmenters[name].encode("the cat sat")) > 0


@pytest.mark.parametrize("name", NAMES)
def test_every_tokenizer_respects_the_vocabulary_budget(name, segmenters):
    """A comparison where one algorithm got a bigger vocabulary would be measuring the
    budget, not the algorithm."""
    assert segmenters[name].vocabulary_size <= VOCAB


@pytest.mark.parametrize("name", NAMES)
def test_every_tokenizer_lowercases(name, segmenters):
    """Same normaliser for all three, or the comparison includes a casing difference."""
    assert segmenters[name].encode("CAT") == segmenters[name].encode("cat")


@pytest.mark.parametrize("name", NAMES)
def test_every_tokenizer_handles_an_unseen_word(name, segmenters):
    """Subword vocabularies exist so that nothing is out-of-vocabulary. A word never seen
    in training must still come back as pieces rather than as nothing."""
    assert len(segmenters[name].encode("zygomorphic")) > 0


@pytest.mark.parametrize("name", NAMES)
def test_every_tokenizer_handles_empty_input(name, segmenters):
    assert segmenters[name].encode("") == []


def test_an_unknown_algorithm_is_rejected():
    with pytest.raises(ValueError, match="unknown tokenizer"):
        train("SentencePieceIsh", CORPUS, VOCAB)


# --- the continuation marker ---------------------------------------------------------------


def test_strip_marker_removes_the_wordpiece_prefix():
    assert strip_marker("##ing") == "ing"


def test_strip_marker_leaves_other_pieces_alone():
    assert strip_marker("ing") == "ing"
    assert strip_marker("#hash") == "#hash"


def test_agreement_is_not_defeated_by_the_marker():
    """WordPiece writes continuations as `##ing` and BPE as `ing`. Comparing raw strings
    would score two identical segmentations as different, making the agreement number
    meaninglessly low - which is exactly the kind of bug that produces a striking result."""

    class Fake:
        def __init__(self, pieces):
            self.pieces = pieces

        def encode(self, _text):
            return self.pieces

    assert (
        segmentation_agreement(Fake(["un", "##happy"]), Fake(["un", "happy"]), ["unhappy"]) == 1.0
    )


# --- intrinsic measurements ----------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_fertility_is_at_least_one(name, segmenters):
    """A word cannot become fewer than one piece."""
    assert fertility(segmenters[name], ["happy", "unhappiness", "zygomorphic"]) >= 1.0


@pytest.mark.parametrize("name", NAMES)
def test_a_frequent_word_is_kept_whole(name, segmenters):
    """`the` appears in nearly every training sentence; any of these algorithms should keep
    it as one piece. If not, the training did not take."""
    assert len(segmenters[name].encode("the")) == 1


# `zygomorphic` contains a `z`, which never occurs in CORPUS. That makes it the sharpest
# available test: an unseen word containing an unseen character. The three algorithms
# handle it three different ways, and the difference is not cosmetic - it decides how much
# of a rare word survives into a retrieval index.


def test_unigram_falls_back_all_the_way_to_characters():
    """Unigram keeps a probabilistic model over segmentations with full character coverage,
    so even the unseen `z` comes back as itself. Nothing is lost."""
    pieces = train("Unigram", CORPUS, VOCAB).encode("zygomorphic")
    assert "[UNK]" not in pieces
    assert "".join(pieces) == "zygomorphic"


def test_bpe_loses_only_the_character_it_never_saw():
    """BPE degrades per *character*: the unseen `z` becomes `[UNK]` and the remaining nine
    pieces survive, so most of the word is still matchable."""
    pieces = train("BPE", CORPUS, VOCAB).encode("zygomorphic")
    assert pieces.count("[UNK]") == 1
    assert "".join(p for p in pieces if p != "[UNK]") == "ygomorphic"


def test_wordpiece_loses_the_entire_word():
    """WordPiece degrades per *word*. It matches the longest prefix in its vocabulary and
    repeats on the rest; if any step fails it discards the whole word for one `[UNK]`.

    This is the single biggest behavioural difference between the three, and it is why
    `run.py` measures `unknown_rate` - without it, a retrieval gap would be blamed on the
    vocabulary WordPiece learned rather than on what it does when that vocabulary misses.
    """
    assert train("WordPiece", CORPUS, VOCAB).encode("zygomorphic") == ["[UNK]"]


@pytest.mark.parametrize("name", NAMES)
def test_intact_rate_is_a_share(name, segmenters):
    assert 0.0 <= intact_rate(segmenters[name], ["the", "cat", "zygomorphic"]) <= 1.0


def test_fertility_and_intact_rate_disagree_about_the_same_tokenizer():
    """They are not the same measurement wearing two names: fertility counts pieces,
    intact rate counts words. A tokenizer that splits a few words into many parts and leaves
    the rest whole scores high on both."""

    class Fake:
        def encode(self, text):
            return ["a", "b", "c"] if text == "long" else ["x"]

    words = ["long", "short", "short", "short"]
    assert fertility(Fake(), words) == pytest.approx(1.5)
    assert intact_rate(Fake(), words) == pytest.approx(0.75)


def test_empty_word_lists_do_not_divide_by_zero():
    class Fake:
        def encode(self, text):
            return ["x"]

    assert fertility(Fake(), []) == 0.0
    assert intact_rate(Fake(), []) == 0.0
    assert segmentation_agreement(Fake(), Fake(), []) == 0.0


def test_a_tokenizer_agrees_completely_with_itself():
    """The upper bound. If this is not 1.0 the agreement measure is broken."""
    one = train("BPE", CORPUS, VOCAB)
    assert segmentation_agreement(one, one, ["happy", "unhappiness", "running"]) == 1.0


@pytest.mark.parametrize("name", NAMES)
def test_unknown_rate_is_a_share(name, segmenters):
    assert 0.0 <= unknown_rate(segmenters[name], ["the", "cat", "zygomorphic"]) <= 1.0


def test_unknown_rate_counts_words_not_pieces():
    class Fake:
        def encode(self, text):
            return ["[UNK]"] if text == "rare" else ["x", "y"]

    assert unknown_rate(Fake(), ["rare", "common", "common", "common"]) == pytest.approx(0.25)


def test_unknown_rate_is_zero_when_nothing_is_unknown():
    class Fake:
        def encode(self, text):
            return ["x"]

    assert unknown_rate(Fake(), ["a", "b"]) == 0.0
