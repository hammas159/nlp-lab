"""Tests for the Split container and its subsampling.

No dataset and no network: the loaders are not exercised here, only the sampling that every
result depends on.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import Split


def split_of(n: int) -> Split:
    return Split([f"doc {i}" for i in range(n)], np.array([i % 2 == 0 for i in range(n)]))


def test_length_is_the_document_count():
    assert len(split_of(7)) == 7


def test_subsample_returns_the_requested_size():
    assert len(split_of(100).subsample(10)) == 10


def test_subsample_keeps_texts_and_labels_aligned():
    """The single most damaging thing that can go wrong here: shuffling one and not the
    other produces a classifier that is measurably terrible for no visible reason."""
    original = split_of(50)
    sampled = original.subsample(10, seed=3)
    lookup = dict(zip(original.texts, original.labels.tolist()))
    for text, label in zip(sampled.texts, sampled.labels.tolist()):
        assert lookup[text] == label


def test_subsample_is_deterministic_given_a_seed():
    a = split_of(100).subsample(20, seed=5)
    b = split_of(100).subsample(20, seed=5)
    assert a.texts == b.texts
    assert a.labels.tolist() == b.labels.tolist()


def test_different_seeds_give_different_samples():
    a = split_of(200).subsample(20, seed=1)
    b = split_of(200).subsample(20, seed=2)
    assert a.texts != b.texts


def test_subsample_draws_randomly_rather_than_taking_a_prefix():
    """Devign is ordered by project, so a prefix comes from a handful of codebases and
    shares far more vocabulary than a random draw. A nearest-neighbour method would look
    much better on that, for a reason unrelated to compression."""
    sampled = split_of(500).subsample(20, seed=0)
    assert sampled.texts != split_of(500).texts[:20]


def test_asking_for_more_than_exists_returns_everything():
    original = split_of(5)
    assert len(original.subsample(50)) == 5


def test_subsample_does_not_repeat_a_document():
    sampled = split_of(100).subsample(40, seed=7)
    assert len(set(sampled.texts)) == 40
