"""Tests for the three tie-breaking rules.

No dataset and no network. The distance matrices are written out by hand so the k nearest
neighbours of each test point can be read off directly, which is the only way to tell a
tie-breaking bug from a plausible accuracy figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from knn import RULES, accuracy, majority_baseline, predict, tie_rate

# Four reference documents, labels alternating True/False.
LABELS = np.array([True, False, True, False])


def distances(*rows: list[float]) -> np.ndarray:
    return np.array(rows, dtype=np.float64)


# --- the rules agree when there is no tie -------------------------------------------------


def test_k_of_one_can_never_tie():
    """One neighbour is always its own majority. This is why the published rule is inert at
    k = 1 and only bites at k = 2."""
    d = distances([0.1, 0.2, 0.3, 0.4])
    assert tie_rate(d, LABELS, 1) == 0.0
    for rule in RULES:
        out = predict(d, LABELS, 1, rule, truth=np.array([False]))
        assert out.tolist() == [True]


def test_an_odd_k_cannot_tie_on_a_binary_task():
    """Three neighbours over two classes always have a 2-1 majority. The published rule
    therefore changes nothing at k = 3, 5 or 11 - and everything at k = 2."""
    d = distances([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1])
    for k in (1, 3):
        assert tie_rate(d, LABELS, k) == 0.0


def test_all_rules_agree_on_a_clear_majority():
    labels = np.array([True, True, False, False])
    d = distances([0.1, 0.2, 0.9, 0.9])  # two nearest are both True
    truth = np.array([False])
    outputs = {rule: predict(d, labels, 3, rule, truth=truth).tolist() for rule in RULES}
    assert outputs["oracle_tie"] == outputs["nearest_tie"] == outputs["random_tie"] == [True]


# --- the rules disagree exactly on ties ----------------------------------------------------


def test_the_oracle_rule_returns_the_truth_on_a_tie():
    """The reproduction of the published behaviour: two neighbours, one of each label, and
    the prediction comes out equal to the answer."""
    d = distances([0.1, 0.2, 0.9, 0.9])  # nearest two are True then False: a tie
    assert tie_rate(d, LABELS, 2) == 1.0
    assert predict(d, LABELS, 2, "oracle_tie", truth=np.array([False])).tolist() == [False]
    assert predict(d, LABELS, 2, "oracle_tie", truth=np.array([True])).tolist() == [True]


def test_the_oracle_rule_scores_a_perfect_one_on_an_all_tie_set():
    """Whatever the data, if every case ties then the published rule is right every time.
    That is the mechanism behind the inflated number, stated as a test."""
    d = distances([0.1, 0.2, 0.9, 0.9], [0.2, 0.1, 0.9, 0.9], [0.9, 0.9, 0.1, 0.2])
    truth = np.array([False, True, False])
    assert tie_rate(d, LABELS, 2) == 1.0
    assert accuracy(predict(d, LABELS, 2, "oracle_tie", truth=truth), truth) == 1.0


def test_the_nearest_rule_ignores_the_truth():
    """The same input, two different answers, one prediction. A rule whose output depends
    on the label is not producing predictions."""
    d = distances([0.1, 0.2, 0.9, 0.9])
    first = predict(d, LABELS, 2, "nearest_tie")
    second = predict(d, LABELS, 2, "nearest_tie")
    assert first.tolist() == second.tolist() == [True]  # the nearer of the tied pair


def test_the_random_rule_is_deterministic_given_a_seed():
    d = distances([0.1, 0.2, 0.9, 0.9])
    a = predict(d, LABELS, 2, "random_tie", seed=7)
    b = predict(d, LABELS, 2, "random_tie", seed=7)
    assert a.tolist() == b.tolist()


def test_the_oracle_rule_refuses_to_run_without_the_answer():
    """It cannot be deployed, and the error says so rather than quietly returning something."""
    d = distances([0.1, 0.2, 0.9, 0.9])
    with pytest.raises(ValueError, match="not a classifier"):
        predict(d, LABELS, 2, "oracle_tie")


def test_an_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown rule"):
        predict(distances([0.1, 0.2, 0.3, 0.4]), LABELS, 2, "whatever")


# --- the tie rate -------------------------------------------------------------------------


def test_tie_rate_counts_the_documents_the_rule_decides():
    d = distances(
        [0.1, 0.2, 0.9, 0.9],  # True then False: tie
        [0.1, 0.9, 0.2, 0.9],  # True then True: no tie
    )
    assert tie_rate(d, LABELS, 2) == 0.5


def test_tie_rate_is_zero_when_every_neighbourhood_agrees():
    labels = np.array([True, True, True, True])
    d = distances([0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1])
    assert tie_rate(d, labels, 2) == 0.0


# --- accuracy and the floor ---------------------------------------------------------------


def test_accuracy_is_the_share_correct():
    assert accuracy(np.array([True, False, True]), np.array([True, False, False])) == pytest.approx(
        2 / 3
    )


def test_the_majority_floor_uses_the_training_distribution():
    """Predicting the most common training label is what a classifier has to beat. On a
    54/46 task that floor is high enough to flatter a method that has learned nothing."""
    train = np.array([True, True, True, False])
    truth = np.array([True, True, False, False])
    assert majority_baseline(train, truth) == pytest.approx(0.5)


def test_ties_are_broken_by_distance_not_by_index_order():
    """`nearest_tie` must follow the distance ordering. Falling back to array order would
    make the result depend on how the reference set happened to be shuffled."""
    labels = np.array([False, True])
    d = distances([0.9, 0.1])  # index 1 is nearer, and is True
    assert predict(d, labels, 2, "nearest_tie").tolist() == [True]
