"""Turning k nearest neighbours into a prediction - three ways, only two of which are a
classifier.

The gzip-kNN result that circulated widely was produced with k = 2 and a tie-breaking rule
that, when the two neighbours carried different labels, counted the prediction correct if
*either* of them matched the truth. With a binary task and k = 2, disagreement is the
common case, so that rule decides a large share of the test set - and it decides them by
looking at the answer.

That is not a classifier. It is **top-k accuracy**: the metric that asks whether the right
label is anywhere in the shortlist, which is a reasonable thing to report and an entirely
different thing from the number it was compared against.

Three rules are implemented so the difference is a measurement rather than an argument:

``oracle_tie``     plurality vote; **ties resolved by picking the true label** when it is
                   among the tied ones. This reproduces the published behaviour. It needs
                   the answer in order to predict, so it cannot be deployed - there is
                   nothing to return.
``nearest_tie``    plurality vote; ties broken by the closest neighbour among the tied
                   labels. Deterministic, and the natural reading of "k nearest".
``random_tie``     plurality vote; ties broken by a seeded coin. The honest floor, because
                   it is what an implementation does when it has no rule.

Only the tie cases differ. Where the k neighbours have a clear majority all three rules
agree, which is why the size of the gap is governed by the tie rate - reported alongside.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

RULES = ("oracle_tie", "nearest_tie", "random_tie")


def predict(
    distances: np.ndarray,
    labels: np.ndarray,
    k: int,
    rule: str,
    truth: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Predictions for each row of ``distances`` (one row per test document).

    ``truth`` is required only by ``top_k``, and requiring it is the point: a rule that
    cannot run without the answer is not producing predictions.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
    if rule == "oracle_tie" and truth is None:
        raise ValueError(
            "the oracle_tie rule needs the true labels to resolve a tie, which is exactly "
            "why it is not a classifier"
        )

    rng = np.random.default_rng([seed, 0x7E5])
    n_test = distances.shape[0]
    out = np.empty(n_test, dtype=labels.dtype)

    for i in range(n_test):
        order = np.argsort(distances[i], kind="stable")[:k]
        neighbour_labels = labels[order]

        counts = Counter(neighbour_labels.tolist())
        best = max(counts.values())
        tied = [label for label, n in counts.items() if n == best]

        if len(tied) == 1:
            out[i] = tied[0]
        elif rule == "oracle_tie":
            out[i] = truth[i] if truth[i] in tied else tied[0]
        elif rule == "nearest_tie":
            # The first tied label encountered walking outwards from the test point.
            out[i] = next(lbl for lbl in neighbour_labels.tolist() if lbl in tied)
        else:
            out[i] = tied[rng.integers(len(tied))]

    return out


def tie_rate(distances: np.ndarray, labels: np.ndarray, k: int) -> float:
    """Share of test documents whose k neighbours have no single most common label.

    This is the fraction of the test set the tie-breaking rule decides, and therefore the
    size of the lever the published rule was pulling.
    """
    ties = 0
    for i in range(distances.shape[0]):
        order = np.argsort(distances[i], kind="stable")[:k]
        counts = Counter(labels[order].tolist())
        best = max(counts.values())
        if sum(1 for n in counts.values() if n == best) > 1:
            ties += 1
    return ties / distances.shape[0]


def accuracy(predictions: np.ndarray, truth: np.ndarray) -> float:
    return float((predictions == truth).mean())


def majority_baseline(train_labels: np.ndarray, truth: np.ndarray) -> float:
    """Always predict the most common training label.

    Any classifier that does not beat this has learned nothing, and on a task with a 54/46
    split that floor is high enough to flatter a weak method.
    """
    most_common = Counter(train_labels.tolist()).most_common(1)[0][0]
    return float((truth == most_common).mean())
