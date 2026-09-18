"""Tests for the depth sweep.

No model, no dataset, no network: the reranker is replaced by an array of scores, which is
all `rerank_prefix` ever sees. That is deliberate - the thing worth testing is the prefix
arithmetic, and the cross-encoder is not part of it.

The sharpest test here is `test_reranking_at_the_evaluation_cutoff_cannot_change_recall`.
It pins a property that makes one row of the results table provably zero, and a sweep that
did not show that row as zero would have a bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from depth import conversion, recall_at, rerank_prefix

CANDIDATES = np.array([10, 11, 12, 13, 14])
#: Deliberately the reverse of the first-stage order, so any reranking is visible.
SCORES = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


# --- the prefix arithmetic ------------------------------------------------------------------


def test_only_the_prefix_is_reordered():
    """Depth 3 reverses the first three and leaves the rest alone."""
    out = rerank_prefix(CANDIDATES, SCORES, 3)
    assert out.tolist() == [12, 11, 10, 13, 14]


def test_the_tail_is_kept_in_first_stage_order():
    """A reranker deployed at depth d does not delete what sits below d. Dropping the tail
    would make recall@k undefined for k > d and would flatter every shallow depth by
    removing exactly the documents it failed to reach."""
    out = rerank_prefix(CANDIDATES, SCORES, 2)
    assert out[2:].tolist() == [12, 13, 14]


def test_depth_zero_changes_nothing():
    assert rerank_prefix(CANDIDATES, SCORES, 0).tolist() == CANDIDATES.tolist()


def test_depth_beyond_the_candidate_list_is_clamped():
    """Asking for depth 500 when the first stage returned 5 must not raise or pad."""
    out = rerank_prefix(CANDIDATES, SCORES, 500)
    assert sorted(out.tolist()) == sorted(CANDIDATES.tolist())
    assert len(out) == len(CANDIDATES)


def test_every_candidate_survives_at_every_depth():
    """Reranking is a permutation. Losing or duplicating a document would corrupt recall
    in a way that looks like a result."""
    for d in range(len(CANDIDATES) + 2):
        out = rerank_prefix(CANDIDATES, SCORES, d)
        assert sorted(out.tolist()) == sorted(CANDIDATES.tolist())


def test_ties_keep_first_stage_order():
    """With equal scores the first stage's ordering must survive, or the measurement picks
    up an arbitrary tiebreak instead of the reranker's opinion."""
    out = rerank_prefix(CANDIDATES, np.ones(5), 5)
    assert out.tolist() == CANDIDATES.tolist()


def test_a_prefix_equals_a_fresh_pass_at_that_depth():
    """The assumption the whole sweep rests on: because a cross-encoder scores pairs
    independently, reranking the top 500 once and slicing gives the same answer as
    rescoring the top d. If this failed, every depth but the largest would be fiction."""
    for d in (1, 2, 3, 4, 5):
        from_prefix = rerank_prefix(CANDIDATES, SCORES, d)
        fresh = rerank_prefix(CANDIDATES[:d], SCORES[:d], d)
        assert from_prefix[:d].tolist() == fresh.tolist()


# --- the property that makes one row provably zero -------------------------------------------


def test_reranking_at_the_evaluation_cutoff_cannot_change_recall():
    """Reranking the top 10 and then measuring recall@10 reorders a set into itself.

    The set of documents in the top 10 is identical before and after, so recall@10 is
    unchanged no matter what the reranker thinks. **A reranking depth must exceed the
    evaluation cutoff to do anything at all for recall**, which is why the depth-10 row of
    the results table reads +0.000 for every first stage.
    """
    gold = {13, 14}
    before = recall_at(CANDIDATES, gold, 5)
    after = recall_at(rerank_prefix(CANDIDATES, SCORES, 5), gold, 5)
    assert before == after


def test_reranking_deeper_than_the_cutoff_can_change_recall():
    """The contrast that makes the test above meaningful rather than a tautology."""
    gold = {14}
    before = recall_at(CANDIDATES, gold, 2)
    after = recall_at(rerank_prefix(CANDIDATES, SCORES, 5), gold, 2)
    assert after > before


# --- metrics ---------------------------------------------------------------------------------


def test_recall_counts_the_share_of_gold_found():
    assert recall_at(np.array([1, 2, 3]), {1, 9}, 3) == pytest.approx(0.5)


def test_recall_respects_the_cutoff():
    assert recall_at(np.array([5, 1]), {1}, 1) == 0.0
    assert recall_at(np.array([5, 1]), {1}, 2) == 1.0


def test_recall_with_no_gold_is_zero_rather_than_a_division_error():
    assert recall_at(np.array([1, 2]), set(), 2) == 0.0


def test_conversion_is_the_share_of_the_ceiling_reached():
    assert conversion(0.9, 0.95) == pytest.approx(0.9 / 0.95)


def test_conversion_of_a_zero_ceiling_is_zero():
    """A first stage that fetched no gold has nothing to convert, and the reranker cannot
    be blamed for it."""
    assert conversion(0.0, 0.0) == 0.0


def test_conversion_is_one_when_everything_reachable_was_reached():
    assert conversion(0.8, 0.8) == pytest.approx(1.0)
