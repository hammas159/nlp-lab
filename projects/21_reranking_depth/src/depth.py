"""Reranking at many depths from one pass of the cross-encoder.

Project 03 fixed the reranker's window at 50 and found that every first stage converts
about 96% of its recall@50 into recall@10. That finding is a function of the number 50,
which nothing in the project justified. This sweeps it.

**The whole sweep costs one pass, not one pass per depth.** A cross-encoder scores each
(query, document) pair independently - the score of a pair does not depend on which other
documents are in the candidate set. So scoring the top *N* once and then taking prefixes
gives exactly the same ranking that rescoring at each depth would, for a sixth of the work.

That property is worth stating because it is not true of every reranker. A listwise model,
or anything that normalises over the candidate set, would need a fresh pass per depth, and
reusing scores would silently produce numbers for an experiment that was never run.
"""

from __future__ import annotations

import numpy as np


def rerank_prefix(candidates: np.ndarray, scores: np.ndarray, depth: int) -> np.ndarray:
    """Rerank only the first `depth` candidates, leaving the rest in first-stage order.

    The tail matters. A reranker deployed at depth *d* does not delete documents ranked
    below *d*; they stay where the first stage put them. Dropping them would make recall@k
    undefined for k > d and would flatter every depth by removing the documents that a
    shallow window failed to reach.
    """
    depth = max(0, min(depth, len(candidates)))
    head = candidates[:depth]
    order = np.argsort(-np.asarray(scores[:depth], dtype=np.float64), kind="stable")
    return np.concatenate([head[order], candidates[depth:]])


def recall_at(order: np.ndarray, gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    return sum(1 for i in order[:k] if i in gold) / len(gold)


def conversion(after_recall: float, ceiling: float) -> float:
    """Share of the reachable gold the reranker actually pulled into the top 10.

    The ceiling is the first stage's recall at the reranking depth: no reranker can place a
    document in the top 10 if the first stage never fetched it.
    """
    return after_recall / ceiling if ceiling else 0.0
