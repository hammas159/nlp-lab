"""NPMI topic coherence, computed on the corpus the topics were fitted on.

Coherence is the number topic-model papers report, and NPMI is the variant that Lau,
Newman and Baldwin (EACL 2014) found tracks human judgement most closely. It scores a topic
by how often its top words co-occur, relative to how often they would co-occur by chance:

    NPMI(a, b) = log( P(a,b) / P(a)P(b) )  /  -log P(a,b)

bounded to [-1, 1], where 1 means the two words always appear together and -1 never. A
topic's coherence is the mean over all pairs of its top words; a model's coherence is the
mean over its topics.

Two decisions that change the number and are usually left unstated:

- **Co-occurrence is per document, boolean.** A word appearing nine times in a paragraph
  counts once. Otherwise coherence rewards repetition rather than association.
- **The reference corpus is the fitted corpus.** Papers often score against Wikipedia
  instead, which measures whether the topics match general English rather than whether they
  describe this collection. Scoring in-corpus is the harder, more honest reading, and it is
  stated here rather than buried.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-12


def document_frequencies(
    documents: list[list[str]], vocabulary: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (per-word document counts, boolean document-term matrix).

    Boolean, not counts: coherence asks whether two words co-occur, not how often.
    """
    presence = np.zeros((len(documents), len(vocabulary)), dtype=bool)
    for i, tokens in enumerate(documents):
        for token in tokens:
            j = vocabulary.get(token)
            if j is not None:
                presence[i, j] = True
    return presence.sum(axis=0).astype(np.float64), presence


def npmi_pair(count_a: float, count_b: float, count_ab: float, n_documents: int) -> float:
    """NPMI for one pair of words.

    A pair that never co-occurs has an undefined logarithm. The convention is to return the
    floor, -1, rather than to drop the pair - dropping it would quietly reward a topic whose
    words never appear together by removing its worst evidence from the average.
    """
    if count_ab == 0:
        return -1.0
    p_a = count_a / n_documents
    p_b = count_b / n_documents
    p_ab = count_ab / n_documents
    return float(np.log(p_ab / (p_a * p_b + EPSILON) + EPSILON) / -np.log(p_ab + EPSILON))


def topic_coherence(
    words: list[str],
    vocabulary: dict[str, int],
    counts: np.ndarray,
    presence: np.ndarray,
) -> float:
    """Mean NPMI over every pair of a topic's top words."""
    indices = [vocabulary[w] for w in words if w in vocabulary]
    if len(indices) < 2:
        return 0.0
    n_documents = presence.shape[0]
    scores = []
    for a in range(len(indices)):
        for b in range(a + 1, len(indices)):
            i, j = indices[a], indices[b]
            together = float(np.count_nonzero(presence[:, i] & presence[:, j]))
            scores.append(npmi_pair(counts[i], counts[j], together, n_documents))
    return float(np.mean(scores))


def model_coherence(
    topics: list[list[str]],
    vocabulary: dict[str, int],
    counts: np.ndarray,
    presence: np.ndarray,
) -> float:
    """Mean coherence over a model's topics - the number that gets reported."""
    if not topics:
        return 0.0
    return float(np.mean([topic_coherence(t, vocabulary, counts, presence) for t in topics]))
