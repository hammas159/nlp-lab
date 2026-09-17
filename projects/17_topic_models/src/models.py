"""LSA, LDA and NMF behind one interface, so the only thing that varies is the model.

All three factor a document-term matrix into topics. They disagree about what a topic *is*:

``LSA``   truncated SVD. Topics are orthogonal directions of maximum variance. Entries may
          be negative, so a topic can mean "these words, and not those" - excellent for
          reconstructing the matrix, awkward to read as a list of words.
``NMF``   the same factorisation with non-negativity imposed (Lee & Seung, 1999). Losing
          negative entries costs reconstruction accuracy and buys parts-based topics that
          read as word lists.
``LDA``   a generative model (Blei, Ng & Jordan, 2003): documents are mixtures over topics,
          topics are distributions over words, both with Dirichlet priors. It is the only
          one of the three that is probabilistic, and the only one that is stochastic.

**The input representation is not free.** LDA is defined over counts - it models how many
times a word was drawn - while LSA and NMF are conventionally given TF-IDF. Feeding LDA
TF-IDF is a common tutorial shortcut and is not what the model says; `run.py` measures what
it costs rather than asserting it.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SEED = 0


class TopicModel:
    """Common interface: fit, then expose doc-topic, topic-word, and readable topics."""

    name = "topic model"
    needs = "tfidf"

    def __init__(self, n_topics: int, seed: int = DEFAULT_SEED) -> None:
        self.n_topics = n_topics
        self.seed = seed

    def fit(self, matrix) -> TopicModel:
        raise NotImplementedError

    def transform(self, matrix) -> np.ndarray:
        raise NotImplementedError

    def top_words(self, terms: list[str], n: int = 10) -> list[list[str]]:
        """The n highest-weighted words of each topic - what a paper prints as its topics."""
        out = []
        for row in self.components:
            out.append([terms[i] for i in np.argsort(-row)[:n]])
        return out


class LSA(TopicModel):
    name = "LSA (truncated SVD)"
    needs = "tfidf"

    def fit(self, matrix) -> LSA:
        from sklearn.decomposition import TruncatedSVD

        # TruncatedSVD raises rather than clamping when n_components exceeds the rank
        # available, so clamp explicitly - project 01 hit exactly this.
        n = min(self.n_topics, min(matrix.shape) - 1)
        self.model = TruncatedSVD(n_components=n, random_state=self.seed)
        self.model.fit(matrix)
        self.components = self.model.components_
        return self

    def transform(self, matrix) -> np.ndarray:
        return self.model.transform(matrix)


class NMF(TopicModel):
    name = "NMF"
    needs = "tfidf"

    def fit(self, matrix) -> NMF:
        from sklearn.decomposition import NMF as SkNMF

        self.model = SkNMF(
            n_components=min(self.n_topics, min(matrix.shape) - 1),
            random_state=self.seed,
            init="nndsvd",
            # 400 did not converge on this corpus and sklearn said so. A comparison where
            # one model is stopped early is measuring the iteration budget.
            max_iter=2_000,
        )
        self.model.fit(matrix)
        self.components = self.model.components_
        return self

    def transform(self, matrix) -> np.ndarray:
        return self.model.transform(matrix)


class LDA(TopicModel):
    name = "LDA"
    needs = "counts"

    def fit(self, matrix) -> LDA:
        from sklearn.decomposition import LatentDirichletAllocation

        self.model = LatentDirichletAllocation(
            n_components=min(self.n_topics, min(matrix.shape) - 1),
            random_state=self.seed,
            # Batch rather than online: online is for corpora too large to hold, and its
            # partial passes leave LDA under-fitted next to two models run to convergence.
            learning_method="batch",
            max_iter=100,
        )
        self.model.fit(matrix)
        self.components = self.model.components_
        return self

    def transform(self, matrix) -> np.ndarray:
        return self.model.transform(matrix)


MODELS = [LSA, NMF, LDA]


def unit(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot product is a cosine similarity.

    LSA's topic weights can be negative and LDA's are a probability simplex, so the vectors
    live on very different scales. Cosine is what makes them comparable at all, and is what
    every one of these methods is used with in practice.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)
