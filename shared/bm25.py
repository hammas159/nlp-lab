"""Okapi BM25, implemented rather than imported.

It is forty lines, it is the baseline almost no embedding comparison reports, and every
project in this lab needs it. Having it here means all three score against the identical
implementation rather than three copies that could drift.

The tokenizer is a parameter because project 02 exists to vary exactly that.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from shared.benchmark import tokenize as default_tokenize


class BM25:
    name = "BM25"
    trained_on = "this corpus"

    def __init__(self, tokenize=None, k1: float = 1.5, b: float = 0.75):
        self.tokenize = tokenize or default_tokenize
        self.k1, self.b = k1, b

    def fit(self, docs: list[str]) -> BM25:
        tokens = [self.tokenize(d) for d in docs]
        self.lengths = np.array([len(t) for t in tokens], dtype=np.float32)
        # Aggressive preprocessing can empty a document; guard the mean so a corpus of
        # empty documents does not divide by zero.
        self.avg_len = float(self.lengths.mean()) or 1.0
        self.n_docs = len(docs)
        self.empty_docs = int((self.lengths == 0).sum())

        self.postings: dict[str, dict[int, int]] = {}
        for i, toks in enumerate(tokens):
            for term, count in Counter(toks).items():
                self.postings.setdefault(term, {})[i] = count

        # Robertson/Sparck-Jones idf, floored at zero so a term appearing in almost every
        # document cannot contribute a negative score.
        self.idf = {
            term: max(0.0, math.log((self.n_docs - len(p) + 0.5) / (len(p) + 0.5) + 1.0))
            for term, p in self.postings.items()
        }
        self.vocabulary = len(self.postings)
        self.total_tokens = int(self.lengths.sum())
        return self

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in self.tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for i, freq in posting.items():
                norm = 1 - self.b + self.b * self.lengths[i] / self.avg_len
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        return scores
