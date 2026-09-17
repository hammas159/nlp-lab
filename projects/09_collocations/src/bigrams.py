"""Bigram contingency tables from a token stream.

Counting is the easy half. The half that decides the answer is the minimum-frequency cutoff
applied before scoring, which every toolkit has, every tutorial sets without comment, and
this project exists to measure.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import tokenize


@dataclass
class Bigrams:
    pairs: list[tuple[str, str]]
    #: a, b, c, d of the 2x2 contingency table, aligned with ``pairs``.
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    total: int
    stats: dict

    def __len__(self) -> int:
        return len(self.pairs)

    def filter_by_count(self, min_count: int) -> Bigrams:
        keep = self.a >= min_count
        return Bigrams(
            pairs=[p for p, k in zip(self.pairs, keep) if k],
            a=self.a[keep],
            b=self.b[keep],
            c=self.c[keep],
            d=self.d[keep],
            total=self.total,
            stats={**self.stats, "min_count": min_count, "bigram_types_kept": int(keep.sum())},
        )


def count(tokens: list[str]) -> Bigrams:
    """Adjacent-pair counts and the marginals each table needs.

    ``b`` is "w1 followed by something else", so the marginal is over *first* positions and
    ``c`` over *second* positions. Using the same unigram count for both would be wrong at
    the edges of the stream, and wrong in a way that is invisible - the tables would still
    sum correctly.
    """
    pair_counts = Counter(pairwise(tokens))
    first = Counter(tokens[:-1])
    second = Counter(tokens[1:])
    total = len(tokens) - 1

    pairs = list(pair_counts)
    a = np.array([pair_counts[p] for p in pairs], dtype=np.float64)
    left = np.array([first[w1] for w1, _ in pairs], dtype=np.float64)
    right = np.array([second[w2] for _, w2 in pairs], dtype=np.float64)

    b = left - a
    c = right - a
    d = total - a - b - c

    return Bigrams(
        pairs=pairs,
        a=a,
        b=b,
        c=c,
        d=d,
        total=total,
        stats={
            "tokens": len(tokens),
            "bigram_tokens": total,
            "bigram_types": len(pairs),
            "hapax_bigrams": int((a == 1).sum()),
            "hapax_share": float((a == 1).mean()) if len(a) else float("nan"),
        },
    )


def from_documents(docs: list[str]) -> Bigrams:
    """Count within documents, not across them.

    Concatenating the corpus into one stream would create a bigram at every document
    boundary out of two words that never appeared together. There are as many of those as
    there are documents, and they are all hapax pairs - exactly the population PMI ranks
    highest.
    """
    pair_counts: Counter = Counter()
    first: Counter = Counter()
    second: Counter = Counter()
    total = 0
    tokens_seen = 0

    for doc in docs:
        tokens = tokenize(doc)
        tokens_seen += len(tokens)
        if len(tokens) < 2:
            continue
        pair_counts.update(pairwise(tokens))
        first.update(tokens[:-1])
        second.update(tokens[1:])
        total += len(tokens) - 1

    pairs = list(pair_counts)
    a = np.array([pair_counts[p] for p in pairs], dtype=np.float64)
    b = np.array([first[w1] for w1, _ in pairs], dtype=np.float64) - a
    c = np.array([second[w2] for _, w2 in pairs], dtype=np.float64) - a
    d = total - a - b - c

    return Bigrams(
        pairs=pairs,
        a=a,
        b=b,
        c=c,
        d=d,
        total=total,
        stats={
            "documents": len(docs),
            "tokens": tokens_seen,
            "bigram_tokens": total,
            "bigram_types": len(pairs),
            "hapax_bigrams": int((a == 1).sum()),
            "hapax_share": float((a == 1).mean()) if len(a) else float("nan"),
        },
    )
