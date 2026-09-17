"""Three character n-gram identifiers over the same profiles.

Language identification is reported as a solved problem, and on a paragraph it is. The
number quoted is almost always measured on documents, while the thing it is used for is a
search query, a tweet, a log line or a code snippet - inputs an order of magnitude shorter.
This module implements three standard approaches so the accuracy-against-length curve can
be measured rather than assumed.

``cavnar_trenkle``  Cavnar & Trenkle (1994). Rank the n-grams of a profile by frequency,
                    rank the n-grams of the input, and score by how far apart the two
                    rankings put each n-gram. The original method, and it uses only order -
                    no probabilities at all.
``naive_bayes``     Multinomial Naive Bayes over character n-gram counts, in log space.
``compression``     Which class model compresses the input best, via the same normalised
                    compression distance project 08 uses. Needs no feature extraction.

All three read from one `Profile` per class, so a difference between them is a difference
in the decision rule and not in what they were shown.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

#: Cavnar & Trenkle use n = 1..5. The penalty for an n-gram absent from a profile is the
#: profile length, which is the largest rank distance possible.
NGRAM_RANGE = (1, 2, 3, 4, 5)
PROFILE_SIZE = 400


def ngrams(text: str, sizes: tuple[int, ...] = NGRAM_RANGE) -> list[str]:
    """Character n-grams, with the string padded so word edges are visible.

    The padding is not cosmetic. ``_the_`` and ``the`` are different features, and the
    difference between English and source code is largely at the edges - identifiers run
    together, prose does not.
    """
    padded = f"_{text}_"
    out: list[str] = []
    for n in sizes:
        out.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return out


@dataclass
class Profile:
    label: str
    counts: Counter
    ranks: dict[str, int]
    total: int
    text: str

    @classmethod
    def build(cls, label: str, text: str, size: int = PROFILE_SIZE) -> Profile:
        counts = Counter(ngrams(text))
        ranked = [gram for gram, _ in counts.most_common(size)]
        return cls(
            label=label,
            counts=counts,
            ranks={gram: rank for rank, gram in enumerate(ranked)},
            total=sum(counts.values()),
            text=text,
        )


def cavnar_trenkle(text: str, profiles: list[Profile], size: int = PROFILE_SIZE) -> str:
    """Rank-order distance: sum over the input's n-grams of how far the profile ranks them.

    An n-gram the profile has never seen costs ``size`` - the maximum possible displacement.
    That single constant is what makes the method robust on long input and brittle on short
    input, because on a ten-character string almost every n-gram is unseen and the score is
    dominated by the penalty rather than by the evidence.
    """
    counts = Counter(ngrams(text))
    ranked = [gram for gram, _ in counts.most_common(size)]
    best, best_score = profiles[0].label, math.inf
    for profile in profiles:
        score = sum(
            abs(rank - profile.ranks[gram]) if gram in profile.ranks else size
            for rank, gram in enumerate(ranked)
        )
        if score < best_score:
            best, best_score = profile.label, score
    return best


#: Smoothing needs the size of the union of every profile's n-gram vocabulary. On a
#: full-corpus profile that union has millions of entries, and rebuilding it per call made
#: a run that should take a minute take over an hour. It depends only on the profiles, so
#: it is computed once and memoised against their identities.
_VOCABULARY_SIZES: dict[tuple[int, ...], int] = {}


def _vocabulary_size(profiles: list[Profile]) -> int:
    key = tuple(id(p) for p in profiles)
    cached = _VOCABULARY_SIZES.get(key)
    if cached is None:
        vocabulary: set[str] = set()
        for profile in profiles:
            vocabulary.update(profile.counts)
        cached = len(vocabulary) or 1
        _VOCABULARY_SIZES[key] = cached
    return cached


def naive_bayes(text: str, profiles: list[Profile], alpha: float = 0.1) -> str:
    """Multinomial Naive Bayes over n-gram counts, in log space.

    Log space is not an optimisation: a product of several hundred n-gram probabilities
    underflows to exactly zero in double precision, and every input would then be assigned
    to whichever class was checked first.
    """
    observed = Counter(ngrams(text))
    size = _vocabulary_size(profiles)

    best, best_score = profiles[0].label, -math.inf
    for profile in profiles:
        denominator = profile.total + alpha * size
        score = sum(
            count * math.log((profile.counts.get(gram, 0) + alpha) / denominator)
            for gram, count in observed.items()
        )
        if score > best_score:
            best, best_score = profile.label, score
    return best


def compression(text: str, profiles: list[Profile], sample: int = 8_000) -> str:
    """Assign to whichever class profile compresses the input best.

    ``C(profile + text) - C(profile)`` is how many extra bytes the text costs once the
    compressor already knows the class - a direct estimate of how surprising it is. The
    profile is truncated to ``sample`` bytes so every class gets the same amount of context,
    otherwise the largest corpus wins by having more to reference.
    """
    import zlib

    best, best_cost = profiles[0].label, math.inf
    encoded = text.encode("utf-8")
    for profile in profiles:
        context = profile.text[:sample].encode("utf-8")
        alone = len(zlib.compress(context, 9))
        together = len(zlib.compress(context + b" " + encoded, 9))
        cost = together - alone
        if cost < best_cost:
            best, best_cost = profile.label, cost
    return best


METHODS = {
    "cavnar_trenkle": cavnar_trenkle,
    "naive_bayes": naive_bayes,
    "compression": compression,
}


def classify(method: str, text: str, profiles: list[Profile]) -> str:
    try:
        return METHODS[method](text, profiles)
    except KeyError:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(METHODS)}") from None
