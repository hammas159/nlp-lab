"""Five ways to pick a sense, and the two baselines that decide whether any of them is good.

The claim this project tests is a claim about *reporting*: word sense disambiguation results
are often shown against a **random** baseline, when the baseline that matters is the
**most frequent sense**. Random is easy to beat. MFS is not.

``random``            uniform over the senses the lemma is ever tagged with. The flattering
                      comparison.
``first sense``       always answer sense 1. WordNet orders senses by frequency, so this is
                      the most-frequent-sense heuristic with no training at all.
``trained MFS``       the sense most often seen for this lemma in the *training* split. The
                      same idea, estimated rather than inherited - which also measures how
                      much of `first sense` is circular, since WordNet's ordering came from
                      a corpus like this one.
``context overlap``   a simple supervised model: represent each sense by the bag of words it
                      appeared with in training, and pick the sense whose bag best matches
                      the test context. The cheapest thing that could be called learning.
``context overlap
  + discourse``       the above, then force every occurrence of a lemma in a document to its
                      majority *predicted* sense. Gale, Church & Yarowsky (1992) observed
                      that a word almost always keeps one sense within a document.
``ORACLE``            the same discourse heuristic, but reading the *gold* senses of the
                      other occurrences. Not a method - a ceiling, and named one.

Every method returns a sense number, never abstains, and sees exactly the same tokens.

**On the oracle.** Propagating senses within a document is a real and well-supported
heuristic, but implementing it by reading neighbouring gold tags means reading the answer
key. The first version of this file did exactly that and scored 0.675, which looked like the
only method to beat the most frequent sense. Splitting it into an honest method and an
explicitly labelled oracle moved it to 0.563, below the baseline. The gap between the two is
what the heuristic is worth if the senses it spreads are already correct.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

CONTEXT_WINDOW = 5


class Disambiguator:
    name = "disambiguator"

    def fit(self, sentences, inventory) -> Disambiguator:
        return self

    def predict(self, token, sentence) -> int:
        raise NotImplementedError


class Random(Disambiguator):
    """Uniform over the lemma's observed senses - the baseline that flatters."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng([seed, 0x5E45])

    def fit(self, sentences, inventory) -> Random:
        self.inventory = inventory
        return self

    def predict(self, token, sentence) -> int:
        senses = sorted(self.inventory.get(token.key, {1}))
        return int(self.rng.choice(senses))


class FirstSense(Disambiguator):
    """Always sense 1. No training, no inventory, no glosses."""

    name = "first sense (WordNet order)"

    def predict(self, token, sentence) -> int:
        return 1


class TrainedMFS(Disambiguator):
    """The most frequent sense for this lemma in the training split."""

    name = "trained MFS"

    def fit(self, sentences, inventory) -> TrainedMFS:
        counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for sentence in sentences:
            for token in sentence:
                counts[token.key][token.sense] += 1
        self.best = {k: c.most_common(1)[0][0] for k, c in counts.items()}
        return self

    def predict(self, token, sentence) -> int:
        # An unseen lemma falls back to sense 1, which is the only answer available without
        # evidence - and is exactly what FirstSense would say.
        return self.best.get(token.key, 1)


class OneSensePerDiscourseOracle(Disambiguator):
    """**An oracle, not a method.** Gale, Church & Yarowsky (1992) observed that a word
    almost always keeps one sense within a document. This measures how true that is here,
    by reading the *gold* senses of the lemma's other occurrences in the same document.

    Excluding the token's own vote is not enough to make this a usable system: those other
    senses are human annotations from the evaluation set. It is reported because the size of
    the gap between it and the real method below is the value the heuristic *could* have if
    the senses it propagates were correct - and that gap turns out to be most of its score.
    """

    name = "one sense per discourse (ORACLE)"

    def fit(self, sentences, inventory) -> OneSensePerDiscourseOracle:
        self.fallback = TrainedMFS().fit(sentences, inventory)
        return self

    def set_documents(self, sentences) -> None:
        self.by_document: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
        for sentence in sentences:
            for token in sentence:
                self.by_document[(token.document, *token.key)][token.sense] += 1

    def predict(self, token, sentence) -> int:
        counts = self.by_document.get((token.document, *token.key))
        if counts:
            others = Counter(counts)
            others[token.sense] -= 1
            others = +others
            if others:
                return others.most_common(1)[0][0]
        return self.fallback.predict(token, sentence)


class ContextOverlap(Disambiguator):
    """Each sense is the bag of words it occurred with; pick the best-matching bag.

    Weighted by inverse document frequency over senses, so a word that appears near every
    sense contributes nothing. This is the cheapest supervised model that is not a constant,
    and it is here to show what beating MFS actually takes.
    """

    name = "context overlap (supervised)"

    def fit(self, sentences, inventory) -> ContextOverlap:
        self.bags: dict[tuple[str, str], dict[int, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        document_count: Counter = Counter()
        total = 0
        for sentence in sentences:
            words = [t.surface.lower() for t in sentence]
            for i, token in enumerate(sentence):
                context = self._context(words, i)
                self.bags[token.key][token.sense].update(context)
                document_count.update(set(context))
                total += 1
        self.idf = {w: math.log(total / (1 + c)) for w, c in document_count.items()}
        self.fallback = TrainedMFS().fit(sentences, inventory)
        return self

    @staticmethod
    def _context(words: list[str], i: int) -> list[str]:
        start = max(0, i - CONTEXT_WINDOW)
        end = min(len(words), i + CONTEXT_WINDOW + 1)
        return [w for j, w in enumerate(words[start:end], start) if j != i]

    def predict(self, token, sentence) -> int:
        senses = self.bags.get(token.key)
        if not senses:
            return self.fallback.predict(token, sentence)
        words = [t.surface.lower() for t in sentence]
        index = next((i for i, t in enumerate(sentence) if t is token), 0)
        context = self._context(words, index)
        best, best_score = None, -1.0
        for sense, bag in sorted(senses.items()):
            score = sum(bag.get(w, 0) * self.idf.get(w, 0.0) for w in context)
            if score > best_score:
                best, best_score = sense, score
        if best_score <= 0:
            return self.fallback.predict(token, sentence)
        return best


class ContextOverlapWithDiscourse(ContextOverlap):
    """The honest one-sense-per-discourse: context overlap, then force every occurrence of a
    lemma inside a document to that lemma's majority *predicted* sense.

    No gold label is ever consulted. This is what the Gale, Church & Yarowsky heuristic is
    worth when the senses it propagates have to be guessed rather than looked up.
    """

    name = "context overlap + discourse"

    def predict_document(self, tokens, sentence_of) -> dict[int, int]:
        """Predict a whole document at once, then vote within each lemma."""
        first = {
            id(t): super(ContextOverlapWithDiscourse, self).predict(t, sentence_of[id(t)])
            for t in tokens
        }
        votes: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
        for token in tokens:
            votes[(token.document, *token.key)][first[id(token)]] += 1
        return {id(t): votes[(t.document, *t.key)].most_common(1)[0][0] for t in tokens}


METHODS = [
    Random,
    FirstSense,
    TrainedMFS,
    ContextOverlap,
    ContextOverlapWithDiscourse,
    OneSensePerDiscourseOracle,
]


def accuracy(predictions: list[int], truth: list[int]) -> float:
    return sum(p == t for p, t in zip(predictions, truth)) / max(1, len(truth))
