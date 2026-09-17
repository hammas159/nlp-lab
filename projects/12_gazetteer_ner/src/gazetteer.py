"""A gazetteer built from HotpotQA paragraph titles, and the ambiguity inside it.

Every HotpotQA paragraph is a Wikipedia article, so its title is an entity name and the
66,581 of them are a gazetteer nobody had to annotate. The paragraph is also *about* that
entity, which gives a recall signal without annotation: the title should appear in its own
text.

What the gazetteer cannot give is precision. A match of the name `Paris` is not a mention of
the entity `Paris`, and no amount of coverage fixes that. The number that bounds a
gazetteer tagger is therefore not how many names it holds but **how many of those names are
ambiguous**, and that is measurable from the gazetteer and corpus alone.

Two kinds of ambiguity are separated here, because they have different fixes:

* **surface collision** - two different entities share a name once the Wikipedia
  disambiguator is stripped (`Paris (film)` and `Paris (band)` are both `paris`). No
  context-free matcher can tell them apart; the entry is unresolvable by construction.
* **common-word entries** - a name that is also ordinary English (`It`, `Water`, `Here`).
  These match constantly and are almost never the entity.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import tokenize

#: Wikipedia's parenthetical disambiguator: "Mercury (element)", "Paris (film)".
DISAMBIGUATOR = re.compile(r"\s*\([^)]*\)\s*$")


@dataclass
class Gazetteer:
    #: Distinct surface forms, as token id lists.
    patterns: list[list[int]]
    #: The surface form of each pattern, for reporting.
    surfaces: list[str]
    #: Titles that produced each surface form. More than one means an unresolvable entry.
    titles_per_surface: list[list[str]]
    vocabulary: dict[str, int]
    stats: dict

    def __len__(self) -> int:
        return len(self.patterns)


def strip_disambiguator(title: str) -> str:
    """`Paris (film)` becomes `Paris`.

    This is what a matcher has to do - the parenthetical never appears in running text, so
    keeping it would make the entry unmatchable. It is also what *creates* the surface
    collisions measured below, and that trade is the point rather than a side effect.
    """
    return DISAMBIGUATOR.sub("", title).strip()


def build(titles: list[str], docs: list[str], min_tokens: int = 1) -> Gazetteer:
    """Intern the corpus vocabulary, then map each title to a token-id pattern.

    Titles whose tokens are entirely absent from the corpus vocabulary cannot match
    anything, so they are dropped and counted - keeping them would inflate the gazetteer
    size without changing a single match.
    """
    vocabulary: dict[str, int] = {}
    for doc in docs:
        for token in tokenize(doc):
            vocabulary.setdefault(token, len(vocabulary))

    by_surface: dict[tuple[int, ...], list[str]] = {}
    surface_text: dict[tuple[int, ...], str] = {}
    unmatchable = 0

    for title in titles:
        surface = strip_disambiguator(title)
        tokens = tokenize(surface)
        if len(tokens) < min_tokens or any(t not in vocabulary for t in tokens):
            unmatchable += 1
            continue
        key = tuple(vocabulary[t] for t in tokens)
        by_surface.setdefault(key, []).append(title)
        surface_text.setdefault(key, surface)

    keys = list(by_surface)
    collisions = sum(1 for k in keys if len(by_surface[k]) > 1)
    titles_collided = sum(len(by_surface[k]) for k in keys if len(by_surface[k]) > 1)

    return Gazetteer(
        patterns=[list(k) for k in keys],
        surfaces=[surface_text[k] for k in keys],
        titles_per_surface=[by_surface[k] for k in keys],
        vocabulary=vocabulary,
        stats={
            "titles": len(titles),
            "unmatchable_titles": unmatchable,
            "distinct_surfaces": len(keys),
            "colliding_surfaces": collisions,
            "titles_lost_to_collision": titles_collided,
            "corpus_vocabulary": len(vocabulary),
            "single_token_surfaces": sum(1 for k in keys if len(k) == 1),
        },
    )


def document_frequency(docs: list[str], vocabulary: dict[str, int]) -> Counter:
    """How many documents each vocabulary token appears in.

    Used to find the gazetteer entries that are also ordinary English: a single-token name
    whose token appears in a large share of the corpus is not going to be the entity when
    it matches.
    """
    counts: Counter = Counter()
    for doc in docs:
        counts.update({vocabulary[t] for t in tokenize(doc) if t in vocabulary})
    return counts
