"""Aho-Corasick multi-pattern matching, implemented rather than imported.

Matching a dictionary of 66,000 entity names against a corpus one name at a time is
66,000 passes over the text. Aho-Corasick does it in one, by building a trie of the
patterns and adding **failure links**: when a character does not extend the current match,
the automaton jumps to the longest proper suffix of what it has matched that is still a
prefix of some pattern, instead of restarting.

The output links are the part that is easy to get wrong and impossible to notice. A node
matching `york` must also report `new york` ending there if `york` is a suffix of it -
otherwise the automaton silently returns only the shortest of any nested pair, and a
gazetteer full of nested names (`Paris`, `Paris, Texas`, `University of Paris`) reports the
wrong one every time. `test_nested_patterns_are_all_reported` asserts that directly.

Matching is over **tokens, not characters**. A character automaton on raw text matches
`Cat` inside `Catalogue` and `Al` inside `Algorithm`, which on a 66,000-name gazetteer
produces far more spurious matches than real ones. Tokens are interned to integers first,
so the automaton's alphabet is the vocabulary rather than Unicode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Node:
    children: dict[int, int] = field(default_factory=dict)
    fail: int = 0
    #: Indices into the pattern list for patterns ending exactly at this node.
    outputs: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class Match:
    pattern: int
    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


class AhoCorasick:
    """A token-level multi-pattern automaton over a fixed pattern set."""

    def __init__(self, patterns: list[list[int]]) -> None:
        self.patterns = patterns
        self.nodes: list[_Node] = [_Node()]
        for index, pattern in enumerate(patterns):
            if pattern:
                self._insert(pattern, index)
        self._build_failure_links()

    def _insert(self, pattern: list[int], index: int) -> None:
        node = 0
        for symbol in pattern:
            child = self.nodes[node].children.get(symbol)
            if child is None:
                self.nodes.append(_Node())
                child = len(self.nodes) - 1
                self.nodes[node].children[symbol] = child
            node = child
        self.nodes[node].outputs.append(index)

    def _build_failure_links(self) -> None:
        """Breadth-first, because a node's failure link is defined in terms of its parent's.

        Outputs are merged along the failure chain at build time rather than walked at match
        time. Walking the chain per position is the textbook formulation and turns a linear
        scan into a quadratic one on a gazetteer with deep nesting.
        """
        queue: deque[int] = deque()
        for child in self.nodes[0].children.values():
            self.nodes[child].fail = 0
            queue.append(child)

        while queue:
            current = queue.popleft()
            for symbol, child in self.nodes[current].children.items():
                fallback = self.nodes[current].fail
                while fallback and symbol not in self.nodes[fallback].children:
                    fallback = self.nodes[fallback].fail
                self.nodes[child].fail = self.nodes[fallback].children.get(symbol, 0)
                if self.nodes[child].fail == child:
                    self.nodes[child].fail = 0
                self.nodes[child].outputs.extend(self.nodes[self.nodes[child].fail].outputs)
                queue.append(child)

    def find(self, tokens: list[int]) -> list[Match]:
        """Every occurrence of every pattern, in one pass."""
        out: list[Match] = []
        node = 0
        for position, symbol in enumerate(tokens):
            while node and symbol not in self.nodes[node].children:
                node = self.nodes[node].fail
            node = self.nodes[node].children.get(symbol, 0)
            for index in self.nodes[node].outputs:
                length = len(self.patterns[index])
                out.append(Match(index, position - length + 1, position + 1))
        return out

    @property
    def size(self) -> int:
        return len(self.nodes)


def longest_non_overlapping(matches: list[Match]) -> list[Match]:
    """Resolve overlaps by preferring the longest match, then the earliest.

    This is the choice every gazetteer tagger has to make and almost none reports. `Paris`
    inside `University of Paris` is a real match of a real entity, and keeping both is
    indefensible - but so is keeping the short one. Longest-match is the usual convention
    and it is a convention, not a result.
    """
    ordered = sorted(matches, key=lambda m: (-len(m), m.start))
    taken: list[Match] = []
    occupied: set[int] = set()
    for match in ordered:
        span = range(match.start, match.end)
        if any(position in occupied for position in span):
            continue
        taken.append(match)
        occupied.update(span)
    return sorted(taken, key=lambda m: m.start)
