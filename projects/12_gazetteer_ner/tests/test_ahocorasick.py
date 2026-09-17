"""Tests for the Aho-Corasick automaton and overlap resolution.

No dataset and no network. Patterns are written as word lists and interned here, so every
expected match can be read off the input by eye - which is the only way to catch a failure
link that is subtly wrong, because a broken automaton returns fewer matches rather than an
error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ahocorasick import AhoCorasick, Match, longest_non_overlapping


class Interner:
    def __init__(self) -> None:
        self.table: dict[str, int] = {}

    def __call__(self, words: str) -> list[int]:
        return [self.table.setdefault(w, len(self.table)) for w in words.split()]


def build(patterns: list[str]) -> tuple[AhoCorasick, Interner]:
    intern = Interner()
    return AhoCorasick([intern(p) for p in patterns]), intern


def found(automaton: AhoCorasick, intern: Interner, text: str) -> set[tuple[int, int, int]]:
    return {(m.pattern, m.start, m.end) for m in automaton.find(intern(text))}


# --- basic matching -----------------------------------------------------------------------


def test_a_single_pattern_is_found():
    automaton, intern = build(["new york"])
    assert found(automaton, intern, "i visited new york once") == {(0, 2, 4)}


def test_a_pattern_absent_from_the_text_is_not_reported():
    automaton, intern = build(["new york"])
    assert found(automaton, intern, "i visited boston once") == set()


def test_every_occurrence_is_reported():
    automaton, intern = build(["paris"])
    assert found(automaton, intern, "paris and paris again") == {(0, 0, 1), (0, 2, 3)}


def test_several_patterns_are_found_in_one_pass():
    automaton, intern = build(["new york", "boston"])
    matches = found(automaton, intern, "new york then boston")
    assert matches == {(0, 0, 2), (1, 3, 4)}


def test_a_single_token_pattern_works():
    automaton, intern = build(["paris"])
    assert found(automaton, intern, "paris") == {(0, 0, 1)}


def test_an_empty_pattern_is_ignored_rather_than_matching_everywhere():
    automaton, intern = build(["", "paris"])
    assert found(automaton, intern, "paris") == {(1, 0, 1)}


def test_an_empty_text_matches_nothing():
    automaton, _ = build(["paris"])
    assert automaton.find([]) == []


# --- the failure and output links ---------------------------------------------------------


def test_nested_patterns_are_all_reported():
    """The output-link test, and the one that fails silently when the links are wrong.

    `york` is a suffix of `new york`, so a text containing `new york` contains both. An
    automaton that merges outputs incorrectly returns only one of them and looks like it is
    working.
    """
    automaton, intern = build(["new york", "york"])
    assert found(automaton, intern, "new york") == {(0, 0, 2), (1, 1, 2)}


def test_a_suffix_pattern_is_found_after_a_failed_longer_match():
    """The failure link doing its job: after `new` fails to continue into `new york`, the
    automaton must still be able to match `york` starting where it does."""
    automaton, intern = build(["new york city", "york"])
    assert found(automaton, intern, "new york harbour") == {(1, 1, 2)}


def test_overlapping_patterns_are_both_reported():
    automaton, intern = build(["a b", "b c"])
    assert found(automaton, intern, "a b c") == {(0, 0, 2), (1, 1, 3)}


def test_three_deep_nesting():
    automaton, intern = build(["university of paris", "of paris", "paris"])
    matches = found(automaton, intern, "university of paris")
    assert matches == {(0, 0, 3), (1, 1, 3), (2, 2, 3)}


def test_a_repeated_pattern_after_a_partial_match():
    automaton, intern = build(["a a b"])
    assert found(automaton, intern, "a a a b") == {(0, 1, 4)}


def test_matching_is_token_level_not_substring():
    """A character automaton matches `cat` inside `catalogue`. On a 66,000-name gazetteer
    that produces more spurious matches than real ones, so the alphabet is the vocabulary."""
    automaton, intern = build(["cat"])
    assert found(automaton, intern, "catalogue cat") == {(0, 1, 2)}


# --- structure ----------------------------------------------------------------------------


def test_shared_prefixes_share_trie_nodes():
    """The reason the automaton is affordable: 66,000 names beginning with `the` cost one
    node between them, not 66,000."""
    shared, _ = build(["new york", "new jersey"])
    separate, _ = build(["new york", "old jersey"])
    assert shared.size < separate.size


def test_the_root_has_no_failure_loop():
    automaton, _ = build(["a b", "b"])
    assert automaton.nodes[0].fail == 0


# --- overlap resolution -------------------------------------------------------------------


def test_the_longest_match_wins():
    matches = [Match(0, 0, 3), Match(1, 2, 3)]
    assert longest_non_overlapping(matches) == [Match(0, 0, 3)]


def test_non_overlapping_matches_are_all_kept():
    matches = [Match(0, 0, 2), Match(1, 3, 5)]
    assert longest_non_overlapping(matches) == matches


def test_results_come_back_in_text_order():
    matches = [Match(1, 5, 7), Match(0, 0, 2)]
    assert [m.start for m in longest_non_overlapping(matches)] == [0, 5]


def test_an_equal_length_tie_goes_to_the_earlier_match():
    matches = [Match(1, 2, 4), Match(0, 0, 2)]
    assert longest_non_overlapping(matches) == [Match(0, 0, 2), Match(1, 2, 4)]


def test_resolution_is_a_convention_not_a_result():
    """`paris` inside `university of paris` is a real match of a real entity. Keeping both
    is indefensible and so is keeping the short one; longest-match is a choice."""
    matches = [Match(0, 0, 3), Match(2, 2, 3)]
    resolved = longest_non_overlapping(matches)
    assert len(resolved) == 1
    assert resolved[0].pattern == 0


def test_nothing_in_gives_nothing_out():
    assert longest_non_overlapping([]) == []


@pytest.mark.parametrize("n", [1, 5, 50])
def test_many_patterns_do_not_break_the_automaton(n):
    automaton, intern = build([f"entity {i}" for i in range(n)])
    assert found(automaton, intern, "entity 0") >= {(0, 0, 2)}
