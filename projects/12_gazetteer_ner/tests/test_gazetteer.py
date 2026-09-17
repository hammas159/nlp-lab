"""Tests for gazetteer construction, disambiguator stripping and collision counting.

No dataset and no network: the titles and paragraphs are written out here so every count
can be checked by hand.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gazetteer import build, document_frequency, strip_disambiguator

DOCS = [
    "Paris is the capital of France",
    "Paris is a 1929 film directed by Edmund Goulding",
    "The Seine flows through Paris and Rouen",
    "Water is a chemical compound",
]


# --- disambiguator ------------------------------------------------------------------------


def test_a_parenthetical_is_stripped():
    assert strip_disambiguator("Paris (film)") == "Paris"


def test_a_title_without_one_is_unchanged():
    assert strip_disambiguator("Paris") == "Paris"


def test_only_a_trailing_parenthetical_is_stripped():
    """`Sgt. Pepper's (album)` loses the tag; `M*A*S*H (TV series)` too. But a parenthetical
    in the middle of a name is part of the name."""
    assert strip_disambiguator("Rondo (a) piece") == "Rondo (a) piece"


def test_nested_text_inside_the_parenthetical():
    assert strip_disambiguator("Mercury (element)") == "Mercury"


def test_whitespace_is_trimmed():
    assert strip_disambiguator("Paris  (film) ") == "Paris"


# --- construction -------------------------------------------------------------------------


def test_titles_become_patterns():
    gazetteer = build(["Paris", "Water"], DOCS)
    assert "Paris" in gazetteer.surfaces
    assert "Water" in gazetteer.surfaces


def test_a_title_with_tokens_absent_from_the_corpus_is_dropped():
    """It cannot match anything, so keeping it would inflate the gazetteer size without
    changing a single match."""
    gazetteer = build(["Paris", "Ouagadougou"], DOCS)
    assert gazetteer.stats["unmatchable_titles"] == 1
    assert "Ouagadougou" not in gazetteer.surfaces


def test_disambiguated_titles_collapse_to_one_surface():
    """The trade this project is about: stripping the parenthetical is what makes an entry
    matchable, and it is also what makes two entities share one entry."""
    gazetteer = build(["Paris (city)", "Paris (film)"], DOCS)
    assert gazetteer.stats["distinct_surfaces"] == 1
    assert gazetteer.stats["colliding_surfaces"] == 1
    assert gazetteer.stats["titles_lost_to_collision"] == 2


def test_a_collided_surface_remembers_every_title():
    gazetteer = build(["Paris (city)", "Paris (film)"], DOCS)
    assert sorted(gazetteer.titles_per_surface[0]) == ["Paris (city)", "Paris (film)"]


def test_distinct_titles_do_not_collide():
    gazetteer = build(["Paris", "Water"], DOCS)
    assert gazetteer.stats["colliding_surfaces"] == 0
    assert gazetteer.stats["distinct_surfaces"] == 2


def test_single_token_surfaces_are_counted():
    gazetteer = build(["Paris", "The Seine"], DOCS)
    assert gazetteer.stats["single_token_surfaces"] == 1


def test_multi_token_titles_become_multi_token_patterns():
    gazetteer = build(["The Seine"], DOCS)
    assert len(gazetteer.patterns[0]) == 2


def test_the_vocabulary_covers_the_corpus():
    gazetteer = build(["Paris"], DOCS)
    assert gazetteer.stats["corpus_vocabulary"] == len({t for d in DOCS for t in d.lower().split()})


def test_an_empty_title_is_unmatchable():
    gazetteer = build(["", "Paris"], DOCS)
    assert gazetteer.stats["unmatchable_titles"] == 1


def test_a_title_that_is_only_a_disambiguator_is_unmatchable():
    gazetteer = build(["(film)"], DOCS)
    assert gazetteer.stats["unmatchable_titles"] == 1


# --- document frequency -------------------------------------------------------------------


def test_document_frequency_counts_documents_not_occurrences():
    """`Paris` appears twice in one of these sentences in other corpora; here the point is
    that a token repeated within a document still counts once."""
    gazetteer = build(["Paris"], DOCS)
    frequency = document_frequency(DOCS, gazetteer.vocabulary)
    assert frequency[gazetteer.vocabulary["paris"]] == 3


def test_a_token_in_one_document_has_frequency_one():
    gazetteer = build(["Water"], DOCS)
    frequency = document_frequency(DOCS, gazetteer.vocabulary)
    assert frequency[gazetteer.vocabulary["water"]] == 1


def test_frequency_finds_the_common_words():
    gazetteer = build(["Paris"], DOCS)
    frequency = document_frequency(DOCS, gazetteer.vocabulary)
    assert frequency[gazetteer.vocabulary["is"]] >= 3
