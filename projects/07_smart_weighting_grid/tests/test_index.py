"""Tests for the count index.

No dataset and no network: the benchmark's query objects are stood in for by a dataclass
with the same two fields the index reads.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from index import build_index


@dataclass(frozen=True)
class FakeQuery:
    question: str
    gold_titles: tuple[str, ...]


DOC_IDS = ["alpha", "beta", "gamma"]
DOCS = ["the cat sat", "the dog ran ran", "birds fly"]


def test_vocabulary_comes_from_the_documents():
    idx = build_index(DOC_IDS, DOCS, [FakeQuery("cat", ("alpha",))])
    assert idx.vocabulary == sorted({"the", "cat", "sat", "dog", "ran", "birds", "fly"})


def test_document_counts_are_raw_counts():
    idx = build_index(DOC_IDS, DOCS, [FakeQuery("cat", ("alpha",))])
    position = {w: i for i, w in enumerate(idx.vocabulary)}
    dense = np.asarray(idx.documents.todense())
    assert dense[1, position["ran"]] == 2.0
    assert dense[0, position["the"]] == 1.0
    assert dense[2, position["cat"]] == 0.0


def test_query_terms_absent_from_the_index_are_dropped():
    """A term with no document frequency cannot be weighted. Counting it would give it an
    undefined idf; keeping it silently would make one scheme look worse for a word no
    scheme could have matched."""
    idx = build_index(DOC_IDS, DOCS, [FakeQuery("cat zebra", ("alpha",))])
    assert idx.queries.sum() == 1.0
    assert idx.stats["query_terms"] == 2
    assert idx.stats["query_terms_in_vocabulary"] == 1


def test_a_query_with_nothing_indexable_is_counted():
    idx = build_index(DOC_IDS, DOCS, [FakeQuery("zebra", ("alpha",))])
    assert idx.stats["queries_with_no_indexable_term"] == 1


def test_gold_titles_become_document_indices():
    idx = build_index(DOC_IDS, DOCS, [FakeQuery("cat", ("alpha", "gamma"))])
    assert idx.gold == [{0, 2}]


def test_unknown_gold_titles_are_dropped_rather_than_crashing():
    idx = build_index(DOC_IDS, DOCS, [FakeQuery("cat", ("alpha", "missing"))])
    assert idx.gold == [{0}]


def test_shapes_line_up():
    queries = [FakeQuery("cat", ("alpha",)), FakeQuery("dog ran", ("beta",))]
    idx = build_index(DOC_IDS, DOCS, queries)
    assert idx.documents.shape == (3, len(idx.vocabulary))
    assert idx.queries.shape == (2, len(idx.vocabulary))


def test_documents_and_queries_share_one_vocabulary():
    """They must index the same columns, or the inner product in `search` multiplies
    unrelated terms together."""
    idx = build_index(DOC_IDS, DOCS, [FakeQuery("the cat", ("alpha",))])
    assert idx.documents.shape[1] == idx.queries.shape[1]


def test_stats_report_what_was_indexed():
    idx = build_index(DOC_IDS, DOCS, [FakeQuery("cat", ("alpha",))])
    assert idx.stats["documents"] == 3
    assert idx.stats["queries"] == 1
    assert idx.stats["vocabulary"] == len(idx.vocabulary)
    assert idx.stats["document_nonzeros"] == int(idx.documents.nnz)
