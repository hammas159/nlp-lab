"""Count matrices for documents and queries over one shared vocabulary.

Raw counts only. Every weighting decision belongs to `smart.py`, so that the grid can vary
the weighting without rebuilding the index - which is the only reason evaluating hundreds
of schemes is affordable.

The vocabulary is built from the documents. A term that appears only in a query has no
document frequency and cannot be weighted; it is dropped, and the count of how often that
happens is reported rather than hidden, because a scheme cannot be blamed for a word the
index never contained.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import tokenize


@dataclass
class Index:
    documents: sparse.csr_matrix
    queries: sparse.csr_matrix
    vocabulary: list[str]
    doc_ids: list[str]
    gold: list[set[int]]
    stats: dict


def _counts(texts: list[str], index: dict[str, int]) -> sparse.csr_matrix:
    rows, cols, data = [], [], []
    for i, text in enumerate(texts):
        for word, n in Counter(t for t in tokenize(text) if t in index).items():
            rows.append(i)
            cols.append(index[word])
            data.append(float(n))
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), len(index)), dtype=np.float64)


def build_index(doc_ids: list[str], docs: list[str], queries) -> Index:
    vocabulary = sorted({t for d in docs for t in tokenize(d)})
    position = {w: i for i, w in enumerate(vocabulary)}

    document_counts = _counts(docs, position)
    query_texts = [q.question for q in queries]
    query_counts = _counts(query_texts, position)

    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    gold = [{title_to_idx[t] for t in q.gold_titles if t in title_to_idx} for q in queries]

    query_terms = sum(len(tokenize(t)) for t in query_texts)
    kept = int(query_counts.sum())
    empty = int((np.diff(query_counts.indptr) == 0).sum())

    return Index(
        documents=document_counts,
        queries=query_counts,
        vocabulary=vocabulary,
        doc_ids=doc_ids,
        gold=gold,
        stats={
            "documents": len(docs),
            "queries": len(queries),
            "vocabulary": len(vocabulary),
            "document_nonzeros": int(document_counts.nnz),
            "query_terms": query_terms,
            "query_terms_in_vocabulary": kept,
            "queries_with_no_indexable_term": empty,
        },
    )
