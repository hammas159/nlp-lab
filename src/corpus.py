"""Build a retrieval benchmark from the cached HotpotQA distractor split.

HotpotQA's `distractor` config gives each question ten paragraphs, exactly two of which
are the gold supporting documents. That makes the ground truth exact and the corpus
self-contained - no relevance judgements to guess at, and no download beyond what is
already in the Hugging Face cache.

Pooling the paragraphs across many questions turns "pick 2 from 10" into "pick 2 from
several thousand", which is a retrieval problem rather than a multiple-choice one.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = "datasets--hotpotqa--hotpot_qa"

TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Query:
    qid: str
    question: str
    gold_titles: tuple[str, ...]
    level: str
    qtype: str


def _cache_roots() -> list[Path]:
    roots = []
    if env := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(env))
    if env := os.environ.get("HF_HOME"):
        roots.append(Path(env) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def find_parquet(split: str = "validation") -> Path | None:
    for root in _cache_roots():
        hits = sorted(
            glob.glob(
                str(root / CACHE_DIR / "snapshots" / "*" / "**" / f"{split}-*.parquet"),
                recursive=True,
            )
        )
        if hits:
            return Path(hits[0])
    return None


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    Deliberately the same tokenizer for every method. A comparison where one model gets
    better preprocessing than another is measuring the preprocessing.
    """
    return TOKEN.findall(text.lower())


def build(limit: int = 1000, split: str = "validation") -> tuple[list[str], list[str], list[Query]]:
    """Return (doc_ids, doc_texts, queries).

    A document is one HotpotQA paragraph, identified by its title. Titles are unique
    within the dataset and are exactly what `supporting_facts` refers to, so they serve
    as document ids without inventing a mapping.
    """
    import pandas as pd

    path = find_parquet(split)
    if path is None:
        raise FileNotFoundError(
            "HotpotQA (distractor) not found in the Hugging Face cache. Fetch it with:\n"
            '  python -c "from datasets import load_dataset; '
            "load_dataset('hotpotqa/hot_pot_qa','distractor')\"\n"
            f"Looked in: {', '.join(str(r) for r in _cache_roots())}"
        )

    frame = pd.read_parquet(path)
    if limit:
        frame = frame.head(limit)

    docs: dict[str, str] = {}
    queries: list[Query] = []

    for _, row in frame.iterrows():
        context = row["context"]
        titles = list(context["title"])
        sentence_lists = list(context["sentences"])
        for title, sentences in zip(titles, sentence_lists):
            # Same title can appear as a distractor for several questions; keep one copy.
            docs.setdefault(str(title), " ".join(str(s) for s in sentences).strip())

        gold = tuple(dict.fromkeys(str(t) for t in row["supporting_facts"]["title"]))
        queries.append(
            Query(
                qid=str(row["id"]),
                question=str(row["question"]),
                gold_titles=gold,
                level=str(row.get("level", "")),
                qtype=str(row.get("type", "")),
            )
        )

    doc_ids = sorted(docs)
    return doc_ids, [docs[d] for d in doc_ids], queries


def coverage(doc_ids: list[str], queries: list[Query]) -> dict:
    """Sanity check: every gold title must exist in the corpus, or recall is capped."""
    known = set(doc_ids)
    missing = [t for q in queries for t in q.gold_titles if t not in known]
    golds = [len(q.gold_titles) for q in queries]
    return {
        "documents": len(doc_ids),
        "queries": len(queries),
        "gold_per_query_min": min(golds),
        "gold_per_query_max": max(golds),
        "missing_gold_titles": len(missing),
    }


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    doc_ids, doc_texts, queries = build(n)
    stats = coverage(doc_ids, queries)
    for key, value in stats.items():
        print(f"  {key:24} {value}")

    lengths = sorted(len(tokenize(t)) for t in doc_texts)
    print(
        f"  doc length (tokens)      min {lengths[0]}, "
        f"median {lengths[len(lengths) // 2]}, max {lengths[-1]}"
    )
    vocab = {w for t in doc_texts for w in tokenize(t)}
    print(f"  corpus vocabulary        {len(vocab):,}")
    print(f"  corpus size (tokens)     {sum(lengths):,}")
    print(f"\n  sample query  : {queries[0].question}")
    print(f"  gold titles   : {queries[0].gold_titles}")
