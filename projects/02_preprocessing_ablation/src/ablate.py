"""Run BM25 once per preprocessing variant, changing one step at a time.

The retriever, the corpus, the queries and the metric are identical across variants.
The only thing that moves is the tokenizer, so any difference in the table is caused by
preprocessing and nothing else.
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import build, coverage

from preprocess import VARIANTS, make_tokenizer

RESULTS = Path(__file__).resolve().parent.parent / "results"
KS = (1, 5, 10, 20)


class BM25:
    """Okapi BM25, parameterised by the tokenizer it is given."""

    def __init__(self, tokenize, k1: float = 1.5, b: float = 0.75):
        self.tokenize, self.k1, self.b = tokenize, k1, b

    def fit(self, docs: list[str]) -> BM25:
        tokens = [self.tokenize(d) for d in docs]
        self.lengths = np.array([len(t) for t in tokens], dtype=np.float32)
        # A document can be emptied by aggressive preprocessing; guard the mean.
        self.avg_len = float(self.lengths.mean()) or 1.0
        self.n_docs = len(docs)
        self.empty_docs = int((self.lengths == 0).sum())

        self.postings: dict[str, dict[int, int]] = {}
        for i, toks in enumerate(tokens):
            for term, count in Counter(toks).items():
                self.postings.setdefault(term, {})[i] = count

        self.idf = {}
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf[term] = max(0.0, math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0))

        self.vocabulary = len(self.postings)
        self.total_tokens = int(self.lengths.sum())
        return self

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        for term in self.tokenize(query):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf[term]
            for i, freq in posting.items():
                norm = 1 - self.b + self.b * self.lengths[i] / self.avg_len
                scores[i] += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
        return scores


def metrics(order: np.ndarray, gold: set[int]) -> dict:
    out = {f"recall@{k}": sum(1 for i in order[:k] if i in gold) / len(gold) for k in KS}
    out["mrr"] = next((1.0 / rank for rank, i in enumerate(order[:100], start=1) if i in gold), 0.0)
    return out


def main(n_queries: int = 300) -> None:
    doc_ids, docs, queries = build(n_queries)
    stats = coverage(doc_ids, queries)
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    print(
        f"corpus: {stats['documents']} docs, {stats['queries']} queries, "
        f"{stats['missing_gold_titles']} missing gold\n"
    )

    rows = []
    for label, kwargs in VARIANTS.items():
        tokenize = make_tokenizer(**kwargs)
        started = time.time()
        bm25 = BM25(tokenize).fit(docs)
        totals = dict.fromkeys([f"recall@{k}" for k in KS] + ["mrr"], 0.0)
        for q in queries:
            order = np.argsort(-bm25.score(q.question))
            gold = {title_to_idx[t] for t in q.gold_titles if t in title_to_idx}
            for key, value in metrics(order, gold).items():
                totals[key] += value
        row = {k: v / len(queries) for k, v in totals.items()}
        row.update(
            {
                "variant": label,
                "vocabulary": bm25.vocabulary,
                "total_tokens": bm25.total_tokens,
                "empty_docs": bm25.empty_docs,
                "seconds": round(time.time() - started, 1),
            }
        )
        rows.append(row)
        print(
            f"  {label:30} r@10 {row['recall@10']:.3f}   "
            f"vocab {row['vocabulary']:6,}   {row['seconds']:5.1f}s"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ablation.json").write_text(
        json.dumps({"corpus": stats, "rows": rows}, indent=2), encoding="utf-8"
    )

    base = rows[0]
    print(f"\n{'variant':32} " + "  ".join(f"r@{k:<3}" for k in KS) + "   MRR     vs base")
    print("-" * 88)
    for r in rows:
        cells = "  ".join(f"{r[f'recall@{k}']:.3f}" for k in KS)
        delta = r["recall@10"] - base["recall@10"]
        mark = "" if r is base else f"{delta:+.3f}"
        print(f"{r['variant']:32} {cells}   {r['mrr']:.3f}   {mark}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
