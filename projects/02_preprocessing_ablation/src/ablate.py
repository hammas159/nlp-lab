"""Run BM25 once per preprocessing variant, changing one step at a time.

The retriever, the corpus, the queries and the metric are identical across variants.
The only thing that moves is the tokenizer, so any difference in the table is caused by
preprocessing and nothing else.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import build, coverage
from shared.bm25 import BM25

from preprocess import VARIANTS, make_tokenizer

RESULTS = Path(__file__).resolve().parent.parent / "results"
KS = (1, 5, 10, 20)


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
