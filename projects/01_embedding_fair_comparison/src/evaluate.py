"""Score every retriever on the same corpus, the same queries, the same metrics.

Writes results/scores.json. Each method is fitted on the corpus and then asked to rank
all documents for each query; nothing is tuned per method.
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
from retrievers import (
    BGE,
    BM25,
    LSA,
    FastText,
    NomicEmbed,
    TfIdf,
    Word2Vec,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
KS = (1, 5, 10, 20)


def metrics(order: np.ndarray, gold_idx: set[int]) -> dict:
    """recall@k and reciprocal rank, for one query.

    Every query has exactly two gold documents, so recall@k is (found in top k) / 2 and
    needs no normalisation caveat.
    """
    out = {}
    for k in KS:
        top = order[:k]
        out[f"recall@{k}"] = sum(1 for i in top if i in gold_idx) / len(gold_idx)
    rr = 0.0
    for rank, i in enumerate(order[:100], start=1):
        if i in gold_idx:
            rr = 1.0 / rank
            break
    out["mrr"] = rr
    return out


def run(method_cls, docs, queries, title_to_idx, nomic_cache=None) -> dict:
    name = method_cls.name
    print(f"  fitting {name} ...", end=" ", flush=True)
    started = time.time()
    method = method_cls()
    if isinstance(method, NomicEmbed):
        method.fit(docs, cache=nomic_cache)
    else:
        method.fit(docs)
    fit_seconds = time.time() - started
    print(f"{fit_seconds:.1f}s", end="  ", flush=True)

    started = time.time()
    totals = {f"recall@{k}": 0.0 for k in KS}
    totals["mrr"] = 0.0
    for q in queries:
        scores = method.score(q.question)
        order = np.argsort(-scores)
        gold_idx = {title_to_idx[t] for t in q.gold_titles if t in title_to_idx}
        for key, value in metrics(order, gold_idx).items():
            totals[key] += value
    query_seconds = time.time() - started

    result = {k: v / len(queries) for k, v in totals.items()}
    result.update(
        {
            "method": name,
            "trained_on": method_cls.trained_on,
            "fit_seconds": round(fit_seconds, 1),
            "query_seconds": round(query_seconds, 1),
        }
    )
    print(f"query {query_seconds:.1f}s   recall@10 {result['recall@10']:.3f}")
    return result


def main(n_queries: int = 1000, only: list[str] | None = None) -> None:
    doc_ids, docs, queries = build(n_queries)
    stats = coverage(doc_ids, queries)
    print(
        f"corpus: {stats['documents']} docs, {stats['queries']} queries, "
        f"{stats['missing_gold_titles']} missing gold"
    )

    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    RESULTS.mkdir(exist_ok=True)

    cache_path = RESULTS / "nomic_cache.json"
    nomic_cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    classes = [BM25, TfIdf, LSA, Word2Vec, FastText, BGE, NomicEmbed]
    if only:
        classes = [c for c in classes if c.__name__ in only]

    rows = []
    for cls in classes:
        try:
            rows.append(run(cls, docs, queries, title_to_idx, nomic_cache))
        except Exception as exc:  # noqa: BLE001 - one missing library must not lose the run
            print(f"  SKIPPED {cls.name}: {type(exc).__name__}: {str(exc)[:90]}")

    if nomic_cache:
        cache_path.write_text(json.dumps(nomic_cache))

    out = RESULTS / "scores.json"
    existing = json.loads(out.read_text()) if out.exists() else {"rows": []}
    by_name = {r["method"]: r for r in existing.get("rows", [])}
    for r in rows:
        by_name[r["method"]] = r
    payload = {"corpus": stats, "rows": list(by_name.values())}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n{'method':26} {'trained on':28} " + "  ".join(f"r@{k:<3}" for k in KS) + "   MRR")
    print("-" * 92)
    for r in sorted(payload["rows"], key=lambda r: -r["recall@10"]):
        cells = "  ".join(f"{r[f'recall@{k}']:.3f}" for k in KS)
        print(f"{r['method']:26} {r['trained_on']:28} {cells}   {r['mrr']:.3f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    main(n, only)
