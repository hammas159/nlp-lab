"""Are the differences between preprocessing variants distinguishable from noise?

A table of six numbers two points apart invites the reader to rank them. Whether that
ranking means anything depends on the sample size, and reporting the table without the
interval is how a 2-point difference becomes a recommendation.

Every variant is scored on the *same* queries, so the comparison is paired: the right
test is on the per-query difference, not on the two means independently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import build, coverage

from ablate import BM25, metrics
from preprocess import VARIANTS, make_tokenizer

RESULTS = Path(__file__).resolve().parent.parent / "results"
BOOTSTRAP = 2000
RNG = np.random.default_rng(0)


def per_query_recall(docs, queries, title_to_idx, tokenize, k: int = 10) -> np.ndarray:
    bm25 = BM25(tokenize).fit(docs)
    out = np.zeros(len(queries), dtype=np.float32)
    for j, q in enumerate(queries):
        order = np.argsort(-bm25.score(q.question))
        gold = {title_to_idx[t] for t in q.gold_titles if t in title_to_idx}
        out[j] = metrics(order, gold)[f"recall@{k}"]
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray) -> dict:
    """Resample queries, not scores. The unit of variation is the query."""
    diff = a - b
    n = len(diff)
    idx = RNG.integers(0, n, size=(BOOTSTRAP, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Two-sided p: how often does a resample land on the other side of zero?
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "mean_difference": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "significant": bool(lo > 0 or hi < 0),
    }


def main(n_queries: int = 300) -> None:
    doc_ids, docs, queries = build(n_queries)
    stats = coverage(doc_ids, queries)
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    print(f"corpus: {stats['documents']} docs, {stats['queries']} queries")
    print(f"paired bootstrap, {BOOTSTRAP} resamples over queries\n")

    scores = {}
    for label, kwargs in VARIANTS.items():
        scores[label] = per_query_recall(docs, queries, title_to_idx, make_tokenizer(**kwargs))
        print(f"  scored {label}")

    base_label = next(iter(VARIANTS))
    base = scores[base_label]

    rows = []
    print(f"\n{'variant':32} {'r@10':>7} {'delta':>8} {'95% CI':>18}  {'p':>7}  verdict")
    print("-" * 92)
    print(f"{base_label:32} {base.mean():7.3f} {'--':>8} {'--':>18}  {'--':>7}  baseline")
    for label, arr in scores.items():
        if label == base_label:
            continue
        res = paired_bootstrap(arr, base)
        rows.append({"variant": label, "recall@10": float(arr.mean()), **res})
        ci = f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}]"
        verdict = "REAL" if res["significant"] else "within noise"
        print(
            f"{label:32} {arr.mean():7.3f} {res['mean_difference']:+8.3f} {ci:>18}  "
            f"{res['p_value']:7.3f}  {verdict}"
        )

    real = [r for r in rows if r["significant"]]
    print(
        f"\n  {len(real)} of {len(rows)} differences are distinguishable from noise "
        f"at n={len(queries)}."
    )
    if not real:
        print("  Every preprocessing choice measured here is within sampling error of")
        print("  doing nothing but lowercasing.")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "significance.json").write_text(
        json.dumps(
            {"corpus": stats, "baseline": base_label, "bootstrap": BOOTSTRAP, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
