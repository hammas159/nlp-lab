"""Pseudo-relevance feedback, reported per query rather than as a mean.

python src/run.py            # the full sweep
python src/run.py --quick    # fewer settings
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import build
from shared.bm25 import BM25

import prf

RESULTS = Path(__file__).resolve().parent.parent / "results"
K = 10


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, resamples: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng([seed, 0xFEED])
    diff = a - b
    idx = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {
        "mean_difference": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(min(p, 1.0)),
        "significant": bool(lo > 0 or hi < 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--queries", type=int, default=300)
    args = parser.parse_args()

    started = time.time()
    rule("1. Corpus and first stage")
    doc_ids, docs, queries = build(args.queries)
    queries = queries[: args.queries]
    bm25 = BM25().fit(docs)
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    gold = [{title_to_idx[t] for t in q.gold_titles if t in title_to_idx} for q in queries]

    print(f"  {len(docs):,} paragraphs, {len(queries):,} queries, {bm25.vocabulary:,} terms")

    baseline_scores = [bm25.score(q.question) for q in queries]
    baseline = np.array([prf.recall_at_k(s, g, K) for s, g in zip(baseline_scores, gold)])
    print(f"  BM25 recall@{K}: {baseline.mean():.3f}")

    settings = (
        [(10, 20, 0.5)]
        if args.quick
        else [
            (5, 10, 0.3),
            (5, 20, 0.5),
            (10, 10, 0.3),
            (10, 20, 0.5),
            (10, 50, 0.5),
            (20, 20, 0.5),
            (10, 20, 0.8),
        ]
    )

    rule("2. What the mean says")
    print(
        f"  {'docs':>5} {'terms':>6} {'alpha':>6} {'recall@10':>10} {'delta':>8} {'p':>7}  verdict"
    )

    rows = []
    for n_feedback, n_terms, alpha in settings:
        expanded = np.zeros(len(queries))
        for j, query in enumerate(queries):
            model = prf.relevance_model(
                bm25, docs, baseline_scores[j], n_feedback=n_feedback, n_terms=n_terms
            )
            weights = prf.expand(query.question, model, alpha=alpha)
            expanded[j] = prf.recall_at_k(prf.score_weighted(bm25, weights), gold[j], K)

        test = paired_bootstrap(expanded, baseline)
        improved = int((expanded > baseline).sum())
        hurt = int((expanded < baseline).sum())
        unchanged = len(queries) - improved - hurt
        worst = float((expanded - baseline).min())

        rows.append(
            {
                "n_feedback": n_feedback,
                "n_terms": n_terms,
                "alpha": alpha,
                "recall_at_10": float(expanded.mean()),
                "improved": improved,
                "hurt": hurt,
                "unchanged": unchanged,
                "worst_drop": worst,
                "significance": test,
                "_per_query": expanded,
            }
        )
        verdict = "REAL" if test["significant"] else "within noise"
        print(
            f"  {n_feedback:>5} {n_terms:>6} {alpha:>6.1f} {expanded.mean():>10.3f} "
            f"{test['mean_difference']:>+8.3f} {test['p_value']:>7.3f}  {verdict}"
        )

    rule("3. What the mean hides")
    print(
        f"  {'docs':>5} {'terms':>6} {'alpha':>6} {'improved':>9} {'hurt':>7} {'unchanged':>10} {'worst drop':>11}"
    )
    for row in rows:
        print(
            f"  {row['n_feedback']:>5} {row['n_terms']:>6} {row['alpha']:>6.1f} "
            f"{row['improved']:>9} {row['hurt']:>7} {row['unchanged']:>10} {row['worst_drop']:>11.3f}"
        )

    best = max(rows, key=lambda r: r["recall_at_10"])
    share_hurt = best["hurt"] / len(queries)
    print(
        f"\n  The best setting ({best['n_feedback']} docs, {best['n_terms']} terms, "
        f"alpha {best['alpha']}) gains {best['significance']['mean_difference']:+.3f} on "
        f"average\n  and makes {best['hurt']} of {len(queries):,} queries "
        f"({share_hurt * 100:.1f}%) worse, the worst by {abs(best['worst_drop']):.3f}."
    )

    rule("4. Which queries does it hurt?")
    per_query = best["_per_query"]
    change = per_query - baseline
    for label, mask in (
        ("first stage already perfect (recall 1.0)", baseline == 1.0),
        ("first stage partly right (0 < r < 1)", (baseline > 0) & (baseline < 1)),
        ("first stage found nothing (recall 0.0)", baseline == 0.0),
    ):
        n = int(mask.sum())
        if not n:
            continue
        print(
            f"  {label:42} n={n:>4}  mean change {change[mask].mean():>+7.3f}  "
            f"hurt {int((change[mask] < 0).sum()):>4}"
        )
    print(
        "\n  Feedback is harvested from documents nothing has checked. When the first stage\n"
        "  was already right there is nothing to gain and a query to drift away from."
    )

    for row in rows:
        row.pop("_per_query", None)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "feedback.json").write_text(
        json.dumps(
            {
                "corpus": {
                    "documents": len(docs),
                    "queries": len(queries),
                    "vocabulary": bm25.vocabulary,
                },
                "baseline_recall": float(baseline.mean()),
                "settings": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'feedback.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
