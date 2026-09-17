"""Evaluate the SMART weighting grid on HotpotQA.

    python src/run.py            # the full grid
    python src/run.py --quick    # a tenth of the corpus

Two axes, varied one at a time. Forty-five document schemes against a fixed query scheme
says how much the document-side decision is worth; forty-five query schemes against a fixed
document scheme says the same for the query side. Varying both at once would give 2,025
numbers and no attribution.
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

import index as index_module
import search
import smart

RESULTS = Path(__file__).resolve().parent.parent / "results"

#: The scheme every textbook writes down, in SMART's own notation. Nothing in the notation
#: says it is the best of the forty-five, which is the question here.
TEXTBOOK_DOCUMENT = "lnc"
TEXTBOOK_QUERY = "ltc"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 76)}", flush=True)


def evaluate_axis(
    idx: index_module.Index,
    idfs: dict[str, np.ndarray],
    document_schemes: list[str],
    query_schemes: list[str],
    k: int = 10,
) -> dict[tuple[str, str], np.ndarray]:
    """Score every (document scheme, query scheme) pair given, reusing weighted matrices."""
    weighted_queries = {
        scheme: smart.weight(idx.queries, scheme, idf=idfs) for scheme in set(query_schemes)
    }
    out = {}
    for doc_scheme in document_schemes:
        weighted_docs = smart.weight(idx.documents, doc_scheme)
        for query_scheme in query_schemes:
            out[(doc_scheme, query_scheme)] = search.recall_at_k(
                weighted_docs, weighted_queries[query_scheme], idx.gold, k=k
            )
    return out


def report(scores: dict[str, np.ndarray], title: str, highlight: str) -> dict:
    means = {scheme: float(v.mean()) for scheme, v in scores.items()}
    ordered = sorted(means.items(), key=lambda kv: -kv[1])
    best, worst = ordered[0], ordered[-1]
    rank = [s for s, _ in ordered].index(highlight) + 1

    print(f"\n  {title}")
    print(f"  {'rank':>4}  {'scheme':8} {'recall@10':>10}")
    for position, (scheme, value) in enumerate(ordered[:5], start=1):
        mark = "  <- textbook" if scheme == highlight else ""
        print(f"  {position:>4}  {scheme:8} {value:>10.3f}{mark}")
    print(f"  {'...':>4}")
    for position, (scheme, value) in enumerate(ordered[-3:], start=len(ordered) - 2):
        mark = "  <- textbook" if scheme == highlight else ""
        print(f"  {position:>4}  {scheme:8} {value:>10.3f}{mark}")

    print(
        f"\n  best {best[0]} {best[1]:.3f}, worst {worst[0]} {worst[1]:.3f}, "
        f"spread {best[1] - worst[1]:.3f}"
    )
    print(f"  textbook {highlight} ranks {rank} of {len(ordered)} at {means[highlight]:.3f}")

    components = {
        name: search.component_spread(means, position)
        for position, name in enumerate(("term frequency", "document frequency", "normalisation"))
    }
    print("\n  mean recall by component, which says which decision carries the weight:")
    for name, group in components.items():
        rendered = "  ".join(f"{letter}={value:.3f}" for letter, value in group.items())
        spread = max(group.values()) - min(group.values())
        print(f"    {name:20} {rendered}   spread {spread:.3f}")

    return {
        "means": means,
        "best": {"scheme": best[0], "recall": best[1]},
        "worst": {"scheme": worst[0], "recall": worst[1]},
        "spread": best[1] - worst[1],
        "textbook": {"scheme": highlight, "recall": means[highlight], "rank": rank},
        "components": components,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    # 300 is not a round number picked for speed. Projects 01, 02 and 03 all score on
    # `build(300)` - 2,964 paragraphs, 300 questions - and the headline of this project is a
    # comparison against project 01's BM25-versus-neural gap. Measured on a different
    # corpus size that comparison would be meaningless, because recall@10 depends on how
    # many documents the gold two are hiding among.
    parser.add_argument("--queries", type=int, default=300)
    args = parser.parse_args()

    started = time.time()
    rule("1. Corpus and index")
    doc_ids, docs, queries = build(100 if args.quick else args.queries)
    queries = queries[: args.queries]
    idx = index_module.build_index(doc_ids, docs, queries)
    for key, value in idx.stats.items():
        print(f"  {key:34} {value:,}")

    idfs = {code: smart.document_frequency(idx.documents, code) for code in smart.DF_CODES}
    schemes = smart.all_schemes()

    # ------------------------------------------------------ the document axis
    rule("2. Forty-five document weightings, one query weighting")
    step = time.time()
    doc_axis = evaluate_axis(idx, idfs, schemes, [TEXTBOOK_QUERY])
    doc_scores = {scheme: doc_axis[(scheme, TEXTBOOK_QUERY)] for scheme in schemes}
    doc_report = report(doc_scores, f"query weighting fixed at {TEXTBOOK_QUERY}", TEXTBOOK_DOCUMENT)
    print(f"  [{time.time() - step:.1f}s]")

    # --------------------------------------------------------- the query axis
    rule("3. Forty-five query weightings, one document weighting")
    step = time.time()
    query_axis = evaluate_axis(idx, idfs, [TEXTBOOK_DOCUMENT], schemes)
    query_scores = {scheme: query_axis[(TEXTBOOK_DOCUMENT, scheme)] for scheme in schemes}
    query_report = report(
        query_scores, f"document weighting fixed at {TEXTBOOK_DOCUMENT}", TEXTBOOK_QUERY
    )

    # Query-side normalisation scales every document's score for that query by one
    # constant, so it cannot reorder anything. A third of the query code is inert for
    # ranking, and the notation gives no hint of it. Measured rather than asserted.
    families: dict[str, set[float]] = {}
    for scheme, value in query_report["means"].items():
        families.setdefault(scheme[:2], set()).add(round(value, 12))
    inert = all(len(values) == 1 for values in families.values())
    query_report["normalisation_is_inert"] = inert
    query_report["distinct_query_schemes"] = len(families)
    print(
        f"\n  Query normalisation changed the ranking for "
        f"{sum(1 for v in families.values() if len(v) > 1)} of {len(families)} "
        f"term-weighting families.\n"
        f"  It scales every score for a query by one constant, so it cannot reorder "
        f"anything:\n  the 45 query schemes are {len(families)} distinct rankings wearing "
        f"45 names."
    )
    print(f"  [{time.time() - step:.1f}s]")

    # ------------------------------------------------------------- BM25 and significance
    rule("4. Against BM25, and is the spread larger than the thing people compare?")
    bm25 = BM25().fit(docs)
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    bm25_per_query = np.zeros(len(queries))
    for j, query in enumerate(queries):
        order = np.argsort(-bm25.score(query.question))[:10]
        gold = {title_to_idx[t] for t in query.gold_titles if t in title_to_idx}
        if gold:
            bm25_per_query[j] = len(gold & set(order.tolist())) / len(gold)

    best_scheme = doc_report["best"]["scheme"]
    print(f"  {'BM25 (shared/bm25.py)':34} {bm25_per_query.mean():.3f}")
    print(f"  {'best document scheme ' + best_scheme:34} {doc_report['best']['recall']:.3f}")
    print(f"  {'textbook ' + TEXTBOOK_DOCUMENT:34} {doc_report['textbook']['recall']:.3f}")

    against_bm25 = search.paired_bootstrap(doc_scores[best_scheme], bm25_per_query)
    textbook_vs_best = search.paired_bootstrap(
        doc_scores[best_scheme], doc_scores[TEXTBOOK_DOCUMENT]
    )
    for label, test in (
        (f"{best_scheme} vs BM25", against_bm25),
        (f"{best_scheme} vs textbook {TEXTBOOK_DOCUMENT}", textbook_vs_best),
    ):
        verdict = "REAL" if test["significant"] else "within noise"
        print(
            f"  {label:34} {test['mean_difference']:+.3f} "
            f"[{test['ci_low']:+.3f}, {test['ci_high']:+.3f}] p={test['p_value']:.3f} {verdict}"
        )

    print(
        f"\n  Spread across the document grid is {doc_report['spread']:.3f}. "
        f"Project 01 reports BM25\n  trailing a pretrained neural embedding by 0.078 on this "
        f"same benchmark."
    )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "grid.json").write_text(
        json.dumps(
            {
                "index": idx.stats,
                "document_axis": doc_report,
                "query_axis": query_report,
                "bm25": float(bm25_per_query.mean()),
                "significance": {
                    "best_vs_bm25": against_bm25,
                    "best_vs_textbook": textbook_vs_best,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'grid.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
