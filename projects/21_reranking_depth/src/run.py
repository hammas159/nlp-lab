"""Sweep the reranking depth that project 03 held fixed at 50.

python src/run.py            # depths 10..500, three first stages
python src/run.py --quick    # 60 queries, depths to 100
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

from shared.benchmark import build, tokenize
from shared.bm25 import BM25

import depth as D

RESULTS = Path(__file__).resolve().parent.parent / "results"
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEPTHS = (10, 20, 50, 100, 200, 500)
QUICK_DEPTHS = (10, 50, 100)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


class TfIdf:
    name = "TF-IDF"

    def fit(self, docs):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.matrix = self.vec.fit_transform(docs)
        return self

    def score(self, query):
        return np.asarray((self.matrix @ self.vec.transform([query]).T).todense()).ravel()


class BGE:
    name = "BGE-small"

    def fit(self, docs):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        m = np.asarray(self.model.encode(docs, batch_size=64, show_progress_bar=False))
        self.matrix = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-10)
        return self

    def score(self, query):
        prefix = "Represent this sentence for searching relevant passages: "
        q = np.asarray(self.model.encode([prefix + query], show_progress_bar=False))
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-10)
        return (self.matrix @ q.T).ravel()


FIRST_STAGES = [BM25, TfIdf, BGE]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    depths = QUICK_DEPTHS if args.quick else DEPTHS
    n_queries = 60 if args.quick else 300
    largest = max(depths)
    started = time.time()

    rule("1. The setup")
    doc_ids, docs, queries = build(300)
    queries = queries[:n_queries]
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    print(f"  {len(docs):,} documents, {len(queries)} queries, exactly 2 gold each")
    print(f"  reranker: {CROSS_ENCODER}")
    print(f"  depths swept: {', '.join(str(d) for d in depths)}")
    print(
        f"\n  Each query's top {largest} is scored once and every depth reads a prefix of it.\n"
        "  A cross-encoder scores pairs independently, so a prefix gives exactly the ranking\n"
        "  a fresh pass at that depth would - at a fraction of the cost."
    )

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(CROSS_ENCODER, max_length=512)

    rows = []
    for stage_cls in FIRST_STAGES:
        stage = stage_cls().fit(docs)
        began = time.time()

        totals = {d: {"after": 0.0, "ceiling": 0.0} for d in depths}
        before_recall10 = 0.0
        pairs = 0
        for q in queries:
            gold = {title_to_idx[t] for t in q.gold_titles if t in title_to_idx}
            order = np.argsort(-stage.score(q.question))
            before_recall10 += D.recall_at(order, gold, 10)

            candidates = order[:largest]
            scores = np.asarray(
                reranker.predict(
                    [(q.question, docs[i]) for i in candidates],
                    batch_size=128,
                    show_progress_bar=False,
                )
            )
            pairs += len(candidates)
            for d in depths:
                reranked = D.rerank_prefix(candidates, scores, d)
                totals[d]["after"] += D.recall_at(reranked, gold, 10)
                totals[d]["ceiling"] += D.recall_at(order, gold, d)
        seconds = time.time() - began
        before_recall10 /= len(queries)

        rule(f"2. {stage_cls.name} - r@10 before reranking: {before_recall10:.3f}")
        print(f"  {'depth':>6} {'ceiling':>9} {'r@10 after':>12} {'change':>9} {'converted':>11}")
        for d in depths:
            after = totals[d]["after"] / len(queries)
            ceiling = totals[d]["ceiling"] / len(queries)
            row = {
                "first_stage": stage_cls.name,
                "depth": d,
                "before_recall@10": before_recall10,
                "ceiling": ceiling,
                "after_recall@10": after,
                "change": after - before_recall10,
                "conversion": D.conversion(after, ceiling),
            }
            rows.append(row)
            print(
                f"  {d:>6} {ceiling:>9.3f} {after:>12.3f} {after - before_recall10:>+9.3f} "
                f"{row['conversion']:>10.1%}"
            )
        print(f"  ({pairs:,} pairs scored in {seconds:.1f}s, reused for every depth)")

    rule("3. Where does the gain stop?")
    for stage_cls in FIRST_STAGES:
        mine = [r for r in rows if r["first_stage"] == stage_cls.name]
        best = max(mine, key=lambda r: r["after_recall@10"])
        deepest = max(mine, key=lambda r: r["depth"])
        print(
            f"  {stage_cls.name:12} best depth {best['depth']:>4} "
            f"(r@10 {best['after_recall@10']:.3f});  at depth {deepest['depth']} it is "
            f"{deepest['after_recall@10']:.3f} ({deepest['after_recall@10'] - best['after_recall@10']:+.3f})"
        )

    rule("4. Does project 03's finding survive the sweep?")
    print("  Project 03 reported that every first stage converts about 96% of its ceiling,")
    print("  measured only at depth 50.\n")
    print(f"  {'depth':>6} " + "".join(f"{s.name:>12}" for s in FIRST_STAGES) + f"{'spread':>9}")
    for d in depths:
        at_depth = [
            next(r for r in rows if r["first_stage"] == s.name and r["depth"] == d)
            for s in FIRST_STAGES
        ]
        conversions = [r["conversion"] for r in at_depth]
        print(
            f"  {d:>6} "
            + "".join(f"{c:>11.1%} " for c in conversions)
            + f"{max(conversions) - min(conversions):>8.1%}"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "depth.json").write_text(
        json.dumps(
            {
                "documents": len(docs),
                "queries": len(queries),
                "model": CROSS_ENCODER,
                "depths": list(depths),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'depth.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
