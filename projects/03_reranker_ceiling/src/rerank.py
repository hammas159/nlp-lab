"""Does a cross-encoder reranker rescue a weak first stage - or only reorder it?

A reranker can only reorder what the first stage handed it. So the first stage's
recall@K is a hard **ceiling** on anything the reranker can achieve at k <= K, no matter
how good the cross-encoder is.

That makes two questions measurable rather than arguable:

1. How much of each first stage's recall@50 does the reranker convert into recall@10?
2. Does the *ordering* of first-stage retrievers survive reranking - or does a cheap
   retriever plus a reranker match an expensive one plus the same reranker?

If the answer to (2) is that the ordering collapses, then the retriever comparison in
project 01 was measuring something that a downstream reranker makes irrelevant.
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

RESULTS = Path(__file__).resolve().parent.parent / "results"
KS = (1, 5, 10, 20)
DEPTH = 50  # how many first-stage results the reranker is allowed to see
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --- first stages ----------------------------------------------------------------------


class TfIdf:
    name = "TF-IDF"

    def fit(self, docs):
        from shared.benchmark import tokenize
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


class Random:
    """A deliberately terrible first stage.

    Included to answer the strongest form of the question: if a reranker can rescue
    *this*, the first stage never mattered. It is the control the comparison needs.
    """

    name = "random (control)"

    def fit(self, docs):
        self.n = len(docs)
        self.rng = np.random.default_rng(0)
        return self

    def score(self, query):
        return self.rng.random(self.n)


FIRST_STAGES = [BM25, TfIdf, BGE, Random]


# --- metrics ------------------------------------------------------------------------------


def metrics(order: np.ndarray, gold: set[int]) -> dict:
    out = {f"recall@{k}": sum(1 for i in order[:k] if i in gold) / len(gold) for k in KS}
    out[f"recall@{DEPTH}"] = sum(1 for i in order[:DEPTH] if i in gold) / len(gold)
    out["mrr"] = next((1.0 / rank for rank, i in enumerate(order[:100], start=1) if i in gold), 0.0)
    return out


def main(n_queries: int = 300) -> None:
    doc_ids, docs, queries = build(n_queries)
    stats = coverage(doc_ids, queries)
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    print(f"corpus: {stats['documents']} docs, {stats['queries']} queries, depth={DEPTH}\n")

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(CROSS_ENCODER, max_length=512)

    rows = []
    for cls in FIRST_STAGES:
        stage = cls().fit(docs)
        before = dict.fromkeys([f"recall@{k}" for k in (*KS, DEPTH)] + ["mrr"], 0.0)
        after = dict.fromkeys([f"recall@{k}" for k in (*KS, DEPTH)] + ["mrr"], 0.0)

        started = time.time()
        for q in queries:
            gold = {title_to_idx[t] for t in q.gold_titles if t in title_to_idx}
            order = np.argsort(-stage.score(q.question))
            for key, value in metrics(order, gold).items():
                before[key] += value

            # The reranker only ever sees the top DEPTH. Everything below is unreachable.
            candidates = order[:DEPTH]
            scores = reranker.predict(
                [(q.question, docs[i]) for i in candidates],
                batch_size=64,
                show_progress_bar=False,
            )
            reranked = candidates[np.argsort(-np.asarray(scores))]
            for key, value in metrics(reranked, gold).items():
                after[key] += value
        seconds = time.time() - started

        row = {
            "first_stage": cls.name,
            "seconds": round(seconds, 1),
            **{f"before_{k}": v / len(queries) for k, v in before.items()},
            **{f"after_{k}": v / len(queries) for k, v in after.items()},
        }
        # What fraction of the reachable gold did the reranker actually pull into the top 10?
        ceiling = row[f"before_recall@{DEPTH}"]
        row["ceiling"] = ceiling
        row["conversion"] = row["after_recall@10"] / ceiling if ceiling else 0.0
        rows.append(row)
        print(
            f"  {cls.name:18} r@10 {row['before_recall@10']:.3f} -> "
            f"{row['after_recall@10']:.3f}   ceiling(r@{DEPTH}) {ceiling:.3f}   "
            f"{seconds:.1f}s"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rerank.json").write_text(
        json.dumps(
            {"corpus": stats, "depth": DEPTH, "model": CROSS_ENCODER, "rows": rows}, indent=2
        ),
        encoding="utf-8",
    )

    print(
        f"\n{'first stage':18} {'r@10 before':>12} {'r@10 after':>11} {'change':>8} "
        f"{'ceiling':>9} {'converted':>10}"
    )
    print("-" * 76)
    for r in rows:
        delta = r["after_recall@10"] - r["before_recall@10"]
        print(
            f"{r['first_stage']:18} {r['before_recall@10']:12.3f} "
            f"{r['after_recall@10']:11.3f} {delta:+8.3f} {r['ceiling']:9.3f} "
            f"{r['conversion']:10.1%}"
        )

    real = [r for r in rows if r["first_stage"] != "random (control)"]
    spread_before = max(r["before_recall@10"] for r in real) - min(
        r["before_recall@10"] for r in real
    )
    spread_after = max(r["after_recall@10"] for r in real) - min(r["after_recall@10"] for r in real)
    print(
        f"\n  spread between real first stages: {spread_before:.3f} before rerank, "
        f"{spread_after:.3f} after."
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
