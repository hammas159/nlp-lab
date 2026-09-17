"""Score LSA, NMF and LDA on coherence and on a task, and see whether they agree.

python src/run.py            # 2,964 documents, k in {10, 20, 50, 100, 200}
python src/run.py --quick    # a smaller sweep
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

import coherence as coh
import models as M

RESULTS = Path(__file__).resolve().parent.parent / "results"
TOPIC_WORDS = 10
KS = (10, 20, 50, 100, 200)
QUICK_KS = (10, 50)
ABLATION_K = 50


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def recall_at_10(doc_vectors: np.ndarray, query_vectors: np.ndarray, gold: list[set[int]]) -> float:
    """Mean recall@10 with topic vectors as the retrieval representation."""
    documents = M.unit(doc_vectors)
    queries = M.unit(query_vectors)
    total = 0.0
    for i, golds in enumerate(gold):
        if not golds:
            continue
        scores = documents @ queries[i]
        top = np.argpartition(-scores, 10)[:10]
        total += len(golds & set(top.tolist())) / len(golds)
    return total / max(1, sum(1 for g in gold if g))


def matrices(docs: list[str], queries: list[str], drop_stopwords: bool = True):
    """Build both representations: TF-IDF for LSA/NMF, raw counts for LDA.

    `max_df=0.5` and stopword removal are standard topic-model preprocessing, and they are
    not cosmetic. Left in, function words co-occur in nearly every document, so a topic made
    entirely of them scores *high* NPMI - coherence rewards it. Section 5 measures that
    directly rather than leaving the choice implicit.
    """
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    shared = {
        "tokenizer": tokenize,
        "lowercase": False,
        "min_df": 2,
        "max_df": 0.5 if drop_stopwords else 1.0,
        "stop_words": "english" if drop_stopwords else None,
        "token_pattern": None,
    }
    tfidf = TfidfVectorizer(**shared)
    counts = CountVectorizer(**shared)
    return {
        "tfidf": (
            tfidf.fit_transform(docs),
            tfidf.transform(queries),
            tfidf.get_feature_names_out(),
        ),
        "counts": (
            counts.fit_transform(docs),
            counts.transform(queries),
            counts.get_feature_names_out(),
        ),
    }


def coherence_for(terms, docs_tokenised):
    """Vocabulary and boolean co-occurrence for whichever term set is in play."""
    vocabulary = {t: i for i, t in enumerate(terms)}
    counts, presence = coh.document_frequencies(docs_tokenised, vocabulary)
    return vocabulary, counts, presence


def evaluate(model_cls, k, representation, reps, gold, coherence_inputs, seed=0):
    """Fit one model at one k on one representation; return coherence, recall and time."""
    doc_matrix, query_matrix, terms = reps[representation]
    started = time.time()
    model = model_cls(k, seed=seed).fit(doc_matrix)
    fit_seconds = time.time() - started

    topics = model.top_words(list(terms), TOPIC_WORDS)
    vocabulary, counts, presence = coherence_inputs
    score = coh.model_coherence(topics, vocabulary, counts, presence)
    recall = recall_at_10(model.transform(doc_matrix), model.transform(query_matrix), gold)
    return {
        "model": model_cls.name,
        "k": k,
        "representation": representation,
        "coherence": score,
        "recall@10": recall,
        "fit_seconds": round(fit_seconds, 1),
        "topics": topics[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    ks = QUICK_KS if args.quick else KS
    started = time.time()

    rule("1. The corpus and the reference point")
    doc_ids, docs, queries = build(300)
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    gold = [{title_to_idx[t] for t in q.gold_titles if t in title_to_idx} for q in queries]
    questions = [q.question for q in queries]
    print(f"  {len(docs):,} documents, {len(queries)} queries, exactly 2 gold each")

    bm25 = BM25().fit(docs)
    total = 0.0
    for i, golds in enumerate(gold):
        scores = bm25.score(questions[i])
        top = np.argpartition(-scores, 10)[:10]
        total += len(golds & set(top.tolist())) / len(golds)
    bm25_recall = total / len(gold)
    print(f"  BM25 on the same corpus and queries: recall@10 {bm25_recall:.3f}")
    print("  Every topic-model number below is read against that, not against zero.")

    reps = matrices(docs, questions)
    terms = list(reps["counts"][2])
    tokenised = [tokenize(d) for d in docs]
    coherence_inputs = coherence_for(terms, tokenised)
    print(f"  vocabulary {len(terms):,} terms (min_df=2, max_df=0.5, English stopwords removed)")
    print("  coherence is NPMI over the top 10 words of each topic, scored on this corpus")

    rule("2. Coherence and retrieval, over a sweep of k")
    print(f"  {'model':20} {'k':>5} {'coherence':>11} {'recall@10':>11} {'fit':>8}")
    rows = []
    for k in ks:
        for model_cls in M.MODELS:
            row = evaluate(model_cls, k, model_cls.needs, reps, gold, coherence_inputs)
            rows.append(row)
            print(
                f"  {row['model']:20} {k:>5} {row['coherence']:>11.3f} "
                f"{row['recall@10']:>11.3f} {row['fit_seconds']:>7.1f}s",
                flush=True,
            )

    rule("3. Do the two metrics pick the same winner?")
    disagreements = 0
    for k in ks:
        at_k = [r for r in rows if r["k"] == k]
        best_coherence = max(at_k, key=lambda r: r["coherence"])["model"]
        best_recall = max(at_k, key=lambda r: r["recall@10"])["model"]
        flag = "" if best_coherence == best_recall else "   <- disagree"
        disagreements += best_coherence != best_recall
        print(f"  k={k:<4} coherence says {best_coherence:20} task says {best_recall:20}{flag}")
    print(f"\n  The two metrics disagree at {disagreements} of {len(ks)} settings.")

    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    if len(rows) > 3:
        flat = [(r["coherence"], r["recall@10"]) for r in rows]
        c = np.array([x for x, _ in flat])
        v = np.array([y for _, y in flat])
        correlation = float(np.corrcoef(c, v)[0, 1]) if c.std() > 0 and v.std() > 0 else 0.0
        print(
            f"  Across all {len(rows)} fits, coherence and recall@10 correlate at r = {correlation:+.3f}."
        )

    rule(f"4. What the input representation costs (k={ABLATION_K})")
    print("  LDA is defined over counts; LSA and NMF are conventionally given TF-IDF.")
    print("  Swapping them is a common tutorial shortcut. This is the price.\n")
    print(f"  {'model':20} {'representation':>15} {'coherence':>11} {'recall@10':>11}")
    ablation = []
    for model_cls in M.MODELS:
        for representation in ("tfidf", "counts"):
            row = evaluate(model_cls, ABLATION_K, representation, reps, gold, coherence_inputs)
            native = " (as defined)" if representation == model_cls.needs else ""
            ablation.append(row)
            print(
                f"  {row['model']:20} {representation:>15} {row['coherence']:>11.3f} "
                f"{row['recall@10']:>11.3f}{native}",
                flush=True,
            )

    rule(f"5. What coherence does when the stopwords are left in (k={ABLATION_K})")
    print("  Function words co-occur in almost every document, so a topic made of nothing")
    print("  but them has near-perfect co-occurrence statistics.\n")
    raw_reps = matrices(docs, questions, drop_stopwords=False)
    raw_terms = list(raw_reps["counts"][2])
    raw_inputs = coherence_for(raw_terms, tokenised)
    print(f"  {'model':20} {'filtered':>10} {'stopwords in':>14} {'change':>9}")
    stopword_rows = []
    for model_cls in M.MODELS:
        raw = evaluate(model_cls, ABLATION_K, model_cls.needs, raw_reps, gold, raw_inputs)
        clean = next(r for r in rows if r["model"] == model_cls.name and r["k"] == ABLATION_K)
        delta = raw["coherence"] - clean["coherence"]
        stopword_rows.append({**raw, "filtered_coherence": clean["coherence"]})
        print(
            f"  {model_cls.name:20} {clean['coherence']:>10.3f} "
            f"{raw['coherence']:>14.3f} {delta:>+9.3f}"
        )
    print(f"\n  vocabulary grows from {len(terms):,} to {len(raw_terms):,} terms.")

    rule("6. Three topics, so the numbers are not the only evidence")
    for r in rows:
        if r["k"] == ABLATION_K:
            print(f"\n  {r['model']} (k={ABLATION_K}, coherence {r['coherence']:.3f}):")
            for topic in r["topics"]:
                print(f"    {' '.join(topic)}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "topic_models.json").write_text(
        json.dumps(
            {
                "documents": len(docs),
                "queries": len(queries),
                "vocabulary": len(terms),
                "bm25_recall@10": bm25_recall,
                "sweep": rows,
                "representation_ablation": ablation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'topic_models.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
