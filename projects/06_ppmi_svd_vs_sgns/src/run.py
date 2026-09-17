"""The hyperparameter transfer ladder, and the SGNS row it is climbing towards.

    python src/run.py            # the full study
    python src/run.py --quick    # a tenth of the corpus and 150 dimensions

Each rung adds exactly one of the decisions that arrived bundled with word2vec and were
never applied to counting models. Adding them one at a time is the only way to say which
one did the work, which is the whole argument of Levy, Goldberg & Dagan (2015).

Order of operations matters here. SGNS is trained *first*, not because it is the point but
because both sides have to be restricted to one vocabulary before anything is scored -
otherwise the retrieval number is partly a coverage number, and the method with the larger
vocabulary wins for a reason that has nothing to do with its objective.
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

import cooccurrence
import evaluate
import factorize
import pmi
import sgns

RESULTS = Path(__file__).resolve().parent.parent / "results"

#: One rung per decision. Each inherits everything above it and changes one thing, so the
#: delta on each row is attributable to the name on that row.
LADDER = [
    ("PPMI + SVD, as usually taught", {}),
    ("+ dynamic context window", {"dynamic": True}),
    ("+ subsample frequent words", {"do_subsample": True}),
    ("+ context distribution smoothing", {"alpha": pmi.ALPHA_SGNS}),
    ("+ shifted PMI (k = 5)", {"k": 5}),
    ("+ eigenvalue weighting p = 0.5", {"eigenvalue_weight": 0.5}),
    ("+ add context vectors (w + c)", {"add_context": True}),
]

BASE = {
    "dynamic": False,
    "do_subsample": False,
    "alpha": pmi.ALPHA_NONE,
    "k": 1,
    "eigenvalue_weight": 1.0,
    "add_context": False,
}


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 76)}", flush=True)


def cumulative_configs() -> list[tuple[str, dict]]:
    config = dict(BASE)
    out = []
    for label, change in LADDER:
        config = {**config, **change}
        out.append((label, dict(config)))
    return out


def build_counts(tokens: list[str], cache: dict, config: dict, vocab_cap: int) -> tuple:
    key = (config["dynamic"], config["do_subsample"])
    if key not in cache:
        cache[key] = cooccurrence.build(
            tokens,
            min_count=sgns.MIN_COUNT,
            max_size=vocab_cap,
            window=sgns.WINDOW,
            dynamic=config["dynamic"],
            do_subsample=config["do_subsample"],
        )
    return cache[key]


def svd_embedding(
    tokens: list[str], cache: dict, config: dict, vocab_cap: int, dim: int, label: str
) -> tuple[factorize.Embedding, dict, dict]:
    counts, vocab, stats = build_counts(tokens, cache, config, vocab_cap)
    weighted = pmi.pmi_matrix(counts, alpha=config["alpha"], k=config["k"], positive=True)
    embedding = factorize.factorize(
        weighted,
        vocab.words,
        dim=dim,
        eigenvalue_weight=config["eigenvalue_weight"],
        add_context=config["add_context"],
        label=label,
    )
    matrix_stats = {"nnz": int(weighted.nnz), "sparsity": pmi.sparsity(weighted)}
    return embedding, stats, matrix_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--vocab", type=int, default=20_000)
    parser.add_argument("--queries", type=int, default=1_000)
    args = parser.parse_args()

    dim = args.dim or (150 if args.quick else 300)
    rows_limit = 700 if args.quick else 7_405
    started = time.time()

    # ---------------------------------------------------------------- corpus
    rule("1. Corpus")
    doc_ids, docs, queries = build(rows_limit)
    queries = queries[: args.queries]
    tokens = [t for d in docs for t in tokenize(d)]
    print(f"  {len(docs):,} paragraphs, {len(tokens):,} tokens, {len(set(tokens)):,} types")
    print(f"  {len(queries):,} queries, two gold paragraphs each")
    print(f"  embeddings: {dim} dimensions, counting vocabulary capped at {args.vocab:,}")

    cache: dict[tuple, tuple] = {}
    configs = cumulative_configs()

    # ------------------------------------------------- SGNS, and the shared vocabulary
    rule("2. The neural model, and the vocabulary both sides are held to")
    neural_models: dict[int, factorize.Embedding] = {}
    if sgns.available():
        sentences = [tokenize(d) for d in docs]
        for epochs in (5,) if args.quick else (5, 25):
            step = time.time()
            neural_models[epochs] = sgns.train(sentences, dim=dim, epochs=epochs)
            print(
                f"  SGNS, {epochs:>2} epochs: {len(neural_models[epochs].words):,} words "
                f"[{time.time() - step:.1f}s]"
            )
    else:
        print("  gensim is not installed; the SGNS rows are skipped.")
        print("  install it with:  uv sync --extra sgns")

    counting_vocab = build_counts(tokens, cache, configs[0][1], args.vocab)[1].words
    if neural_models:
        any_model = next(iter(neural_models.values()))
        shared = [w for w in counting_vocab if w in any_model]
    else:
        shared = counting_vocab
    print(
        f"  counting side: {len(counting_vocab):,} words, shared vocabulary: {len(shared):,}\n"
        f"  Every embedding below is restricted to those {len(shared):,} words, so no method\n"
        f"  can score by covering text another method cannot read."
    )

    doc_pooler = evaluate.Pooler(shared, docs)
    query_pooler = evaluate.Pooler(shared, [q.question for q in queries])
    pooled = (doc_pooler, query_pooler)
    print(
        f"  {int(query_pooler.empty.sum())} of {len(queries):,} queries contain no word in "
        f"that vocabulary and are scored zero."
    )

    # ------------------------------------------------------------- the ladder
    rule("3. Transferring word2vec's hyperparameters to a counting model")
    print(f"  {'configuration':38} {'recall@10':>10} {'delta':>8} {'nnz':>12} {'seconds':>8}")

    results = []
    baseline = None

    for label, config in configs:
        step = time.time()
        embedding, stats, matrix_stats = svd_embedding(
            tokens, cache, config, args.vocab, dim, label
        )
        scored = evaluate.recall_at_k(
            embedding.restrict_to(shared), doc_ids, docs, queries, pooled=pooled
        )
        if baseline is None:
            baseline = scored["recall"]
        delta = scored["recall"] - baseline

        results.append(
            {
                "label": label,
                "config": config,
                "recall_at_10": scored["recall"],
                "delta_vs_baseline": delta,
                "matrix": matrix_stats,
                "corpus": stats,
                "seconds": round(time.time() - step, 1),
                "_per_query": scored["per_query"],
            }
        )
        print(
            f"  {label:38} {scored['recall']:>10.3f} {delta:>+8.3f} "
            f"{matrix_stats['nnz']:>12,} {results[-1]['seconds']:>8.1f}",
            flush=True,
        )

    # ------------------------------------------------------------- the comparison
    sgns_entry = None
    if neural_models:
        rule("4. Counting against SGNS, on one vocabulary")
        print(f"  {'model':38} {'recall@10':>10}")
        scored_neural = {}
        for epochs, model in neural_models.items():
            scored_neural[epochs] = evaluate.recall_at_k(
                model.restrict_to(shared), doc_ids, docs, queries, pooled=pooled
            )
            print(f"  {f'SGNS, {epochs} epochs':38} {scored_neural[epochs]['recall']:>10.3f}")

        best_epochs = max(scored_neural, key=lambda e: scored_neural[e]["recall"])
        best_neural = scored_neural[best_epochs]
        best_count = max(results, key=lambda r: r["recall_at_10"])

        print(f"\n  best counting model: {best_count['label']} at {best_count['recall_at_10']:.3f}")
        print(f"  best SGNS:           {best_epochs} epochs at {best_neural['recall']:.3f}")
        print(f"  difference: {best_count['recall_at_10'] - best_neural['recall']:+.3f}")

        test = evaluate.paired_bootstrap(best_count["_per_query"], best_neural["per_query"])
        verdict = "REAL" if test["significant"] else "within noise"
        print(
            f"  paired bootstrap over {len(queries):,} queries: {test['mean_difference']:+.3f} "
            f"[{test['ci_low']:+.3f}, {test['ci_high']:+.3f}] p={test['p_value']:.3f} -> {verdict}"
        )

        rule("5. Do the two spaces put the same words next to each other?")
        print(f"  {'counting model':38} {'mean Jaccard@10':>16} {'no overlap':>12}")
        reference = neural_models[best_epochs].restrict_to(shared)
        agreements = {}
        for entry in (results[0], best_count):
            embedding, _, _ = svd_embedding(
                tokens, cache, entry["config"], args.vocab, dim, entry["label"]
            )
            agreement = evaluate.neighbour_agreement(embedding.restrict_to(shared), reference)
            agreements[entry["label"]] = agreement
            print(
                f"  {entry['label']:38} {agreement['mean_jaccard']:>16.3f} "
                f"{agreement['share_with_no_overlap'] * 100:>11.1f}%"
            )

        sgns_entry = {
            "by_epochs": {str(e): scored_neural[e]["recall"] for e in scored_neural},
            "best_epochs": best_epochs,
            "recall_at_10": best_neural["recall"],
            "shared_vocabulary": len(shared),
            "significance_vs_best_count": test,
            "agreement": agreements,
        }

    for entry in results:
        entry.pop("_per_query", None)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "ladder.json").write_text(
        json.dumps(
            {
                "corpus": {
                    "paragraphs": len(docs),
                    "tokens": len(tokens),
                    "types": len(set(tokens)),
                    "queries": len(queries),
                    "queries_with_no_known_words": int(query_pooler.empty.sum()),
                },
                "dim": dim,
                "vocabulary_cap": args.vocab,
                "shared_vocabulary": len(shared),
                "ladder": results,
                "sgns": sgns_entry,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'ladder.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
