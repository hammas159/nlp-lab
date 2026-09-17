"""Score compression-based classification three ways, and against two learned baselines.

    python src/run.py            # 1,000 reference / 500 evaluation documents
    python src/run.py --quick    # 300 / 150

The distance matrix is the expensive part: one compression per (reference, evaluation) pair.
It is computed once and every decision rule, every k and every tie-breaking policy is scored
against the same matrix, so the differences between them cannot be sampling noise.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baselines
import data
import knn
import ncd

RESULTS = Path(__file__).resolve().parent.parent / "results"
K_VALUES = (1, 2, 3, 5, 11)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 76)}", flush=True)


def distance_matrix(
    reference: data.Split, evaluation: data.Split, algorithm: str = "gzip"
) -> np.ndarray:
    """NCD from every evaluation document to every reference document."""
    ref_lengths = ncd.CompressedLengths(reference.texts, algorithm)
    eval_lengths = ncd.CompressedLengths(evaluation.texts, algorithm)

    out = np.empty((len(evaluation), len(reference)), dtype=np.float64)
    for i, text in enumerate(evaluation.texts):
        c_x = eval_lengths[i]
        for j, other in enumerate(reference.texts):
            out[i, j] = ncd.ncd(text, other, c_x, ref_lengths[j], algorithm)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reference", type=int, default=1_000)
    parser.add_argument("--evaluation", type=int, default=500)
    args = parser.parse_args()

    n_ref = 300 if args.quick else args.reference
    n_eval = 150 if args.quick else args.evaluation
    started = time.time()

    rule("1. Data")
    reference, evaluation = data.load_pair(n_ref, n_eval)
    print(f"  reference (Devign validation): {len(reference):,} functions")
    print(f"  evaluation (Devign test):      {len(evaluation):,} functions")
    print(
        f"  positive rate, reference {reference.labels.mean():.3f}, "
        f"evaluation {evaluation.labels.mean():.3f}"
    )
    pressure = ncd.window_pressure(reference.texts + evaluation.texts)
    print(
        f"  median document {pressure['median_bytes']:,} bytes; a median pair is "
        f"{pressure['median_pair_bytes']:,} against gzip's {pressure['window_bytes']:,} byte window"
    )
    print(f"  worst-case pair exceeds the window: {pressure['worst_case_exceeds_window']}")

    rule("2. The distance matrix")
    step = time.time()
    distances = distance_matrix(reference, evaluation, "gzip")
    gzip_seconds = time.time() - step
    print(
        f"  gzip   {len(evaluation) * len(reference):>9,} pairs [{gzip_seconds:.1f}s]",
        flush=True,
    )

    rule("3. The same neighbours, three ways of turning them into a prediction")
    majority = knn.majority_baseline(reference.labels, evaluation.labels)
    print(f"  majority-class floor: {majority:.3f}\n")
    print(
        f"  {'k':>3}  {'tie rate':>9}  {'oracle_tie (published)':>23} "
        f"{'nearest_tie':>13} {'random_tie':>12}"
    )

    gzip_rows = []
    for k in K_VALUES:
        ties = knn.tie_rate(distances, reference.labels, k)
        scores = {}
        for rule_name in knn.RULES:
            predictions = knn.predict(
                distances,
                reference.labels,
                k,
                rule_name,
                truth=evaluation.labels if rule_name == "oracle_tie" else None,
            )
            scores[rule_name] = knn.accuracy(predictions, evaluation.labels)
        gzip_rows.append({"k": k, "tie_rate": ties, **scores})
        print(
            f"  {k:>3}  {ties:>9.3f}  {scores['oracle_tie']:>23.3f} "
            f"{scores['nearest_tie']:>13.3f} {scores['random_tie']:>12.3f}"
        )

    at_two = next(r for r in gzip_rows if r["k"] == 2)
    gap = at_two["oracle_tie"] - at_two["nearest_tie"]
    print(
        f"\n  At k = 2, resolving ties by looking at the answer reports "
        f"{at_two['oracle_tie']:.3f}.\n  Scored as a classifier it is "
        f"{at_two['nearest_tie']:.3f} - a difference of {gap:.3f}, decided entirely on the\n"
        f"  {at_two['tie_rate'] * 100:.1f}% of documents where the two neighbours disagree."
    )

    rule("4. Does the compressor matter?")
    # Deliberately small. lzma at its default preset costs about ten times gzip per pair,
    # so the full matrix in lzma would run for hours to answer a secondary question.
    n_sub_ref = min(150, len(reference))
    n_sub_eval = min(80, len(evaluation))
    sub_reference = reference.subsample(n_sub_ref, seed=1)
    sub_evaluation = evaluation.subsample(n_sub_eval, seed=1)
    print(
        f"  Measured on a {n_sub_ref} x {n_sub_eval} subsample: at their default settings "
        f"bz2 and lzma\n  cost 3x and 10x gzip per pair, and the full matrix in lzma would "
        f"take hours.\n"
    )
    print(f"  {'compressor':10} {'k=2':>8} {'best k':>8} {'best acc':>10} {'ms/pair':>10}")
    compressor_rows = []
    for algorithm in ("gzip", "bz2", "lzma"):
        step = time.time()
        matrix = distance_matrix(sub_reference, sub_evaluation, algorithm)
        per_pair = (time.time() - step) / (n_sub_ref * n_sub_eval) * 1000
        by_k = {
            k: knn.accuracy(
                knn.predict(matrix, sub_reference.labels, k, "nearest_tie"), sub_evaluation.labels
            )
            for k in K_VALUES
        }
        best_k = max(by_k, key=lambda key: by_k[key])
        compressor_rows.append(
            {"compressor": algorithm, "by_k": by_k, "best_k": best_k, "ms_per_pair": per_pair}
        )
        print(
            f"  {algorithm:10} {by_k[2]:>8.3f} {best_k:>8} {by_k[best_k]:>10.3f} {per_pair:>10.2f}",
            flush=True,
        )

    rule("5. Against baselines that were actually trained")
    baseline_rows = []
    for name, fn in (
        ("multinomial Naive Bayes", baselines.multinomial_nb),
        ("nearest centroid (TF-IDF)", baselines.nearest_centroid),
    ):
        step = time.time()
        predictions = fn(reference.texts, reference.labels, evaluation.texts)
        score = knn.accuracy(predictions, evaluation.labels)
        seconds = time.time() - step
        baseline_rows.append({"name": name, "accuracy": score, "seconds": seconds})
        print(f"  {name:28} {score:>8.3f}   [{seconds:.1f}s to train and predict]")
    print(f"  {'majority class':28} {majority:>8.3f}")
    best_gzip = max(r["nearest_tie"] for r in gzip_rows)
    print(f"  {'best gzip-kNN (any k)':28} {best_gzip:>8.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "classification.json").write_text(
        json.dumps(
            {
                "data": {
                    "reference": len(reference),
                    "evaluation": len(evaluation),
                    "reference_positive_rate": float(reference.labels.mean()),
                    "evaluation_positive_rate": float(evaluation.labels.mean()),
                    "window": pressure,
                },
                "majority_baseline": majority,
                "gzip_by_k": gzip_rows,
                "compressors": compressor_rows,
                "baselines": baseline_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'classification.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
