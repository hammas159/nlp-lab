"""Choose k by silhouette, then check it against the labels.

python src/run.py            # 1,500 functions
python src/run.py --quick    # 500
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cluster
import features

RESULTS = Path(__file__).resolve().parent.parent / "results"
K_VALUES = (2, 3, 4, 5, 6, 8, 10)
SEEDS = (0, 1, 2)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--documents", type=int, default=1_500)
    args = parser.parse_args()

    n_docs = 500 if args.quick else args.documents
    started = time.time()

    rule("1. Data and features")
    texts, truth, names = features.load_devign(limit=n_docs)
    matrix, vocabulary = features.tfidf(texts)
    counts = np.bincount(truth)
    print(f"  {len(texts):,} Devign functions, {len(vocabulary):,} terms after pruning")
    print(
        "  true classes: "
        + ", ".join(
            f"{name} {counts[i]:,} ({counts[i] / len(truth) * 100:.1f}%)"
            for i, name in enumerate(names)
        )
    )
    print(f"  the true number of clusters is {len(names)}")

    rule("2. Spherical k-means: what silhouette chooses, and what the labels say")
    print(f"  {'k':>3} {'silhouette':>11} {'ARI':>8} {'purity':>8}   (mean of 3 seeds)")

    rows = []
    for k in K_VALUES:
        sil, ari, pur = [], [], []
        for seed in SEEDS:
            result = cluster.kmeans(matrix, k, spherical=True, seed=seed)
            sil.append(cluster.silhouette(matrix, result.labels))
            ari.append(cluster.adjusted_rand(result.labels, truth))
            pur.append(cluster.purity(result.labels, truth))
        rows.append(
            {
                "k": k,
                "silhouette": float(np.mean(sil)),
                "ari": float(np.mean(ari)),
                "purity": float(np.mean(pur)),
                "silhouette_sd": float(np.std(sil)),
                "ari_sd": float(np.std(ari)),
            }
        )
        print(f"  {k:>3} {np.mean(sil):>11.4f} {np.mean(ari):>8.4f} {np.mean(pur):>8.4f}")

    by_silhouette = max(rows, key=lambda r: r["silhouette"])
    by_ari = max(rows, key=lambda r: r["ari"])
    print(
        f"\n  silhouette is maximised at k = {by_silhouette['k']}; "
        f"agreement with the labels is maximised at k = {by_ari['k']}; "
        f"the truth is k = {len(names)}."
    )
    print(
        f"  purity rises monotonically from {rows[0]['purity']:.3f} to {rows[-1]['purity']:.3f} "
        f"as k grows,\n  which is what makes it useless for choosing k - it reaches 1.0 when "
        f"every point is its own cluster."
    )

    rule("3. Spherical against Euclidean, on the same unit-normalised vectors")
    print(
        "  Both are 'cosine k-means': the input is unit length, so assignment by maximum dot\n"
        "  product and assignment by minimum squared distance agree. They differ only in\n"
        "  whether the centroid is re-normalised after each update.\n"
    )
    print(f"  {'k':>3} {'ARI spherical':>14} {'ARI euclidean':>14} {'label agreement':>16}")

    comparison = []
    for k in K_VALUES:
        agreements, spherical_ari, euclidean_ari = [], [], []
        for seed in SEEDS:
            a = cluster.kmeans(matrix, k, spherical=True, seed=seed)
            b = cluster.kmeans(matrix, k, spherical=False, seed=seed)
            spherical_ari.append(cluster.adjusted_rand(a.labels, truth))
            euclidean_ari.append(cluster.adjusted_rand(b.labels, truth))
            # How much the two partitions agree with *each other*, chance-corrected.
            agreements.append(cluster.adjusted_rand(a.labels, b.labels))
        comparison.append(
            {
                "k": k,
                "ari_spherical": float(np.mean(spherical_ari)),
                "ari_euclidean": float(np.mean(euclidean_ari)),
                "agreement_between_them": float(np.mean(agreements)),
            }
        )
        print(
            f"  {k:>3} {np.mean(spherical_ari):>14.4f} {np.mean(euclidean_ari):>14.4f} "
            f"{np.mean(agreements):>16.4f}"
        )

    worst = min(comparison, key=lambda r: r["agreement_between_them"])
    print(
        f"\n  At k = {worst['k']} the two partitions agree with each other only "
        f"{worst['agreement_between_them']:.3f}.\n  Skipping the centroid re-normalisation "
        f"does not approximate spherical k-means; it runs a different algorithm."
    )

    rule("4. How much does the seed decide?")
    print(f"  {'k':>3} {'ARI mean':>10} {'ARI sd':>9} {'silhouette sd':>15}")
    for row in rows:
        print(
            f"  {row['k']:>3} {row['ari']:>10.4f} {row['ari_sd']:>9.4f} "
            f"{row['silhouette_sd']:>15.4f}"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "clustering.json").write_text(
        json.dumps(
            {
                "data": {
                    "documents": len(texts),
                    "vocabulary": len(vocabulary),
                    "classes": {name: int(counts[i]) for i, name in enumerate(names)},
                    "true_k": len(names),
                },
                "by_k": rows,
                "spherical_vs_euclidean": comparison,
                "chosen_by_silhouette": by_silhouette["k"],
                "chosen_by_ari": by_ari["k"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'clustering.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
