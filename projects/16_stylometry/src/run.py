"""Attribute Devign functions to their codebase, stripping topic one view at a time.

python src/run.py            # 4,000 functions
python src/run.py --quick    # 800
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribute
import views

RESULTS = Path(__file__).resolve().parent.parent / "results"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
DATASET = "datasets--google--code_x_glue_cc_defect_detection"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def load(limit: int, seed: int = 0) -> tuple[list[str], np.ndarray, list[str]]:
    """Devign test + validation, labelled by codebase rather than by vulnerability."""
    import pandas as pd

    frames = []
    for split in ("test", "validation"):
        hits = sorted(
            glob.glob(
                str(HUB / DATASET / "snapshots" / "*" / "**" / f"{split}-*.parquet"), recursive=True
            )
        )
        if hits:
            frames.append(pd.read_parquet(hits[0]))
    if not frames:
        raise FileNotFoundError(
            "Devign not found in the Hugging Face cache. Fetch it with:\n"
            '  python -c "from datasets import load_dataset; '
            "load_dataset('google/code_x_glue_cc_defect_detection')\""
        )
    frame = pd.concat(frames, ignore_index=True)

    names = sorted(frame["project"].unique())
    index = {n: i for i, n in enumerate(names)}
    texts = [str(f) for f in frame["func"]]
    labels = np.array([index[p] for p in frame["project"]], dtype=np.int64)

    if limit < len(texts):
        rng = np.random.default_rng([seed, 0x57A1])
        pick = rng.choice(len(texts), size=limit, replace=False)
        texts = [texts[i] for i in pick]
        labels = labels[pick]
    return texts, labels, names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--documents", type=int, default=4_000)
    args = parser.parse_args()

    n_docs = 800 if args.quick else args.documents
    started = time.time()
    rng = np.random.default_rng([0, 0x5719])

    rule("1. The task")
    texts, labels, names = load(n_docs)
    counts = np.bincount(labels)
    print(f"  {len(texts):,} Devign C functions, labelled by codebase")
    print(
        "  classes: "
        + ", ".join(
            f"{n} {counts[i]:,} ({counts[i] / len(labels) * 100:.1f}%)" for i, n in enumerate(names)
        )
    )

    order = rng.permutation(len(texts))
    split = int(len(texts) * 0.7)
    train_idx, test_idx = order[:split], order[split:]
    train_labels, test_labels = labels[train_idx], labels[test_idx]
    floor = attribute.majority_baseline(train_labels, test_labels)
    print(f"  70/30 split; majority-class floor on the test half: {floor:.3f}")

    rule("2. Accuracy as topic is stripped away")
    print("  Each view removes more of what the code is *about*, holding the classifier fixed.\n")
    print(f"  {'view':14} {'features':>10} {'Naive Bayes':>13} {'Burrows Delta':>15}")

    rows = []
    for name in views.VIEWS:
        vocabulary = views.vocabulary_of([texts[i] for i in train_idx], name)
        if not vocabulary:
            continue
        documents = [views.view(name, t) for t in texts]
        counts_matrix = attribute.count_matrix(documents, vocabulary)
        freq_matrix = attribute.frequency_matrix(documents, vocabulary)

        nb = attribute.NaiveBayes().fit(counts_matrix[train_idx], train_labels)
        nb_acc = attribute.accuracy(nb.predict(counts_matrix[test_idx]), test_labels)

        delta = attribute.BurrowsDelta().fit(freq_matrix[train_idx], train_labels)
        delta_acc = attribute.accuracy(delta.predict(freq_matrix[test_idx]), test_labels)

        rows.append(
            {
                "view": name,
                "features": len(vocabulary),
                "naive_bayes": nb_acc,
                "burrows_delta": delta_acc,
            }
        )
        print(f"  {name:14} {len(vocabulary):>10,} {nb_acc:>13.3f} {delta_acc:>15.3f}")

    print(f"  {'majority class':14} {'':>10} {floor:>13.3f} {floor:>15.3f}")

    by_view = {r["view"]: r for r in rows}
    lex, struct = by_view.get("lexical"), by_view.get("structural")
    lay = by_view.get("layout")
    if lex and struct:
        drop = lex["naive_bayes"] - struct["naive_bayes"]
        kept = (struct["naive_bayes"] - floor) / max(1e-9, lex["naive_bayes"] - floor)
        print(
            f"\n  Removing identifiers costs {drop:.3f} of accuracy. Of everything the lexical\n"
            f"  view had above the floor, the structural view keeps {kept * 100:.0f}%."
        )
    if lay:
        above = lay["naive_bayes"] - floor
        print(
            f"  Layout alone - line lengths, indentation, brace placement, no token content\n"
            f"  at all - scores {lay['naive_bayes']:.3f}, {above:+.3f} over the floor."
        )

    rule("3. What the two attributors disagree about")
    for r in rows:
        gap = r["naive_bayes"] - r["burrows_delta"]
        print(f"  {r['view']:14} NB - Delta = {gap:+.3f}")
    print(
        "\n  Delta z-scores every feature, so a habit appearing in a tenth of a percent of\n"
        "  tokens counts as much as one appearing in five percent. That is the point of the\n"
        "  method and also why it is sensitive to anything correlated with the author,\n"
        "  including topic."
    )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "stylometry.json").write_text(
        json.dumps(
            {
                "documents": len(texts),
                "classes": {n: int(counts[i]) for i, n in enumerate(names)},
                "majority_floor": floor,
                "views": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'stylometry.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
