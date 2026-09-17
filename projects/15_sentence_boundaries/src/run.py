"""How much do four reasonable sentence splitters disagree, and about what?

    python src/run.py            # the full corpus
    python src/run.py --quick    # a tenth

**What the reference is, and is not.** HotpotQA ships each paragraph pre-split into
sentences. That segmentation was produced by a tool, not by a person, so this project
measures **agreement with a reference segmentation** and not accuracy. Nothing here can
say which splitter is right, because nothing here knows.

That is still worth measuring, and arguably more honestly than a gold-standard score would
be. Sentence segmentation is treated as settled preprocessing; if two reasonable
implementations disagree on a material share of boundaries, then the choice of splitter is
a parameter of every downstream result, and it is one that is almost never reported.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
HUB = Path.home() / ".cache" / "huggingface" / "hub"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import splitters


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def load_paragraphs(limit: int) -> list[tuple[str, list[int]]]:
    """Paragraphs with the reference boundary offsets implied by HotpotQA's own split."""
    import pandas as pd

    hits = sorted(
        glob.glob(
            str(
                HUB
                / "datasets--hotpotqa--hotpot_qa"
                / "snapshots"
                / "*"
                / "**"
                / "validation-*.parquet"
            ),
            recursive=True,
        )
    )
    if not hits:
        raise FileNotFoundError("HotpotQA not found in the Hugging Face cache.")

    out: list[tuple[str, list[int]]] = []
    seen: set[str] = set()
    for context in pd.read_parquet(hits[0]).head(limit)["context"]:
        for title, sentences in zip(list(context["title"]), list(context["sentences"])):
            key = str(title)
            if key in seen:
                continue
            seen.add(key)
            parts = [str(s) for s in sentences]
            text = "".join(parts)
            boundaries, offset = [], 0
            for part in parts[:-1]:
                offset += len(part)
                boundaries.append(offset)
            if len(text) > 40 and boundaries:
                out.append((text, boundaries))
    return out


def score(predicted: list[int], reference: list[int], tolerance: int = 1) -> tuple[int, int, int]:
    """True positives, false positives and false negatives, within ``tolerance`` characters.

    The tolerance exists because HotpotQA's sentences carry a leading space, so a boundary
    the reference puts before the space and a splitter puts after it are the same decision
    written differently. Scoring those as errors would measure whitespace convention.
    """
    remaining = list(reference)
    hits = 0
    for position in predicted:
        for candidate in remaining:
            if abs(candidate - position) <= tolerance:
                remaining.remove(candidate)
                hits += 1
                break
    return hits, len(predicted) - hits, len(remaining)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    started = time.time()
    rule("1. Corpus and reference")
    paragraphs = load_paragraphs(700 if args.quick else 7_405)
    total_boundaries = sum(len(b) for _, b in paragraphs)
    print(f"  {len(paragraphs):,} paragraphs, {total_boundaries:,} reference boundaries")
    print(
        "\n  The reference is HotpotQA's own sentence split, produced by a tool rather than\n"
        "  by a person. What follows is agreement with that segmentation, not accuracy."
    )

    learned = splitters.learn_abbreviations([text for text, _ in paragraphs])
    print(f"\n  learned abbreviations: {len(learned):,} tokens bound to a trailing period")
    sample = sorted(learned)[:14]
    print(f"  sample: {', '.join(sample)}")
    print(f"  supplied list for comparison: {len(splitters.COMMON_ABBREVIATIONS)} tokens")

    rule("2. Agreement with the reference")
    print(f"  {'splitter':14} {'precision':>10} {'recall':>9} {'F1':>8} {'predicted':>11}")

    predictions: dict[str, list[list[int]]] = {}
    rows = []
    for name in splitters.SPLITTERS:
        tp = fp = fn = 0
        per_paragraph = []
        for text, reference in paragraphs:
            if name == "naive":
                found = splitters.naive(text)
            elif name == "abbreviation":
                found = splitters.abbreviation(text)
            elif name == "trained":
                found = splitters.trained(text, learned)
            else:
                found = splitters.conservative(text)
            per_paragraph.append(found)
            a, b, c = score(found, reference)
            tp, fp, fn = tp + a, fp + b, fn + c
        predictions[name] = per_paragraph

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "splitter": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted": tp + fp,
            }
        )
        print(f"  {name:14} {precision:>10.3f} {recall:>9.3f} {f1:>8.3f} {tp + fp:>11,}")

    best = max(rows, key=lambda r: r["f1"])
    worst = min(rows, key=lambda r: r["f1"])
    print(
        f"\n  Best {best['splitter']} at F1 {best['f1']:.3f}, worst {worst['splitter']} at "
        f"{worst['f1']:.3f}.\n  A spread of {best['f1'] - worst['f1']:.3f} between four "
        f"implementations that a pipeline\n  description would all call 'sentence splitting'."
    )

    rule("3. What the disagreements are about")
    reference_sets = [set(b) for _, b in paragraphs]
    taxonomy: dict[str, Counter] = {}
    for name in splitters.SPLITTERS:
        counts: Counter = Counter()
        for (text, _), found, reference in zip(paragraphs, predictions[name], reference_sets):
            for position in found:
                if not any(abs(r - position) <= 1 for r in reference):
                    counts[splitters.classify_disagreement(text, position)] += 1
        taxonomy[name] = counts

    categories = sorted({c for counts in taxonomy.values() for c in counts})
    print("  false boundaries by construction\n")
    print(f"  {'construction':26}" + "".join(f"{n:>14}" for n in splitters.SPLITTERS))
    for category in categories:
        print(
            f"  {category:26}"
            + "".join(f"{taxonomy[n][category]:>14,}" for n in splitters.SPLITTERS)
        )
    print(
        f"  {'TOTAL':26}"
        + "".join(f"{sum(taxonomy[n].values()):>14,}" for n in splitters.SPLITTERS)
    )

    naive_top = taxonomy["naive"].most_common(1)
    if naive_top:
        category, count = naive_top[0]
        share = count / max(1, sum(taxonomy["naive"].values()))
        print(
            f"\n  The naive splitter's commonest false boundary is '{category}' - "
            f"{share * 100:.1f}% of its\n  errors. Errors are not spread over the text; "
            f"they sit on a handful of constructions,\n  and which of those a splitter "
            f"handles is a design decision rather than a quality level."
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "boundaries.json").write_text(
        json.dumps(
            {
                "paragraphs": len(paragraphs),
                "reference_boundaries": total_boundaries,
                "learned_abbreviations": sorted(learned)[:80],
                "learned_count": len(learned),
                "supplied_count": len(splitters.COMMON_ABBREVIATIONS),
                "agreement": rows,
                "taxonomy": {n: dict(c) for n, c in taxonomy.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'boundaries.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
