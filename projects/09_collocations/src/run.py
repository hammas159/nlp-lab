"""Five association measures, one set of bigram counts, and the cutoff that decides them.

python src/run.py            # the full corpus
python src/run.py --quick    # a tenth of it
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

import association
import bigrams

RESULTS = Path(__file__).resolve().parent.parent / "results"
CUTOFFS = (1, 2, 5, 10, 25, 50)
TOP_N = 20


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 76)}", flush=True)


def top_pairs(
    data: bigrams.Bigrams, measure: str, n: int = TOP_N
) -> list[tuple[str, str, float, int]]:
    scores = association.score(measure, data.a, data.b, data.c, data.d)
    order = np.argsort(-scores)[:n]
    return [(data.pairs[i][0], data.pairs[i][1], float(scores[i]), int(data.a[i])) for i in order]


def overlap(left: list, right: list) -> float:
    a = {(w1, w2) for w1, w2, _, _ in left}
    b = {(w1, w2) for w1, w2, _, _ in right}
    return len(a & b) / len(a | b) if a | b else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    started = time.time()
    rule("1. Corpus and bigram counts")
    _, docs, _ = build(700 if args.quick else 7_405)
    data = bigrams.from_documents(docs)
    for key, value in data.stats.items():
        print(f"  {key:22} {value:,}" if isinstance(value, int) else f"  {key:22} {value:.4f}")

    print(
        f"\n  {data.stats['hapax_share'] * 100:.1f}% of bigram types occur exactly once. "
        f"That population is\n  what PMI ranks highest, and it is most of the data."
    )

    rule("2. What each measure puts at the top, with no frequency cutoff")
    unfiltered = {m: top_pairs(data, m) for m in association.MEASURES}
    for measure in association.MEASURES:
        rows = unfiltered[measure]
        counts = [c for _, _, _, c in rows]
        sample = ", ".join(f"{w1} {w2}" for w1, w2, _, _ in rows[:4])
        print(
            f"  {measure:8} median count of its top {TOP_N}: {int(np.median(counts)):>6,}   {sample}"
        )

    print("\n  Jaccard overlap between the top-20 lists:")
    print(f"  {'':10}" + "".join(f"{m:>10}" for m in association.MEASURES))
    for left in association.MEASURES:
        row = f"  {left:10}"
        for right in association.MEASURES:
            row += f"{overlap(unfiltered[left], unfiltered[right]):>10.2f}"
        print(row)

    rule("3. The cutoff, not the statistic")
    print(
        f"  {'min count':>9} {'bigrams kept':>13}"
        + "".join(f"{m:>10}" for m in association.MEASURES)
    )
    print(f"  {'':9} {'':13}" + "".join(f"{'med. freq':>10}" for _ in association.MEASURES))

    cutoff_rows = []
    previous: dict[str, list] = {}
    for cutoff in CUTOFFS:
        filtered = data.filter_by_count(cutoff)
        if len(filtered) < TOP_N:
            continue
        tops = {m: top_pairs(filtered, m) for m in association.MEASURES}
        medians = {m: int(np.median([c for _, _, _, c in tops[m]])) for m in association.MEASURES}
        churn = {m: overlap(tops[m], previous[m]) for m in tops} if previous else {}
        cutoff_rows.append(
            {
                "min_count": cutoff,
                "bigrams_kept": len(filtered),
                "median_frequency_of_top": medians,
                "overlap_with_previous_cutoff": churn,
                "top": {m: [(w1, w2, s, c) for w1, w2, s, c in tops[m][:10]] for m in tops},
            }
        )
        print(
            f"  {cutoff:>9} {len(filtered):>13,}"
            + "".join(f"{medians[m]:>10,}" for m in association.MEASURES)
        )
        previous = tops

    print("\n  Overlap of each measure's top-20 with its own list at the previous cutoff:")
    print(f"  {'min count':>9}" + "".join(f"{m:>10}" for m in association.MEASURES))
    for row in cutoff_rows:
        if not row["overlap_with_previous_cutoff"]:
            continue
        print(
            f"  {row['min_count']:>9}"
            + "".join(
                f"{row['overlap_with_previous_cutoff'][m]:>10.2f}" for m in association.MEASURES
            )
        )

    rule("4. Is chi-squared even admissible here?")
    invalid = association.expected_cell_below_five(data.a, data.b, data.c, data.d)
    print(
        f"  {invalid.mean() * 100:.1f}% of bigram types have an expected cell below 5, which is\n"
        f"  the condition Pearson's chi-squared needs for its normal approximation.\n"
        f"  Dunning wrote the log-likelihood ratio for exactly this reason."
    )
    for cutoff in (5, 25, 50):
        filtered = data.filter_by_count(cutoff)
        if not len(filtered):
            continue
        share = association.expected_cell_below_five(
            filtered.a, filtered.b, filtered.c, filtered.d
        ).mean()
        print(f"    at min count {cutoff:>3}: {share * 100:>5.1f}% still inadmissible")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "collocations.json").write_text(
        json.dumps(
            {
                "corpus": data.stats,
                "no_cutoff": {m: unfiltered[m] for m in association.MEASURES},
                "by_cutoff": cutoff_rows,
                "chi2_inadmissible_share": float(invalid.mean()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'collocations.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
