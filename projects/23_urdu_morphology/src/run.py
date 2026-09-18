"""Repeat project 18's tokenisation experiment on Urdu, where subwords are supposed to win.

python src/fetch.py --groups 12   # once, to cache the corpus
python src/run.py                 # the comparison
python src/run.py --quick         # 800 documents, two vocabulary sizes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
#: Project 18's tokenizers, imported rather than reimplemented. The entire point is to
#: compare against its English numbers, and a reimplementation would leave any difference
#: ambiguous between "Urdu" and "a different BPE".
sys.path.insert(0, str(HERE.parents[1] / "18_tokenisation" / "src"))

import segmenters as S
from shared.bm25 import BM25

import urdu as U

RESULTS = HERE.parent / "results"
SIZES = (2_000, 4_000, 8_000, 16_000)
QUICK_SIZES = (2_000, 8_000)
SAMPLE_WORDS = 4_000

#: Project 18's English results, for the comparison this project exists to make.
ENGLISH = {
    "word_baseline": 0.865,
    "best_subword": 0.867,
    "rows": {
        (2_000, "BPE"): (1.668, 0.797),
        (8_000, "BPE"): (1.188, 0.862),
        (16_000, "BPE"): (1.080, 0.858),
    },
}


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def recall_at_10(retriever, queries: list[str], gold: list[int]) -> float:
    total = 0.0
    for i, want in enumerate(gold):
        scores = retriever.score(queries[i])
        top = np.argpartition(-scores, 10)[:10]
        total += float(want in top.tolist())
    return total / len(gold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--documents", type=int, default=6_000)
    args = parser.parse_args()
    sizes = QUICK_SIZES if args.quick else SIZES
    started = time.time()

    rule("1. The corpus, and a tokenizer that returns something for it")
    _titles, bodies, queries = U.build(limit=800 if args.quick else args.documents)
    gold = list(range(len(bodies)))

    words = [w for b in bodies for w in U.tokenize(b)]
    types = sorted(set(words))
    print(f"  {len(bodies):,} Urdu Wikipedia articles, title-to-body retrieval")
    print(f"  {len(words):,} word tokens, {len(types):,} word types")
    print(f"  type/token ratio {len(types) / max(1, len(words)):.4f}")
    print(
        "  (Project 18's English corpus was 25,295 types over 277,259 tokens. The ratios are"
        "\n   NOT comparable - this corpus is 17x larger, and project 05 measured exactly how"
        "\n   much the ratio falls with corpus size. Heaps' law makes the bigger corpus look"
        "\n   less diverse whatever the language.)"
    )

    from shared.benchmark import tokenize as english_tokenize

    sample_bodies = bodies[:200]
    urdu_total = sum(len(tokenize_urdu) for tokenize_urdu in (U.tokenize(b) for b in sample_bodies))
    english_total = sum(len(english_tokenize(b)) for b in sample_bodies)
    silent = sum(1 for b in sample_bodies if not english_tokenize(b))
    print(
        f"\n  The lab's shared English tokenizer keeps {english_total:,} of this corpus's "
        f"{urdu_total:,} tokens\n"
        f"  ({english_total / max(1, urdu_total):.1%}) over {len(sample_bodies)} articles, and "
        f"returns something non-empty for\n"
        f"  {len(sample_bodies) - silent} of them. It matches `[a-z0-9]+`, so what survives is "
        "stray Latin and digits -\n"
        "  'kh', 'mi', '1922'. An index built with it would look populated and score non-zero\n"
        "  while holding almost none of the text. That is worse than an empty list, which at\n"
        "  least fails loudly."
    )

    word_recall = recall_at_10(BM25(tokenize=U.tokenize).fit(bodies), queries, gold)
    print(f"\n  BM25 over Urdu words, no subwords: recall@10 {word_recall:.3f}")
    print(f"  Project 18's English word baseline, same measure: {ENGLISH['word_baseline']:.3f}")

    counts = Counter(words)
    rng = np.random.default_rng([0, 0x0072_6475])
    population = [w for w, _ in counts.most_common()]
    weights = np.array([counts[w] for w in population], dtype=np.float64)
    weights /= weights.sum()
    sample = list(
        rng.choice(population, size=min(SAMPLE_WORDS, len(population)), replace=True, p=weights)
    )

    rule("2. Three subword algorithms at matched vocabulary sizes")
    print(
        f"  {'tokenizer':12} {'vocab':>7} {'fertility':>10} {'intact':>8} {'[UNK]':>7} "
        f"{'recall@10':>11}"
    )
    rows = []
    for size in sizes:
        for name in S.TRAINERS:
            segmenter = S.train(name, bodies, size)
            recall = recall_at_10(BM25(tokenize=segmenter.encode).fit(bodies), queries, gold)
            row = {
                "tokenizer": name,
                "requested_vocabulary": size,
                "vocabulary": segmenter.vocabulary_size,
                "fertility": S.fertility(segmenter, sample),
                "intact_rate": S.intact_rate(segmenter, sample),
                "unknown_rate": S.unknown_rate(segmenter, sample),
                "recall@10": recall,
            }
            rows.append(row)
            print(
                f"  {name:12} {row['vocabulary']:>7,} {row['fertility']:>10.3f} "
                f"{row['intact_rate']:>8.3f} {row['unknown_rate']:>7.3f} {recall:>11.3f}",
                flush=True,
            )
    print(
        f"  {'words':12} {len(types):>7,} {1.0:>10.3f} {1.0:>8.3f} {0.0:>7.3f} {word_recall:>11.3f}"
    )

    rule("3. Do subwords earn their keep here?")
    best = max(rows, key=lambda r: r["recall@10"])
    gain = best["recall@10"] - word_recall
    print(
        f"  Best subword: {best['tokenizer']} at {best['vocabulary']:,} pieces, "
        f"recall@10 {best['recall@10']:.3f}"
    )
    print(f"  Plain words:  {word_recall:.3f}")
    print(f"  Subwords are worth {gain:+.3f} on Urdu.")
    print(
        f"  Project 18 measured {ENGLISH['best_subword'] - ENGLISH['word_baseline']:+.3f} on English."
    )

    spread_by_size = {}
    for size in sizes:
        at_size = [r["recall@10"] for r in rows if r["requested_vocabulary"] == size]
        spread_by_size[size] = max(at_size) - min(at_size)
    algorithm_spread = max(spread_by_size.values())
    vocabulary_spread = max(r["recall@10"] for r in rows) - min(r["recall@10"] for r in rows)
    print(
        f"\n  At a fixed vocabulary size the three algorithms differ by at most "
        f"{algorithm_spread:.3f}."
    )
    print(f"  Changing the vocabulary size moves it by {vocabulary_spread:.3f}.")
    print("  Project 18 on English: 0.030 and 0.075.")

    rule("4. How much harder is Urdu to tokenise?")
    print(f"  {'tokenizer':12} {'vocab':>7} {'fertility (ur)':>15} {'fertility (en)':>15}")
    for size, name in ((2_000, "BPE"), (8_000, "BPE"), (16_000, "BPE")):
        mine = next(
            (r for r in rows if r["requested_vocabulary"] == size and r["tokenizer"] == name), None
        )
        if mine and (size, name) in ENGLISH["rows"]:
            english_fertility, _ = ENGLISH["rows"][(size, name)]
            print(f"  {name:12} {size:>7,} {mine['fertility']:>15.3f} {english_fertility:>15.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "urdu.json").write_text(
        json.dumps(
            {
                "documents": len(bodies),
                "word_tokens": len(words),
                "word_types": len(types),
                "word_baseline_recall@10": word_recall,
                "rows": rows,
                "english_reference": {
                    "word_baseline": ENGLISH["word_baseline"],
                    "best_subword": ENGLISH["best_subword"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'urdu.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
