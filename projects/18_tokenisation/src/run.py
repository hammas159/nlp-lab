"""Three subword algorithms, one corpus, matched vocabulary sizes - scored on a task.

python src/run.py            # vocabularies of 2k, 4k, 8k, 16k
python src/run.py --quick    # 2k and 8k only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import build, tokenize
from shared.bm25 import BM25

import segmenters as S

RESULTS = Path(__file__).resolve().parent.parent / "results"
SIZES = (2_000, 4_000, 8_000, 16_000)
QUICK_SIZES = (2_000, 8_000)
AGREEMENT_SIZE = 8_000
SAMPLE_WORDS = 4_000


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def recall_at_10(retriever, questions: list[str], gold: list[set[int]]) -> float:
    total = 0.0
    for i, golds in enumerate(gold):
        if not golds:
            continue
        scores = retriever.score(questions[i])
        top = np.argpartition(-scores, 10)[:10]
        total += len(golds & set(top.tolist())) / len(golds)
    return total / max(1, sum(1 for g in gold if g))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    sizes = QUICK_SIZES if args.quick else SIZES
    started = time.time()

    rule("1. The corpus and the baseline everybody skips")
    doc_ids, docs, queries = build(300)
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    gold = [{title_to_idx[t] for t in q.gold_titles if t in title_to_idx} for q in queries]
    questions = [q.question for q in queries]

    words = [w for d in docs for w in tokenize(d)]
    types = sorted(set(words))
    print(f"  {len(docs):,} documents, {len(words):,} word tokens, {len(types):,} word types")
    print(f"  {len(queries)} queries, exactly 2 gold each")

    word_recall = recall_at_10(BM25().fit(docs), questions, gold)
    print(f"\n  BM25 over plain words, no subwords at all: recall@10 {word_recall:.3f}")
    print("  Every row below is read against that. A subword vocabulary has to earn its place.")

    # A frequency-weighted sample of the vocabulary: intrinsic metrics computed over unique
    # types describe the dictionary, while what a tokenizer actually does to a corpus is
    # dominated by the words that occur often.
    counts = Counter(words)
    rng = np.random.default_rng([0, 0x70CE_4159])
    population = [w for w, _ in counts.most_common()]
    weights = np.array([counts[w] for w in population], dtype=np.float64)
    weights /= weights.sum()
    sample = list(
        rng.choice(population, size=min(SAMPLE_WORDS, len(population)), replace=True, p=weights)
    )

    rule("2. Intrinsic measurements, and the task, at matched vocabulary sizes")
    print("  fertility = pieces per word, 1.000 means nothing was ever split.\n")
    print(
        f"  {'tokenizer':12} {'vocab':>7} {'fertility':>10} {'intact':>8} "
        f"{'[UNK]':>7} {'recall@10':>11} {'train':>8}"
    )

    rows = []
    trained: dict[tuple[str, int], S.Segmenter] = {}
    for size in sizes:
        for name in S.TRAINERS:
            began = time.time()
            segmenter = S.train(name, docs, size)
            train_seconds = time.time() - began
            trained[(name, size)] = segmenter

            retriever = BM25(tokenize=segmenter.encode).fit(docs)
            recall = recall_at_10(retriever, questions, gold)
            row = {
                "tokenizer": name,
                "requested_vocabulary": size,
                "vocabulary": segmenter.vocabulary_size,
                "fertility": S.fertility(segmenter, sample),
                "intact_rate": S.intact_rate(segmenter, sample),
                "unknown_rate": S.unknown_rate(segmenter, sample),
                "recall@10": recall,
                "train_seconds": round(train_seconds, 1),
            }
            rows.append(row)
            print(
                f"  {name:12} {row['vocabulary']:>7,} {row['fertility']:>10.3f} "
                f"{row['intact_rate']:>8.3f} {row['unknown_rate']:>7.3f} "
                f"{recall:>11.3f} {row['train_seconds']:>7.1f}s",
                flush=True,
            )

    rule("3. Does any intrinsic measurement predict the task?")
    fert = np.array([r["fertility"] for r in rows])
    intact = np.array([r["intact_rate"] for r in rows])
    vocab = np.array([r["vocabulary"] for r in rows], dtype=np.float64)
    recall = np.array([r["recall@10"] for r in rows])
    correlations = {}
    unknown = np.array([r["unknown_rate"] for r in rows])
    for label, series in (
        ("fertility", fert),
        ("intact rate", intact),
        ("[UNK] rate", unknown),
        ("vocabulary size", vocab),
    ):
        r = float(np.corrcoef(series, recall)[0, 1]) if series.std() > 0 else 0.0
        correlations[label] = r
        print(f"  {label:18} vs recall@10:  r = {r:+.3f}")

    # A correlation across the whole grid mostly measures the vocabulary knob, since every
    # intrinsic number moves with it. The decision-relevant question is narrower: at a size
    # you have already chosen, does the intrinsic metric pick the algorithm the task picks?
    print("\n  Winner agreement at a fixed vocabulary size - where the algorithm is the")
    print("  only variable, and the only place an intrinsic metric could guide a choice:\n")
    agreements_by_metric = {}
    for label, key, better_low in (
        ("fertility", "fertility", True),
        ("intact rate", "intact_rate", False),
    ):
        matched = 0
        detail = []
        for size in sizes:
            at_size = [r for r in rows if r["requested_vocabulary"] == size]
            pick = (
                min(at_size, key=lambda r: r[key])
                if better_low
                else max(at_size, key=lambda r: r[key])
            )
            best_recall = max(r["recall@10"] for r in at_size)
            winners = {r["tokenizer"] for r in at_size if r["recall@10"] == best_recall}
            matched += pick["tokenizer"] in winners
            detail.append(
                f"{size // 1000}k: {label} says {pick['tokenizer']}, task says {'/'.join(sorted(winners))}"
            )
        agreements_by_metric[label] = matched
        print(f"  {label:14} agrees at {matched} of {len(sizes)} sizes")
        for line in detail:
            print(f"      {line}")

    spread_by_size = {}
    for size in sizes:
        at_size = [r["recall@10"] for r in rows if r["requested_vocabulary"] == size]
        spread_by_size[size] = max(at_size) - min(at_size)
    worst = max(spread_by_size.values())
    print(
        f"\n  At a fixed vocabulary size the three algorithms differ by at most "
        f"{worst:.3f} recall@10."
    )
    print(f"  Changing the vocabulary size moves it by {recall.max() - recall.min():.3f}.")

    rule(f"4. How much do the three actually disagree? (vocab {AGREEMENT_SIZE:,})")
    names = list(S.TRAINERS)
    agreements = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = trained[(names[i], AGREEMENT_SIZE)], trained[(names[j], AGREEMENT_SIZE)]
            share = S.segmentation_agreement(a, b, sample)
            agreements.append({"pair": f"{names[i]} vs {names[j]}", "agreement": share})
            print(
                f"  {names[i]:12} vs {names[j]:12} identical segmentation on {share:.1%} of words"
            )

    rule("5. What they do to a word none of them saw whole")
    for word in ("entanglement", "microbiologist", "unhappiness", "reproducibility"):
        print(f"\n  {word}")
        for name in names:
            pieces = trained[(name, AGREEMENT_SIZE)].encode(word)
            print(f"    {name:12} {' | '.join(pieces)}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tokenisation.json").write_text(
        json.dumps(
            {
                "documents": len(docs),
                "word_tokens": len(words),
                "word_types": len(types),
                "word_baseline_recall@10": word_recall,
                "rows": rows,
                "correlations": correlations,
                "winner_agreement_at_fixed_size": agreements_by_metric,
                "agreement": agreements,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'tokenisation.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
