"""Rank seven string measures under two error models, and watch the ranking change.

python src/run.py            # the full sweep
python src/run.py --quick    # fewer words
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

import corrupt
import metrics

RESULTS = Path(__file__).resolve().parent.parent / "results"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def vocabulary(docs: list[str], size: int, min_length: int = 4) -> list[str]:
    """The most frequent alphabetic words of at least ``min_length`` characters.

    Short words are excluded because a three-letter word corrupted once is usually a
    different three-letter word, and the task stops being retrieval and becomes a coin
    toss that no measure can win.
    """
    counts: Counter = Counter()
    for doc in docs:
        counts.update(t for t in tokenize(doc) if t.isalpha() and len(t) >= min_length)
    return [word for word, _ in counts.most_common(size)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--words", type=int, default=2_000, help="candidate vocabulary size")
    parser.add_argument("--trials", type=int, default=600, help="corrupted words per model")
    args = parser.parse_args()

    n_words = 400 if args.quick else args.words
    n_trials = 150 if args.quick else args.trials
    started = time.time()
    rng = np.random.default_rng([0, 0x57A1])

    rule("1. The task")
    _, docs, _ = build(300)
    words = vocabulary(docs, n_words)
    print(f"  candidate list: {len(words):,} words from {len(docs):,} HotpotQA paragraphs")
    print(f"  corrupted words per model: {n_trials:,}, one edit each")
    print(
        "\n  A word is corrupted, and each measure ranks the whole candidate list against\n"
        "  the corruption. Accuracy is how often the original comes back first."
    )

    rule("2. Accuracy by measure and error model")
    print(f"  {'measure':14}" + "".join(f"{m:>12}" for m in corrupt.MODELS) + f"{'gap':>9}")

    targets = [words[int(rng.integers(len(words)))] for _ in range(n_trials)]
    corrupted: dict[str, list[tuple[str, str]]] = {}
    for model in corrupt.MODELS:
        pairs = []
        for word in targets:
            noisy = corrupt.corrupt(model, word, rng)
            pairs.append((word, noisy))
        corrupted[model] = pairs
        unchanged = sum(1 for original, noisy in pairs if original == noisy)
        print(f"  # {model}: {unchanged} of {len(pairs)} words had no applicable corruption")

    rows = []
    for measure in metrics.SIMILARITIES:
        scores = {}
        for model, pairs in corrupted.items():
            correct = 0
            considered = 0
            for original, noisy in pairs:
                if original == noisy:
                    continue
                considered += 1
                best_score, best_words = -1.0, []
                for candidate in words:
                    value = metrics.similarity(measure, noisy, candidate)
                    if value > best_score:
                        best_score, best_words = value, [candidate]
                    elif value == best_score:
                        best_words.append(candidate)
                # A tie is not a hit. The phonetic measures return 1 or 0, so they tie
                # constantly, and counting a tie as correct would score them on the size of
                # the equivalence class rather than on whether they found the word.
                if len(best_words) == 1 and best_words[0] == original:
                    correct += 1
            scores[model] = correct / considered if considered else float("nan")
        gap = scores["typing"] - scores["phonetic"]
        rows.append({"measure": measure, **scores, "gap": gap})
        print(
            f"  {measure:14}"
            + "".join(f"{scores[m]:>12.3f}" for m in corrupt.MODELS)
            + f"{gap:>+9.3f}"
        )

    best_typing = max(rows, key=lambda r: r["typing"])
    best_phonetic = max(rows, key=lambda r: r["phonetic"])
    print(f"\n  Best under typing noise:   {best_typing['measure']} ({best_typing['typing']:.3f})")
    print(
        f"  Best under phonetic noise: {best_phonetic['measure']} ({best_phonetic['phonetic']:.3f})"
    )
    if best_typing["measure"] != best_phonetic["measure"]:
        print(
            "\n  Different measures win under the two models. A paper that corrupts a word\n"
            "  list one way and ranks measures is reporting a property of its corruption."
        )
    else:
        print(
            f"\n  The same measure wins under both. Its margin still moves: "
            f"{best_typing['typing']:.3f} against\n  {best_typing['phonetic']:.3f}, a gap of "
            f"{best_typing['gap']:+.3f} from changing nothing but the noise."
        )

    rule("3. How far the ranking moves")
    by_typing = sorted(rows, key=lambda r: -r["typing"])
    by_phonetic = sorted(rows, key=lambda r: -r["phonetic"])
    print(f"  {'rank':>4}  {'under typing':16}  {'under phonetic':16}")
    for position, (left, right) in enumerate(zip(by_typing, by_phonetic), start=1):
        mark = "  <- same" if left["measure"] == right["measure"] else ""
        print(f"  {position:>4}  {left['measure']:16}  {right['measure']:16}{mark}")

    moved = sum(
        1
        for position, (left, right) in enumerate(zip(by_typing, by_phonetic))
        if left["measure"] != right["measure"]
    )
    print(f"\n  {moved} of {len(rows)} positions hold a different measure.")

    keyboard = next(r for r in rows if r["measure"] == "keyboard")
    keyboard_rank = [r["measure"] for r in by_typing].index("keyboard") + 1
    print(
        f"\n  The measure that encodes the typing error model ranks {keyboard_rank} of "
        f"{len(rows)} on typing\n  noise, at {keyboard['typing']:.3f}. Making near-key "
        f"substitutions cheap forgives the\n  corruption and equally forgives every wrong "
        f"candidate that differs by a near-key\n  substitution. Encoding the error model "
        f"buys tolerance and pays in discrimination."
    )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "similarity.json").write_text(
        json.dumps(
            {
                "candidates": len(words),
                "trials": n_trials,
                "rows": rows,
                "best_under_typing": best_typing["measure"],
                "best_under_phonetic": best_phonetic["measure"],
                "ranking_typing": [r["measure"] for r in by_typing],
                "ranking_phonetic": [r["measure"] for r in by_phonetic],
                "positions_changed": moved,
                "keyboard_rank_under_typing": keyboard_rank,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'similarity.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
