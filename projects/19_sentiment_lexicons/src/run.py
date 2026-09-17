"""Which matters more: the lexicon you pick, or what you do with it?

python src/run.py            # every corpus and lexicon present
python src/run.py --quick    # 2,000 items per corpus
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpora as C
import lexicons as L
import rules as R

RESULTS = Path(__file__).resolve().parent.parent / "results"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def accuracy(texts, labels, rule_name, lexicon) -> tuple[float, float, float]:
    """Overall accuracy, the share of items scored zero, and accuracy where it did fire.

    The third number matters because coverage and precision pull in opposite directions. A
    lexicon that abstains on a third of a balanced corpus still collects about half of those
    by luck, so overall accuracy blends "how often it speaks" with "how often it is right" -
    and those turn out to rank the lexicons differently.
    """
    correct = 0
    silent = 0
    scored_correct = 0
    for text, label in zip(texts, labels):
        value = R.score(rule_name, text, lexicon)
        hit = R.classify(value) == label
        correct += hit
        if value == 0.0:
            silent += 1
        else:
            scored_correct += hit
    spoke = len(texts) - silent
    return (
        correct / len(texts),
        silent / len(texts),
        scored_correct / spoke if spoke else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    limit = 2_000 if args.quick else None
    started = time.time()

    rule("1. What is present")
    lexicons = L.load_all()
    for lex in lexicons:
        print(f"  {lex.name:14} {len(lex):>7,} entries   {lex.note}")
    if not lexicons:
        print("  No lexicons found. See the README for how to fetch them.")
        return

    corpus_names = C.available()
    if not corpus_names:
        print("  No corpora found. See the README for how to fetch them.")
        return

    rule("2. Do the lexicons even disagree?")
    print("  If they agreed, 'which lexicon' would not be a question worth asking.\n")
    overlaps = []
    for i in range(len(lexicons)):
        for j in range(i + 1, len(lexicons)):
            share, count = L.agreement(lexicons[i], lexicons[j])
            overlaps.append(
                {
                    "pair": f"{lexicons[i].name} vs {lexicons[j].name}",
                    "agreement": share,
                    "shared": count,
                }
            )
            print(
                f"  {lexicons[i].name:14} vs {lexicons[j].name:14} "
                f"agree on {share:.1%} of {count:,} shared words"
            )

    all_rows = []
    all_firing = []
    for corpus_name in corpus_names:
        texts, labels = C.CORPORA[corpus_name]()
        if limit and limit < len(texts):
            rng = np.random.default_rng([0, 0x5E17])
            pick = rng.choice(len(texts), size=limit, replace=False)
            texts = [texts[i] for i in pick]
            labels = [labels[i] for i in pick]
        floor = max(np.mean(labels), 1 - np.mean(labels))

        rule(f"3. {corpus_name} - {len(texts):,} items, floor {floor:.3f}")
        header = (
            "  "
            + f"{'lexicon':14}"
            + "".join(f"{r:>16}" for r in R.RULES)
            + f"{'unscored':>10}{'acc|fired':>11}"
        )
        print(header)
        rows = []
        for lex in lexicons:
            scores = []
            silences = []
            fired = []
            for rule_name in R.RULES:
                acc, silent, when_fired = accuracy(texts, labels, rule_name, lex)
                scores.append(acc)
                silences.append(silent)
                fired.append(when_fired)
                rows.append(
                    {
                        "corpus": corpus_name,
                        "lexicon": lex.name,
                        "rule": rule_name,
                        "accuracy": acc,
                        "unscored": silent,
                        "accuracy_when_scored": when_fired,
                    }
                )
            # Unscored is per lexicon, not per rule - it is how often the lexicon matched
            # nothing at all, which is coverage rather than reasoning.
            print(
                f"  {lex.name:14}"
                + "".join(f"{s:>16.3f}" for s in scores)
                + f"{silences[0]:>10.1%}{fired[0]:>11.3f}",
                flush=True,
            )
        print(f"  {'floor':14}" + "".join(f"{floor:>16.3f}" for _ in R.RULES))

        # The comparison the project exists for. Holding one axis fixed, how far can the
        # other one move the number?
        by_rule = {}
        by_lexicon = {}
        for r in rows:
            by_rule.setdefault(r["rule"], []).append(r["accuracy"])
            by_lexicon.setdefault(r["lexicon"], []).append(r["accuracy"])
        lexicon_spread = max(max(v) - min(v) for v in by_rule.values())
        rule_spread = max(max(v) - min(v) for v in by_lexicon.values())
        print(
            f"\n  Holding the rules fixed, changing the lexicon moves accuracy by at most "
            f"{lexicon_spread:.3f}."
        )
        print(
            f"  Holding the lexicon fixed, changing the rules moves it by at most "
            f"{rule_spread:.3f}."
        )
        winner = "the rules" if rule_spread > lexicon_spread else "the lexicon"
        print(f"  The bigger axis on {corpus_name} is {winner}.")

        # The whole claim is that one spread exceeds the other, and both are small. A
        # paired bootstrap over items says whether that ordering survives resampling, or
        # whether it is an artefact of these particular 10,662 sentences.
        correct = {
            (r["lexicon"], r["rule"]): np.array(
                [
                    R.classify(
                        R.score(r["rule"], t, next(x for x in lexicons if x.name == r["lexicon"]))
                    )
                    == y
                    for t, y in zip(texts, labels)
                ]
            )
            for r in rows
        }
        boot_rng = np.random.default_rng([0, 0xB007])
        differences = []
        n = len(texts)
        for _ in range(1_000):
            pick = boot_rng.integers(0, n, size=n)
            acc = {k: float(v[pick].mean()) for k, v in correct.items()}
            per_rule, per_lex = {}, {}
            for (lex_name, rule_name), value in acc.items():
                per_rule.setdefault(rule_name, []).append(value)
                per_lex.setdefault(lex_name, []).append(value)
            differences.append(
                max(max(v) - min(v) for v in per_rule.values())
                - max(max(v) - min(v) for v in per_lex.values())
            )
        low, high = np.percentile(differences, [2.5, 97.5])
        share = float(np.mean(np.array(differences) > 0))
        print(
            f"  Paired bootstrap, 1,000 resamples: lexicon spread minus rule spread is "
            f"{lexicon_spread - rule_spread:+.3f}\n"
            f"  with a 95% interval of [{low:+.3f}, {high:+.3f}]; the lexicon axis is larger "
            f"in {share:.0%} of resamples."
        )

        # "The rules barely matter" and "the rules barely fire" look identical in an
        # accuracy table and mean completely different things. This separates them.
        print("\n  How often does each rule actually change the score, and does it help?\n")
        print(f"  {'rule':16} {'fires on':>10} {'flips the call':>16} {'of those, correct':>19}")
        firing = []
        base = next(iter(R.RULES))
        reference = lexicons[0]
        for rule_name in list(R.RULES)[1:]:
            changed = flipped = flipped_right = 0
            for text, label in zip(texts, labels):
                before = R.score(base, text, reference)
                after = R.score(rule_name, text, reference)
                if before != after:
                    changed += 1
                    if R.classify(before) != R.classify(after):
                        flipped += 1
                        flipped_right += R.classify(after) == label
            firing.append(
                {
                    "corpus": corpus_name,
                    "rule": rule_name,
                    "changes_score": changed / len(texts),
                    "flips_classification": flipped / len(texts),
                    "flip_precision": flipped_right / flipped if flipped else 0.0,
                }
            )
            print(
                f"  {rule_name:16} {changed / len(texts):>9.1%} {flipped / len(texts):>15.1%} "
                f"{(flipped_right / flipped if flipped else 0):>18.1%}"
            )
        print(f"  (measured against '{base}', using {reference.name})")
        all_rows.extend(rows)
        all_firing.extend(firing)

    rule("4. Coverage against precision")
    print("  Sorted by how often each lexicon abstains. If wider coverage were an advantage,")
    print("  these two columns would rise together.\n")
    coverage_rows = []
    for corpus_name in corpus_names:
        baseline = next(iter(R.RULES))
        at_corpus = {
            r["lexicon"]: (r["unscored"], r["accuracy_when_scored"])
            for r in all_rows
            if r["corpus"] == corpus_name and r["rule"] == baseline
        }
        ordered = sorted(at_corpus.items(), key=lambda kv: kv[1][0])
        print(f"  {corpus_name}")
        for name, (unscored, when_fired) in ordered:
            print(
                f"    {name:14} abstains {unscored:>6.1%}   correct where it fired {when_fired:.3f}"
            )
        precisions = [p for _, (_, p) in ordered]
        monotone = all(a < b for a, b in itertools.pairwise(precisions))
        coverage_rows.append({"corpus": corpus_name, "monotone_inverse": monotone})
        print(
            f"    -> precision rises with abstention at every step: {monotone}\n"
            if monotone
            else "    -> not monotone\n"
        )

    rule("5. Does the best lexicon stay the best across domains?")
    for corpus_name in corpus_names:
        best = max((r for r in all_rows if r["corpus"] == corpus_name), key=lambda r: r["accuracy"])
        print(
            f"  {corpus_name:18} best = {best['lexicon']} with {best['rule']} "
            f"({best['accuracy']:.3f})"
        )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sentiment.json").write_text(
        json.dumps(
            {
                "lexicons": [{"name": x.name, "entries": len(x), "note": x.note} for x in lexicons],
                "lexicon_agreement": overlaps,
                "rows": all_rows,
                "rule_firing": all_firing,
                "coverage_vs_precision": coverage_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'sentiment.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
