"""Score five sense disambiguators against both baselines, and see which one you report.

python src/run.py            # all of SemCor
python src/run.py --quick    # 60 files
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import disambiguate as D
import semcor as S

RESULTS = Path(__file__).resolve().parent.parent / "results"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def split_by_document(sentences, holdout: float = 0.3, seed: int = 0):
    """Split on documents, not on tokens.

    A token-level split would put other sentences from the same document into training,
    which hands `one sense per discourse` most of its answer and makes every context-based
    method look better than it is.
    """
    documents = sorted({t.document for s in sentences for t in s})
    rng = np.random.default_rng([seed, 0x5E4C])
    order = rng.permutation(len(documents))
    cut = int(len(documents) * (1 - holdout))
    train_docs = {documents[i] for i in order[:cut]}
    train = [s for s in sentences if s and s[0].document in train_docs]
    test = [s for s in sentences if s and s[0].document not in train_docs]
    return train, test, len(train_docs), len(documents) - len(train_docs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    started = time.time()

    rule("1. The corpus")
    sentences, stats = S.load(limit_files=60 if args.quick else None)
    print(
        f"  {stats['files']} files, {stats['sentences']:,} sentences, "
        f"{stats['tagged_tokens']:,} sense-tagged tokens"
    )
    print(
        f"  {stats['multi_sense_tags']:,} tokens carry more than one sense tag; the first is used"
    )

    senses = S.inventory(sentences)
    train, test, n_train, n_test = split_by_document(sentences)
    train_senses = S.inventory(train)

    test_tokens = S.polysemous(test, senses)
    all_test_tokens = [t for s in test for t in s]
    print(f"  split by document: {n_train} train, {n_test} test")
    print(
        f"  test tokens: {len(all_test_tokens):,} tagged, of which {len(test_tokens):,} "
        f"({len(test_tokens) / len(all_test_tokens):.1%}) are polysemous"
    )

    rule("2. Only polysemous words are a disambiguation problem")
    monosemous = len(all_test_tokens) - len(test_tokens)
    print(
        f"  {monosemous:,} test tokens have one observed sense. Every method scores them\n"
        f"  correctly, so including them adds {monosemous / len(all_test_tokens):.1%} to every\n"
        "  number and compresses the differences the evaluation exists to show.\n"
        "  Everything below is on polysemous tokens only."
    )

    sentence_of = {}
    for sentence in test:
        for token in sentence:
            sentence_of[id(token)] = sentence
    truth = [t.sense for t in test_tokens]

    rule("3. Accuracy on polysemous tokens")
    print(f"  {'method':32} {'accuracy':>10} {'vs random':>11} {'vs first sense':>16}")
    rows = []
    scores = {}
    for method_cls in D.methods():
        model = method_cls().fit(train, train_senses)
        if isinstance(model, D.OneSensePerDiscourseOracle):
            model.set_documents(test)
        if isinstance(model, D.ContextOverlapWithDiscourse):
            # Needs the whole document at once to vote over its own predictions.
            resolved = model.predict_document([t for s_ in test for t in s_], sentence_of)
            predictions = [resolved[id(t)] for t in test_tokens]
        else:
            predictions = [model.predict(t, sentence_of[id(t)]) for t in test_tokens]
        acc = D.accuracy(predictions, truth)
        scores[model.name] = acc
        rows.append({"method": model.name, "accuracy": acc})
        print(f"  {model.name:32} {acc:>10.3f}", end="", flush=True)
        print(
            f" {acc - scores.get('random', acc):>+11.3f}"
            f" {acc - scores.get('first sense (WordNet order)', acc):>+16.3f}"
        )

    random_acc = scores["random"]
    mfs_acc = scores["first sense (WordNet order)"]

    rule("4. Which baseline you report decides the story")
    # The oracle is excluded from "methods" - it reads gold labels and is a ceiling, not
    # something anyone could run.
    real = [r for r in rows if "ORACLE" not in r["method"]]
    beat_random = [r for r in real if r["accuracy"] > random_acc and r["method"] != "random"]
    beat_mfs = [
        r for r in real if r["accuracy"] > mfs_acc and r["method"] != "first sense (WordNet order)"
    ]
    n_methods = len(real) - 1
    print(f"  Against random ({random_acc:.3f}): {len(beat_random)} of {n_methods} methods win.")
    print(f"  Against first sense ({mfs_acc:.3f}): {len(beat_mfs)} of {n_methods} methods win.")
    if beat_mfs:
        print("  Winners over MFS: " + ", ".join(r["method"] for r in beat_mfs))
    else:
        print("  Nothing beats the most frequent sense.")
    print(
        f"\n  The most frequent sense is {mfs_acc - random_acc:+.3f} above random on its own,\n"
        "  before any method has done anything at all."
    )

    rule("5. How circular is 'first sense'?")
    trained = scores["trained MFS"]
    print(
        "  WordNet orders senses by frequency, and that ordering was derived from a tagged\n"
        "  corpus. Estimating the most frequent sense from this training split instead gives\n"
        f"  {trained:.3f}, against WordNet's {mfs_acc:.3f} - a difference of {trained - mfs_acc:+.3f}."
    )
    print(
        "  They are close because they are nearly the same statistic, computed twice. That is\n"
        "  not a flaw in either, but it is a reason not to call 'first sense' knowledge-free."
    )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "word_sense.json").write_text(
        json.dumps(
            {
                "corpus": stats,
                "train_documents": n_train,
                "test_documents": n_test,
                "test_tokens": len(all_test_tokens),
                "polysemous_test_tokens": len(test_tokens),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'word_sense.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
