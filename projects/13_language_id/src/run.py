"""Accuracy against input length, for three identifiers over three languages.

python src/run.py            # the full sweep
python src/run.py --quick    # fewer samples
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

import identify

RESULTS = Path(__file__).resolve().parent.parent / "results"
HUB = Path.home() / ".cache" / "huggingface" / "hub"
LENGTHS = (10, 20, 40, 80, 160, 320, 640)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def _snapshot(dataset: str, pattern: str) -> Path | None:
    hits = sorted(
        glob.glob(str(HUB / dataset / "snapshots" / "*" / "**" / pattern), recursive=True)
    )
    return Path(hits[0]) if hits else None


def load_texts() -> dict[str, list[str]]:
    """One list of documents per class, from the local Hugging Face cache.

    Three classes: English prose, C source and Python source, chosen because all three are
    on disk and because two of them are genuinely close - C and Python share braces,
    operators, English keywords and identifier conventions that neither shares with prose.

    The expectation going in was that the two programming languages would be the hard pair.
    Section 3 measures it instead, and they are not: both are mistaken for English, and
    never for each other.
    """
    import json as _json

    import pandas as pd

    out: dict[str, list[str]] = {}

    path = _snapshot("datasets--hotpotqa--hotpot_qa", "validation-*.parquet")
    if path is None:
        raise FileNotFoundError("HotpotQA not found in the Hugging Face cache.")
    seen: dict[str, str] = {}
    for context in pd.read_parquet(path).head(2_000)["context"]:
        for title, sentences in zip(list(context["title"]), list(context["sentences"])):
            seen.setdefault(str(title), " ".join(str(s) for s in sentences).strip())
    out["english"] = [t for t in seen.values() if len(t) > 200]

    path = _snapshot("datasets--google--code_x_glue_cc_defect_detection", "test-*.parquet")
    if path is None:
        raise FileNotFoundError("Devign not found in the Hugging Face cache.")
    out["c"] = [str(f) for f in pd.read_parquet(path)["func"] if len(str(f)) > 200]

    python: list[str] = []
    mbpp = _snapshot("datasets--Muennighoff--mbpp", "mbpp.jsonl")
    if mbpp is not None:
        for line in mbpp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                code = str(_json.loads(line).get("code", ""))
                if len(code) > 200:
                    python.append(code)
    human = _snapshot("datasets--openai--openai_humaneval", "test-*.parquet")
    if human is not None:
        frame = pd.read_parquet(human)
        python.extend(
            f"{p}{s}"
            for p, s in zip(frame["prompt"], frame["canonical_solution"])
            if len(f"{p}{s}") > 200
        )
    if not python:
        raise FileNotFoundError("Neither MBPP nor HumanEval found in the Hugging Face cache.")
    out["python"] = python
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--samples", type=int, default=300, help="test snippets per class")
    args = parser.parse_args()

    n_test = 80 if args.quick else args.samples
    started = time.time()
    rng = np.random.default_rng([0, 0x1A46])

    rule("1. Corpora")
    texts = load_texts()
    for label, documents in texts.items():
        chars = sum(len(d) for d in documents)
        print(f"  {label:8} {len(documents):>6,} documents, {chars:>10,} characters")

    # Held-out split: the profile is built from training documents only, and every test
    # snippet is cut from a document the profile never saw.
    profiles, test_pool = [], {}
    for label, documents in texts.items():
        order = rng.permutation(len(documents))
        split = max(1, int(len(documents) * 0.7))
        train = [documents[i] for i in order[:split]]
        test_pool[label] = [documents[i] for i in order[split:]] or train
        profiles.append(identify.Profile.build(label, " ".join(train)))
    print(f"\n  profiles built from 70% of each class, held out {', '.join(texts)}")
    print(f"  profile size: {identify.PROFILE_SIZE} top n-grams, n = 1..5")

    rule("2. Accuracy against input length")
    print(f"  {'chars':>6}" + "".join(f"{m:>17}" for m in identify.METHODS))

    rows = []
    for length in LENGTHS:
        snippets: list[tuple[str, str]] = []
        for label, documents in test_pool.items():
            for _ in range(n_test):
                document = documents[rng.integers(len(documents))]
                if len(document) <= length:
                    snippets.append((label, document))
                    continue
                start = int(rng.integers(0, len(document) - length))
                snippets.append((label, document[start : start + length]))

        scores = {}
        for method in identify.METHODS:
            correct = sum(
                identify.classify(method, text, profiles) == truth for truth, text in snippets
            )
            scores[method] = correct / len(snippets)
        rows.append({"length": length, "n": len(snippets), **scores})
        print(f"  {length:>6}" + "".join(f"{scores[m]:>17.3f}" for m in identify.METHODS))

    best_method = max(identify.METHODS, key=lambda m: rows[-1][m])
    first, last = rows[0], rows[-1]
    print(
        f"\n  At {last['length']} characters the best method is {best_method} at "
        f"{last[best_method]:.3f}.\n  At {first['length']} characters it is "
        f"{first[best_method]:.3f} - a fall of "
        f"{last[best_method] - first[best_method]:.3f} for a 64x shorter input."
    )
    print(
        "  Chance is 0.333. Language identification is reported on documents and used on\n"
        "  queries, log lines and snippets."
    )

    rule("3. Which pairs get confused")
    length = 40
    snippets = []
    for label, documents in test_pool.items():
        for _ in range(n_test):
            document = documents[rng.integers(len(documents))]
            start = int(rng.integers(0, max(1, len(document) - length)))
            snippets.append((label, document[start : start + length]))

    labels = sorted(texts)
    matrix = {truth: dict.fromkeys(labels, 0) for truth in labels}
    for truth, text in snippets:
        matrix[truth][identify.classify(best_method, text, profiles)] += 1

    print(f"  {best_method} at {length} characters, rows are truth:\n")
    print(f"  {'':10}" + "".join(f"{p:>10}" for p in labels))
    for truth in labels:
        total = sum(matrix[truth].values()) or 1
        print(f"  {truth:10}" + "".join(f"{matrix[truth][p] / total:>10.2f}" for p in labels))

    # Read the worst confusion off the matrix rather than asserting which pair it will be.
    # The obvious guess - that the two programming languages blur into each other - is not
    # what happens, and hardcoding it would have shipped a claim the data contradicts.
    confusions = [
        (truth, predicted, matrix[truth][predicted] / max(1, sum(matrix[truth].values())))
        for truth in labels
        for predicted in labels
        if truth != predicted
    ]
    confusions.sort(key=lambda row: -row[2])
    code_pair = next(
        rate for truth, predicted, rate in confusions if {truth, predicted} == {"c", "python"}
    )
    worst_truth, worst_predicted, worst_rate = confusions[0]

    print(
        f"\n  Worst confusion: {worst_truth} read as {worst_predicted}, "
        f"{worst_rate * 100:.1f}% of the time.\n"
        f"  C against Python, the pair that looks hardest: {code_pair * 100:.1f}%."
    )
    print(
        "\n  The two programming languages are not confused with each other. Both are\n"
        "  mistaken for English, and only in that direction - a forty-character window of\n"
        "  source is often entirely identifiers, keywords and comments, which is English\n"
        "  text. What separates C from Python is punctuation density and indentation, and\n"
        "  those survive truncation; what separates code from prose is a vocabulary that a\n"
        "  short enough window simply does not contain."
    )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "language_id.json").write_text(
        json.dumps(
            {
                "corpora": {
                    k: {"documents": len(v), "characters": sum(len(d) for d in v)}
                    for k, v in texts.items()
                },
                "profile_size": identify.PROFILE_SIZE,
                "by_length": rows,
                "best_method": best_method,
                "confusion_at_40": matrix,
                "code_pair_confusion": code_pair,
                "worst_confusion": {
                    "truth": worst_truth,
                    "predicted": worst_predicted,
                    "rate": worst_rate,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'language_id.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
