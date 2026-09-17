"""Produce every table in the README.

python src/run.py           # the full study, ~10 minutes
python src/run.py --quick   # smaller N and fewer bootstrap samples, ~1 minute
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpora
import heaps
import validate
import zipf

RESULTS = Path(__file__).resolve().parent.parent / "results"

#: Every cross-register number is measured at this many tokens, because the Heaps
#: exponent depends on corpus size and the three corpora differ by two orders of magnitude.
#: Set by the smaller of the two large corpora: Devign holds 1,007,287 word tokens, so a
#: million is the most that can be compared without padding one side.
MATCHED_N = 1_000_000

VOCAB_SIZES = (2_000, 8_000, 16_000, 32_000, 50_000, 100_000)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 72)}")


# ------------------------------------------------------------------ 1. the corpora


def section_corpora(quick: bool) -> tuple[dict, dict]:
    rule("1. Corpora, as loaded")
    loaded = corpora.load_all()
    matched_n = 200_000 if quick else MATCHED_N

    summary = {}
    print(f"{'key':8} {'corpus':24} {'register':14} {'tokens':>12} {'types':>10}  {'TTR':>7}")
    for key, corpus in loaded.items():
        summary[key] = {
            "name": corpus.name,
            "register": corpus.register,
            "tokens": len(corpus),
            "types": corpus.types,
            "type_token_ratio": corpus.types / len(corpus),
        }
        print(
            f"{key:8} {corpus.name:24} {corpus.register:14} {len(corpus):>12,} "
            f"{corpus.types:>10,}  {corpus.types / len(corpus):>7.4f}"
        )

    usable = {k: c for k, c in loaded.items() if len(c) >= matched_n}
    print(
        f"\n  matched-N comparison uses {matched_n:,} tokens: "
        f"{', '.join(usable) or 'none'} qualify."
    )
    for key, corpus in loaded.items():
        if key not in usable:
            print(f"  {key} has {len(corpus):,} tokens and is held back for section 5.")
    return loaded, {"corpora": summary, "matched_n": matched_n}


# --------------------------------------------- 2. which estimator should be believed


def section_validation(quick: bool) -> dict:
    rule("2. Estimator bias, measured on text whose exponent is known")
    started = time.time()
    rows = validate.bias_table(
        n_tokens=40_000 if quick else 200_000,
        support=20_000 if quick else 100_000,
        repeats=3 if quick else 8,
    )
    summary = validate.summarise(rows)

    true_values = sorted({r["true_a"] for r in rows})
    estimators = [r["estimator"] for r in rows if r["true_a"] == true_values[0]]

    header = f"{'estimator':26}" + "".join(f"{f'a = {t}':>16}" for t in true_values)
    print(header)
    print("-" * len(header))
    for estimator in estimators:
        line = f"{estimator:26}"
        for t in true_values:
            row = next(r for r in rows if r["estimator"] == estimator and r["true_a"] == t)
            line += f"{row['mean_a']:>9.3f} {row['bias']:>+6.3f}"
        print(line)

    print(f"\n  {'estimator':26} mean |bias|   max sd across settings")
    for row in summary:
        spread = max(r["std"] for r in rows if r["estimator"] == row["estimator"])
        print(f"  {row['estimator']:26} {row['mean_abs_bias']:>10.3f}   {spread:>10.3f}")
    best = summary[0]
    print(
        f"\n  Least biased on i.i.d. Zipfian text: {best['estimator']} "
        f"({best['mean_abs_bias']:.3f}). Real text is not i.i.d., so this is a floor."
    )
    print("  Sections 5 and 7 take their exponent from this estimator, not from a default.")
    print(f"  [{time.time() - started:.1f}s]")
    return {"rows": rows, "summary": summary, "best_estimator": best["estimator"]}


# ---------------------------------------------------- 3. the exponent, five ways


def section_zipf(loaded: dict, matched_n: int, quick: bool) -> dict:
    rule(f"3. The Zipf exponent at a matched {matched_n:,} tokens")
    out = {}
    for key, corpus in loaded.items():
        if len(corpus) < matched_n:
            continue
        clipped = corpus.truncate(matched_n)
        fits = zipf.all_estimates(clipped.tokens)
        out[key] = [f.as_dict() for f in fits]

        print(f"\n  {corpus.name} ({corpus.register}), {clipped.types:,} types")
        print(f"  {'estimator':26} {'a':>7} {'used':>9} {'dropped':>9}  note")
        for f in fits:
            print(f"  {f.estimator:26} {f.a:>7.3f} {f.n_used:>9,} {f.n_discarded:>9,}  {f.note}")
        spread = max(f.a for f in fits) - min(f.a for f in fits)
        print(f"  {'spread across estimators':26} {spread:>7.3f}")

    return out


# ------------------------------------------------------- 4. is it a power law at all


def section_gof(loaded: dict, matched_n: int, quick: bool) -> dict:
    rule("4. Clauset goodness-of-fit: is the power law even the right shape?")
    n_synthetic = 20 if quick else 100
    out = {}
    for key, corpus in loaded.items():
        if len(corpus) < matched_n:
            continue
        started = time.time()
        freqs = zipf.counts(corpus.truncate(matched_n).tokens)
        result = zipf.goodness_of_fit(freqs, n_synthetic=n_synthetic)
        result["seconds"] = round(time.time() - started, 1)
        out[key] = result
        verdict = "cannot be ruled out" if result["power_law_plausible"] else "REJECTED"
        print(
            f"  {corpus.name:24} x_min={result['x_min']:>5}  g={result['gamma']:.3f}  "
            f"KS={result['ks']:.4f}  p={result['p_value']:.2f}  -> power law {verdict}"
            f"   [{result['seconds']}s]"
        )
    print(f"\n  Clauset's rule: reject when p < 0.1. {n_synthetic} synthetic datasets each.")
    return out


# ------------------------------------------------------- 5. Heaps, and finite size


def section_heaps(
    loaded: dict, zipf_fits: dict, matched_n: int, best_estimator: str, quick: bool
) -> dict:
    rule("5. Heaps' law: the exponent is a function of how much you read")
    print(f"  Zipf exponent taken from '{best_estimator}', the least biased in section 2.\n")
    out = {}

    print(
        f"  {'corpus':24} {'N':>10} {'b measured':>11} {'R2':>7} {'b asympt':>10} {'b finite-N':>11}"
    )
    for key, corpus in loaded.items():
        n = min(len(corpus), matched_n)
        clipped = corpus.truncate(n)
        sizes, types = heaps.vocabulary_curve(clipped.tokens)
        measured = heaps.fit_beta(sizes, types)

        fits = zipf_fits.get(key)
        if fits:
            a_hat = next(f["a"] for f in fits if f["estimator"] == best_estimator)
        else:
            a_hat = next(
                f.a for f in zipf.all_estimates(clipped.tokens) if f.estimator == best_estimator
            )
        asympt = heaps.beta_asymptotic(a_hat)
        simulated = heaps.beta_simulated(
            a_hat,
            n_tokens=min(n, 200_000 if quick else 500_000),
            support=max(2 * clipped.types, 1000),
        )

        out[key] = {
            "n_tokens": n,
            "measured": measured,
            "a_used": a_hat,
            "beta_asymptotic": asympt,
            "beta_simulated": simulated,
            "drift": heaps.beta_drift(sizes, types),
            "curve": {"sizes": sizes.tolist(), "types": types.tolist()},
        }
        print(
            f"  {corpus.name:24} {n:>10,} {measured['beta']:>11.3f} "
            f"{measured['r_squared']:>7.4f} {asympt:>10.3f} {simulated['beta']:>11.3f}"
        )

    print("\n  b within one corpus, fitted in sliding windows:")
    for key, entry in out.items():
        drift = entry["drift"]
        span = f"{drift[0]['beta']:.3f} -> {drift[-1]['beta']:.3f}"
        print(
            f"  {loaded[key].name:24} {drift[0]['from_tokens']:>9,} to "
            f"{drift[-1]['to_tokens']:>10,} tokens   b {span}"
        )
    print("\n  A Heaps exponent quoted without a token count names no quantity.")
    return out


# ---------------------------------------------------------- 6. the bill for a cutoff


def section_oov(loaded: dict, matched_n: int) -> dict:
    rule("6. What a fixed vocabulary costs, per register")
    out = {}
    header = f"  {'corpus':24}" + "".join(f"{v // 1000:>8}k" for v in VOCAB_SIZES) + f"{'floor':>9}"
    print(header)
    print(f"  {'(OOV rate over tokens; a * marks a vocabulary larger than the training half)':24}")
    for key, corpus in loaded.items():
        clipped = corpus.truncate(min(len(corpus), matched_n))
        rows = heaps.oov_cost(clipped.tokens, VOCAB_SIZES)
        floor = heaps.irreducible_oov(clipped.tokens)
        sizes, types = heaps.vocabulary_curve(clipped.tokens)
        fit = heaps.fit_beta(sizes, types)
        out[key] = {
            "n_tokens": len(clipped),
            "rows": rows,
            "irreducible": floor,
            "beta_used": fit["beta"],
            "tokens_for_100k_types": heaps.tokens_for_vocabulary(fit, 100_000),
        }
        line = f"  {corpus.name:24}"
        for row in rows:
            mark = "*" if row["capped_by_training_corpus"] else " "
            line += f"{row['oov_token_rate'] * 100:>7.2f}%{mark}"
        line += f"{floor['oov_token_rate'] * 100:>8.2f}%"
        print(line)

    print()
    for key, entry in out.items():
        print(
            f"  {loaded[key].name:24} training half holds "
            f"{entry['irreducible']['train_types']:>7,} types; "
            f"{entry['irreducible']['oov_token_rate'] * 100:.2f}% of held-out tokens are of "
            f"types it never contained at any vocabulary size."
        )

    print()
    for key, entry in out.items():
        print(
            f"  {loaded[key].name:24} extrapolating b={entry['beta_used']:.3f}: "
            f"{entry['tokens_for_100k_types']:,.0f} tokens to reach 100,000 types"
        )
    print("\n  Extrapolation assumes a constant b, which section 5 shows is false.")
    return out


# ------------------------------------------------------- 7. did the tokenizer decide it


def section_tokenizer(matched_n: int, best_estimator: str) -> dict:
    rule("7. Sensitivity: how much of this was the tokenizer?")
    out = {}
    print(f"  {'corpus':24} {'tokenizer':12} {'tokens':>11} {'types':>9} {'a':>10}")
    for key in corpora.SOURCES:
        for label, tok in (("word", corpora.tokenize), ("code-aware", corpora.code_tokenize)):
            corpus = corpora.load(key, tok)
            clipped = corpus.truncate(min(len(corpus), matched_n))
            a = next(
                f.a for f in zipf.all_estimates(clipped.tokens) if f.estimator == best_estimator
            )
            out.setdefault(key, {})[label] = {
                "tokens": len(clipped),
                "types": clipped.types,
                "a": a,
            }
            print(
                f"  {corpus.name:24} {label:12} {len(clipped):>11,} {clipped.types:>9,} {a:>10.3f}"
            )
    return out


# ---------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smaller N, fewer bootstrap samples")
    parser.add_argument("--skip-tokenizer", action="store_true")
    args = parser.parse_args()

    started = time.time()
    loaded, meta = section_corpora(args.quick)
    matched_n = meta["matched_n"]

    results = {"meta": meta}
    results["validation"] = section_validation(args.quick)
    best = results["validation"]["best_estimator"]
    results["zipf"] = section_zipf(loaded, matched_n, args.quick)
    results["goodness_of_fit"] = section_gof(loaded, matched_n, args.quick)
    results["heaps"] = section_heaps(loaded, results["zipf"], matched_n, best, args.quick)
    results["oov"] = section_oov(loaded, matched_n)
    if not args.skip_tokenizer:
        results["tokenizer_sensitivity"] = section_tokenizer(matched_n, best)

    RESULTS.mkdir(exist_ok=True)
    curves = {
        key: results["heaps"][key].pop("curve")
        for key in list(results["heaps"])
        if "curve" in results["heaps"][key]
    }
    (RESULTS / "study.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (RESULTS / "vocabulary_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS / 'study.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
