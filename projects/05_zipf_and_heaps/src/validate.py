"""Which estimator is right? Ask text whose exponent is known because it was chosen.

Every number in the corpus tables is an estimate from an estimator that could be wrong,
and the corpora cannot say which. Synthetic text can: draw tokens from a Zipf
distribution with a chosen ``a``, hand the token stream to all five estimators, and the
error is then observable rather than argued about.

This is the only part of the project where the truth is known, so it is what decides how
the corpus table is read.
"""

from __future__ import annotations

import numpy as np

from zipf import all_estimates


def zipf_stream(a: float, n_tokens: int, support: int, rng: np.random.Generator) -> list[str]:
    """A token stream whose rank-frequency exponent is ``a`` by construction.

    Tokens are drawn i.i.d., which real text is not - there is no burstiness here, no
    topic, no document structure. That makes this a *lower bound* on estimator error:
    whatever bias shows up on data this clean does not go away on data that is not.
    """
    ranks = np.arange(1, support + 1, dtype=np.float64)
    weights = ranks**-a
    weights /= weights.sum()
    draws = rng.choice(support, size=n_tokens, p=weights)
    return [f"w{i}" for i in draws.tolist()]


def bias_table(
    true_as: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5),
    n_tokens: int = 200_000,
    support: int = 100_000,
    repeats: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Mean signed error of each estimator, per true exponent.

    ``repeats`` independent streams per setting, so the reported error is a mean over
    samples rather than one draw that happened to land well.
    """
    rows: list[dict] = []
    for a_true in true_as:
        collected: dict[str, list[float]] = {}
        for r in range(repeats):
            rng = np.random.default_rng(seed + r)
            tokens = zipf_stream(a_true, n_tokens, support, rng)
            for fit in all_estimates(tokens, seed=seed + r):
                collected.setdefault(fit.estimator, []).append(fit.a)
        for estimator, values in collected.items():
            arr = np.array(values, dtype=np.float64)
            rows.append(
                {
                    "true_a": a_true,
                    "estimator": estimator,
                    "mean_a": float(arr.mean()),
                    "bias": float(arr.mean() - a_true),
                    "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "n_tokens": n_tokens,
                    "repeats": repeats,
                }
            )
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Mean absolute bias per estimator across the exponents tested."""
    by_estimator: dict[str, list[float]] = {}
    for row in rows:
        by_estimator.setdefault(row["estimator"], []).append(abs(row["bias"]))
    return sorted(
        (
            {"estimator": estimator, "mean_abs_bias": float(np.mean(values))}
            for estimator, values in by_estimator.items()
        ),
        key=lambda r: r["mean_abs_bias"],
    )
