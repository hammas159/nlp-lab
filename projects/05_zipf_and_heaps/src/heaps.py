"""Vocabulary growth: how fast it grows, whether the exponent is a constant, and the bill.

Heaps' law says the number of distinct types in a text of ``N`` tokens goes as
``V(N) = K * N ** b`` with ``b`` below 1. Because ``b`` is below 1 and never reaches it,
V has no asymptote: a corpus does not run out of new words, it only finds them more
slowly. Every fixed vocabulary size is therefore a truncation with a price, and this
module measures the price instead of assuming it is small.

The relation to Zipf, asymptotically, is ``b = min(1, 1/a)``. Lu, Zhang & Zhou (2010),
*Zipf's law leads to Heaps' law*, PLoS ONE 5(12), showed that this is an infinite-size
result and that at finite N the exponent depends on N - which is the thing that makes a
Heaps exponent quoted without a corpus size uninterpretable. Both the asymptotic
prediction and a finite-size simulation at the corpus's own N are computed here, and both
are compared against the measured value rather than substituted for it.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from zipf import _ols_slope


def vocabulary_curve(tokens: list[str], n_points: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """``(N, V(N))`` at log-spaced prefix lengths.

    Log spacing, not linear: a linear grid puts almost every point in the last decade,
    where the curve is closest to flat, and then least squares is fitted mostly to that.
    """
    total = len(tokens)
    grid = np.unique(np.geomspace(max(10, total // 10_000), total, n_points).astype(np.int64))

    seen: set[str] = set()
    sizes, types = [], []
    cursor = 0
    for n in grid:
        seen.update(tokens[cursor:n])
        cursor = int(n)
        sizes.append(int(n))
        types.append(len(seen))
    return np.array(sizes), np.array(types)


def fit_beta(sizes: np.ndarray, types: np.ndarray, n_min: int = 0) -> dict:
    """OLS of log V on log N. ``n_min`` drops the short prefixes, where V is nearly N."""
    keep = sizes >= n_min
    log_n, log_v = np.log(sizes[keep]), np.log(types[keep])
    beta = _ols_slope(log_n, log_v)
    intercept = float(log_v.mean() - beta * log_n.mean())

    predicted = intercept + beta * log_n
    ss_res = float(((log_v - predicted) ** 2).sum())
    ss_tot = float(((log_v - log_v.mean()) ** 2).sum())
    return {
        "beta": float(beta),
        "K": float(np.exp(intercept)),
        "r_squared": 1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        "n_points": int(keep.sum()),
        "fitted_from": int(sizes[keep].min()),
        "fitted_to": int(sizes[keep].max()),
    }


def beta_drift(sizes: np.ndarray, types: np.ndarray, window: int = 12) -> list[dict]:
    """Fit b in a sliding window along the curve.

    A single b for a whole corpus is only meaningful if this is flat. It is not, and the
    direction it moves is the finite-size effect Lu et al. describe.
    """
    out = []
    for start in range(0, len(sizes) - window + 1, max(1, window // 3)):
        stop = start + window
        log_n, log_v = np.log(sizes[start:stop]), np.log(types[start:stop])
        out.append(
            {
                "from_tokens": int(sizes[start]),
                "to_tokens": int(sizes[stop - 1]),
                "beta": float(_ols_slope(log_n, log_v)),
            }
        )
    return out


def beta_asymptotic(a: float) -> float:
    """``b = min(1, 1/a)``.

    Equivalently ``min(1, g - 1)`` in the distribution exponent, since ``g = 1 + 1/a``.
    Valid only as N goes to infinity, which no corpus does.
    """
    return float(min(1.0, 1.0 / a))


def beta_simulated(
    a: float, n_tokens: int, support: int, seed: int = 0, n_points: int = 40
) -> dict:
    """Draw ``n_tokens`` from a Zipf distribution with exponent ``a`` and fit b to that.

    This is the finite-size prediction: what Heaps exponent *would* be measured on a
    corpus this size if the ranks really were Zipfian with this exponent. Comparing it to
    the measured b separates "the corpus is not Zipfian" from "the corpus is not infinite".

    ``support`` is the number of possible types, and it is a modelling choice with no
    correct value - the theoretical distribution is unbounded, a real language is not.
    `run.py` sets it from the observed type count and reports the sensitivity.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, support + 1, dtype=np.float64)
    weights = ranks**-a
    weights /= weights.sum()

    draws = rng.choice(support, size=n_tokens, p=weights)
    grid = np.unique(np.geomspace(max(10, n_tokens // 10_000), n_tokens, n_points).astype(np.int64))

    seen: set[int] = set()
    sizes, types = [], []
    cursor = 0
    for n in grid:
        seen.update(draws[cursor:n].tolist())
        cursor = int(n)
        sizes.append(int(n))
        types.append(len(seen))

    fit = fit_beta(np.array(sizes), np.array(types))
    fit["support"] = support
    fit["final_types"] = types[-1]
    return fit


def oov_cost(tokens: list[str], vocab_sizes: tuple[int, ...], split: float = 0.5) -> list[dict]:
    """Build a vocabulary on the first part of the stream, price it on the rest.

    This is how a vocabulary is actually used - fixed from a training corpus, then applied
    to text it has never seen - so measuring OOV on the same text the vocabulary was cut
    from would report a number no deployment ever gets.

    The rate that matters is over *tokens*, not types: a 2% type-level miss can be a
    fraction of a percent of running text, and quoting the type rate overstates the damage
    while quoting neither understates it.
    """
    cut = int(len(tokens) * split)
    train, test = tokens[:cut], tokens[cut:]
    ranked = [w for w, _ in Counter(train).most_common()]
    test_counts = Counter(test)
    total_tokens = sum(test_counts.values())
    total_types = len(test_counts)

    out = []
    for size in vocab_sizes:
        kept = set(ranked[:size])
        oov_tokens = sum(c for w, c in test_counts.items() if w not in kept)
        oov_types = sum(1 for w in test_counts if w not in kept)
        out.append(
            {
                "vocab_size": size,
                "train_types": len(ranked),
                # Above this the vocabulary parameter stops doing anything: there are no
                # more types in the training half to add, and the remaining OOV is text
                # the vocabulary could never have contained at any size.
                "capped_by_training_corpus": size > len(ranked),
                "oov_token_rate": oov_tokens / total_tokens if total_tokens else float("nan"),
                "oov_type_rate": oov_types / total_types if total_types else float("nan"),
            }
        )
    return out


def irreducible_oov(tokens: list[str], split: float = 0.5) -> dict:
    """The OOV rate an unlimited vocabulary still pays.

    Heaps' law says new types keep arriving, so some fraction of any held-out text is made
    of types the training half never contained. No vocabulary size reaches it, which is the
    part of the table that looks like a plateau and is actually a floor.
    """
    cut = int(len(tokens) * split)
    train, test = set(tokens[:cut]), Counter(tokens[cut:])
    total = sum(test.values())
    unseen_tokens = sum(c for w, c in test.items() if w not in train)
    unseen_types = sum(1 for w in test if w not in train)
    return {
        "train_types": len(train),
        "oov_token_rate": unseen_tokens / total if total else float("nan"),
        "oov_type_rate": unseen_types / len(test) if test else float("nan"),
    }


def tokens_for_vocabulary(fit: dict, target_types: int) -> float:
    """Invert ``V = K * N ** b``. How much text to reach a vocabulary of this size?

    Extrapolation, and stated as such: it assumes b holds outside the range it was fitted
    over, which `beta_drift` shows it does not.
    """
    return float((target_types / fit["K"]) ** (1.0 / fit["beta"]))
