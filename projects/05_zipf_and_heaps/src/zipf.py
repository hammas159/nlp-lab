"""Five estimators of "the Zipf exponent", and the fact that they do not all estimate it.

Two different exponents travel under the same name:

    rank-frequency     f(r)  proportional to  r ** -a       (a is what Zipf plotted)
    frequency distribution  P(f)  proportional to  f ** -g   (g is what Clauset fits)

They are related by ``g = 1 + 1/a``. A least-squares slope on a log-log rank-frequency
plot estimates ``a``. A maximum-likelihood power-law fit to the *counts* estimates ``g``.
Quoting one against the other - which happens whenever a paper reports "the Zipf exponent
is 1.0" beside another reporting 2.0 - compares two different parameters.

Everything here returns ``a``, converting where needed, so the five numbers in the output
table are commensurable.

References
----------
Clauset, Shalizi & Newman (2009), *Power-law distributions in empirical data*, SIAM Review
51(4) - the MLE, the KS-minimising choice of ``x_min``, and the synthetic goodness-of-fit
test implemented below.

Piantadosi (2014), *Zipf's word frequency law in natural language: a critical review*,
Psychon. Bull. Rev. 21 - rank and frequency measured on the same tokens have correlated
errors, which manufactures regularity at the tail; estimate them on independent halves.

Pilgrim & Hills (2021), *Bias in Zipf's law estimators*, Sci. Rep. 11 - the MLE is itself
positively biased for exponents below about 1.5, which is exactly where natural language
sits, and the split-half fix leaves a residual bias because unobserved tail types cannot
be ranked at all.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from hurwitz import zeta

RNG_DEFAULT = 0

#: Second entropy word for the split-half mask, so its uniform stream cannot coincide with
#: a caller's data-generating stream at the same seed. See `split_half`.
SPLIT_STREAM = 0x5150_1177


@dataclass(frozen=True)
class Fit:
    """One estimate, always expressed as the rank-frequency exponent ``a``."""

    estimator: str
    a: float
    #: The distribution exponent, where the estimator produces one natively.
    gamma: float | None = None
    #: Lower cutoff, in whatever units the estimator works in (rank, or count).
    cutoff: float | None = None
    #: Number of observations the estimate actually used.
    n_used: int = 0
    #: Number of types the estimator had to discard, and why.
    n_discarded: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def a_from_gamma(gamma: float) -> float:
    """``g = 1 + 1/a``, so ``a = 1/(g - 1)``. Undefined at g = 1."""
    return float("inf") if gamma <= 1.0 else 1.0 / (gamma - 1.0)


def gamma_from_a(a: float) -> float:
    return 1.0 + 1.0 / a


def counts(tokens: list[str]) -> np.ndarray:
    """Type frequencies, descending. The input to every estimator here."""
    return np.array(sorted(Counter(tokens).values(), reverse=True), dtype=np.int64)


# --------------------------------------------------------------------------- OLS family


MIN_POINTS = 10


def _ols_slope(log_x: np.ndarray, log_y: np.ndarray) -> float:
    """Least-squares slope, refusing to fit a line to almost nothing.

    ``lstsq`` on two points returns an exact fit and on one point returns a minimum-norm
    solution, in both cases a float that looks like an answer. That is how a broken
    split-half mask produced a confident -2.44 from a single surviving point instead of an
    error, so too few points is an exception here rather than a number.
    """
    if len(log_x) < MIN_POINTS:
        raise ValueError(f"refusing to fit a slope to {len(log_x)} points (need {MIN_POINTS})")
    design = np.vstack([log_x, np.ones_like(log_x)]).T
    slope, _ = np.linalg.lstsq(design, log_y, rcond=None)[0]
    return float(slope)


def ols_rank_frequency(freqs: np.ndarray, r_min: int = 1) -> Fit:
    """Least squares of log frequency on log rank. The plot in every textbook.

    Ranks are sequential, so the thousands of types sharing a count of 1 are spread over
    thousands of consecutive ranks. That flat shelf at the tail is a ranking artefact, and
    it is a large fraction of the points the regression sees.
    """
    freqs = np.sort(freqs)[::-1]
    ranks = np.arange(1, len(freqs) + 1)
    keep = ranks >= r_min
    slope = _ols_slope(np.log(ranks[keep]), np.log(freqs[keep]))
    return Fit(
        estimator="OLS rank-frequency" if r_min == 1 else f"OLS rank-frequency (r>={r_min})",
        a=-slope,
        cutoff=float(r_min),
        n_used=int(keep.sum()),
        n_discarded=int((~keep).sum()),
        note="every type, including the singleton shelf" if r_min == 1 else "head only",
    )


def ols_log_binned(freqs: np.ndarray, bins_per_decade: int = 10) -> Fit:
    """The same regression after logarithmic binning over rank.

    Binning is offered as a remedy for the noisy tail. It changes the answer because it
    silently reweights: one bin at the head holds a single rank, one bin at the tail holds
    tens of thousands, and least squares then treats those two bins as equally informative.
    """
    freqs = np.sort(freqs)[::-1]
    ranks = np.arange(1, len(freqs) + 1)
    n_bins = max(2, int(bins_per_decade * np.log10(len(freqs))))
    edges = np.unique(np.geomspace(1, len(freqs) + 1, n_bins + 1))
    idx = np.digitize(ranks, edges) - 1

    xs, ys = [], []
    for b in range(len(edges)):
        sel = idx == b
        if sel.sum() == 0:
            continue
        xs.append(np.log(ranks[sel]).mean())
        ys.append(np.log(freqs[sel]).mean())

    slope = _ols_slope(np.asarray(xs), np.asarray(ys))
    return Fit(
        estimator="OLS, log-binned",
        a=-slope,
        n_used=len(xs),
        n_discarded=int(len(freqs) - len(xs)),
        note=f"{len(xs)} bins stand in for {len(freqs):,} types",
    )


def split_half(tokens: list[str], seed: int = RNG_DEFAULT) -> Fit:
    """Piantadosi's fix: rank from one half of the tokens, frequency from the other.

    Measuring both coordinates on the same tokens correlates their errors - a type that
    happened to be over-counted is pushed both up the frequency axis and left along the
    rank axis, which tightens the line without any of that tightness being real.

    Assignment is per token occurrence, not per document, so the two halves are drawn from
    the same distribution rather than from different topics.

    The residual bias Pilgrim & Hills name is visible here as ``n_discarded``: a type seen
    in the ranking half but never in the frequency half cannot be placed, and dropping it
    removes exactly the rarest types - the ones the tail exponent is made of.

    The split generator is seeded from ``[seed, SPLIT_STREAM]`` rather than from ``seed``.
    That is not decoration. ``Generator.choice(p=...)`` draws one uniform per sample and
    maps it through the cumulative distribution, so a split mask taken as ``u < 0.5`` from
    an identically seeded generator selects *exactly the tokens whose cumulative
    probability is below one half* - the head of the distribution and nothing else. With
    synthetic text generated at the same seed this put 12 types in the ranking half
    instead of 11,000, and the estimator returned a negative exponent fitted to a single
    surviving point. Independent entropy makes the collision impossible rather than
    unlikely.
    """
    rng = np.random.default_rng([seed, SPLIT_STREAM])
    mask = rng.random(len(tokens)) < 0.5
    arr = np.asarray(tokens, dtype=object)

    rank_counts = Counter(arr[mask].tolist())
    freq_counts = Counter(arr[~mask].tolist())

    ordered = [w for w, _ in rank_counts.most_common()]
    paired = [(r, freq_counts[w]) for r, w in enumerate(ordered, start=1) if freq_counts[w] > 0]
    discarded = len(ordered) - len(paired)

    ranks = np.array([r for r, _ in paired], dtype=np.float64)
    freqs = np.array([f for _, f in paired], dtype=np.float64)
    slope = _ols_slope(np.log(ranks), np.log(freqs))
    return Fit(
        estimator="split-half (Piantadosi)",
        a=-slope,
        n_used=len(paired),
        n_discarded=discarded,
        note="rank from half A, frequency from half B",
    )


# ------------------------------------------------------------------- Clauset MLE family


def _discrete_cdf(x: np.ndarray, gamma: float, x_min: int) -> np.ndarray:
    """P(X <= x) for the discrete power law, via the Hurwitz zeta.

    The survival ratio ``zeta(g, x+1) / zeta(g, x_min)`` is formed from the *scaled* zeta
    so that the two ``q ** -g`` factors cancel analytically. Taken literally the ratio is
    0/0 whenever ``g`` is large enough for both to underflow, which the KS search reaches
    routinely on data that is not power-law.
    """
    upper = x + 1.0
    survival = (upper / x_min) ** -gamma * (
        zeta(gamma, upper, scaled=True) / zeta(gamma, float(x_min), scaled=True)
    )
    return 1.0 - survival


def mle_gamma(data: np.ndarray, x_min: int) -> float:
    """Clauset eq. 3.7, the approximate discrete MLE.

        g = 1 + n * [ sum_i ln( x_i / (x_min - 0.5) ) ] ** -1

    The half-integer offset is what makes the continuous estimator usable on counts.
    """
    return _mle_on_tail(np.sort(data)[np.searchsorted(np.sort(data), x_min) :], x_min)


def _mle_on_tail(tail: np.ndarray, x_min: int) -> float:
    if len(tail) == 0:
        return float("nan")
    return 1.0 + len(tail) / np.sum(np.log(tail.astype(np.float64) / (x_min - 0.5)))


def ks_distance(data: np.ndarray, gamma: float, x_min: int) -> float:
    ordered = np.sort(data)
    return _ks_on_tail(ordered[np.searchsorted(ordered, x_min) :], gamma, x_min)


def _ks_on_tail(tail: np.ndarray, gamma: float, x_min: int) -> float:
    """KS distance on an already-sorted tail, computed over *distinct* values.

    Counts are massively tied - most of a word-frequency tail is the value ``x_min``
    repeated - and the textbook ``i/n`` form of the statistic assumes no ties. Applied to
    tied data it compares the theoretical CDF at the tie against ``(i-1)/n`` for the first
    member of the block, so it reports the height of the atom at ``x_min`` no matter what
    the data are. That is a constant of the model, which made the statistic return the
    same value for a power-law sample and a geometric one.

    The empirical CDF is therefore evaluated once per distinct value, from both sides.
    """
    n = len(tail)
    if n < 2:
        return float("inf")

    values, counts_per_value = np.unique(tail, return_counts=True)
    upper_ecdf = np.cumsum(counts_per_value) / n
    lower_ecdf = upper_ecdf - counts_per_value / n

    floats = values.astype(np.float64)
    at_value = _discrete_cdf(floats, gamma, x_min)
    # The support is the integers, so the largest support point below v is v-1. Comparing
    # the left limit of the ECDF against F(v) instead of F(v-1) measures the height of the
    # atom rather than the misfit, and returns a constant of the model for any data.
    below_value = np.where(values > x_min, _discrete_cdf(floats - 1.0, gamma, x_min), 0.0)

    return float(max(np.abs(upper_ecdf - at_value).max(), np.abs(lower_ecdf - below_value).max()))


def fit_xmin(
    data: np.ndarray, min_tail: int = 50, max_candidates: int = 400
) -> tuple[int, float, float]:
    """Choose x_min by minimising the KS distance, as Clauset prescribes.

    Returns ``(x_min, gamma, ks)``. The data are sorted once and each candidate cutoff is
    a ``searchsorted`` offset into that array, so the scan is linear rather than a fresh
    boolean mask per candidate - which matters because the bootstrap in `goodness_of_fit`
    runs this whole search again for every synthetic dataset.

    ``min_tail`` stops the search from choosing a cutoff so high that the fit is excellent
    because almost nothing is left to fit. ``max_candidates`` caps the scan on corpora
    with thousands of distinct counts; candidates are then spread logarithmically, which
    is where the KS curve actually varies.
    """
    ordered = np.sort(data)
    candidates = np.unique(ordered)
    candidates = candidates[candidates >= 1]
    if len(candidates) > max_candidates:
        picks = np.unique(np.geomspace(1, len(candidates), max_candidates).astype(np.int64) - 1)
        candidates = candidates[picks]

    best = (1, float("nan"), float("inf"))
    for value in candidates:
        x_min = int(value)
        tail = ordered[np.searchsorted(ordered, x_min) :]
        if len(tail) < min_tail:
            break
        gamma = _mle_on_tail(tail, x_min)
        if not np.isfinite(gamma) or gamma <= 1.0:
            continue
        ks = _ks_on_tail(tail, gamma, x_min)
        if ks < best[2]:
            best = (x_min, float(gamma), ks)
    return best


def mle_clauset(freqs: np.ndarray) -> Fit:
    """Maximum likelihood on the counts, with x_min chosen by KS, converted to ``a``."""
    x_min, gamma, ks = fit_xmin(freqs)
    n_tail = int((freqs >= x_min).sum())
    sigma = (gamma - 1.0) / np.sqrt(n_tail) if n_tail else float("nan")
    return Fit(
        estimator="MLE (Clauset)",
        a=a_from_gamma(gamma),
        gamma=gamma,
        cutoff=float(x_min),
        n_used=n_tail,
        n_discarded=int(len(freqs) - n_tail),
        note=f"x_min={x_min} by KS, KS={ks:.4f}, se(g)={sigma:.4f}",
    )


# ------------------------------------------------------------------ goodness of fit


def sample_discrete_powerlaw(
    gamma: float, x_min: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    """Clauset appendix D: the continuous inverse transform, rounded.

    Documented as accurate for x_min of roughly 6 and above. ``test_sampler_recovers_gamma``
    measures the error rather than trusting the documentation.
    """
    u = rng.random(size)
    x = (x_min - 0.5) * (1.0 - u) ** (-1.0 / (gamma - 1.0)) + 0.5
    return np.floor(x).astype(np.int64)


def goodness_of_fit(freqs: np.ndarray, n_synthetic: int = 100, seed: int = RNG_DEFAULT) -> dict:
    """Clauset's semi-parametric bootstrap. Is the power law even the right shape?

    Each synthetic dataset is built the way the fitted model says the data arose: the tail
    from the fitted power law, the body resampled from the observed values below x_min.
    Each is then refitted **from scratch**, x_min included, so the synthetic KS distances
    carry the same selection advantage the real fit had.

    ``p`` is the fraction of synthetic datasets fitting their own model *worse* than the
    data fits its own. Clauset's rule is to reject the power law when p < 0.1. A large p
    is not evidence for the power law - it means the data cannot rule it out.
    """
    rng = np.random.default_rng(seed)
    x_min, gamma, ks_observed = fit_xmin(freqs)
    body = freqs[freqs < x_min]
    n = len(freqs)
    p_tail = float((freqs >= x_min).sum()) / n

    worse = 0
    for _ in range(n_synthetic):
        in_tail = rng.random(n) < p_tail
        n_tail = int(in_tail.sum())
        synth = np.empty(n, dtype=np.int64)
        synth[in_tail] = sample_discrete_powerlaw(gamma, x_min, n_tail, rng)
        if len(body):
            synth[~in_tail] = rng.choice(body, size=n - n_tail, replace=True)
        else:
            synth[~in_tail] = x_min
        # Refitted from scratch, x_min included, so the synthetic KS distances carry the
        # same selection advantage the observed fit had. Only the distance is compared.
        ks_synth = fit_xmin(synth)[2]
        if np.isfinite(ks_synth) and ks_synth >= ks_observed:
            worse += 1

    p = worse / n_synthetic
    return {
        "x_min": int(x_min),
        "gamma": float(gamma),
        "ks": float(ks_observed),
        "p_value": float(p),
        "n_synthetic": n_synthetic,
        "power_law_plausible": bool(p >= 0.1),
    }


# --------------------------------------------------------------------------- the panel


def all_estimates(tokens: list[str], head_rank: int = 1000, seed: int = RNG_DEFAULT) -> list[Fit]:
    """Every estimator, on one token stream, all reported as ``a``."""
    freqs = counts(tokens)
    return [
        ols_rank_frequency(freqs),
        _ols_head(freqs, head_rank),
        ols_log_binned(freqs),
        mle_clauset(freqs),
        split_half(tokens, seed=seed),
    ]


def _ols_head(freqs: np.ndarray, head_rank: int) -> Fit:
    """OLS over the first `head_rank` ranks only - the other common presentation.

    Truncating the tail is usually described as removing noise. It removes most of the
    vocabulary: on a million-token corpus the first thousand ranks are under 2% of types.
    """
    freqs = np.sort(freqs)[::-1][:head_rank]
    ranks = np.arange(1, len(freqs) + 1)
    slope = _ols_slope(np.log(ranks), np.log(freqs))
    return Fit(
        estimator=f"OLS, top {head_rank} ranks",
        a=-slope,
        cutoff=float(head_rank),
        n_used=len(freqs),
        note="tail discarded as noise",
    )
