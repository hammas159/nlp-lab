"""Tests for the five estimators, the KS machinery and the goodness-of-fit bootstrap.

No dataset and no network. Where an estimator is supposed to recover a number, the number
is constructed first and the recovery is asserted - an estimator that runs without
erroring is not an estimator that works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from validate import zipf_stream
from zipf import (
    _discrete_cdf,
    a_from_gamma,
    all_estimates,
    counts,
    fit_xmin,
    gamma_from_a,
    goodness_of_fit,
    ks_distance,
    mle_gamma,
    ols_log_binned,
    ols_rank_frequency,
    sample_discrete_powerlaw,
    split_half,
)

# --- the two exponents are not the same number -------------------------------------------


def test_exponent_conversion_round_trips():
    for a in (0.8, 1.0, 1.3, 2.0):
        assert a_from_gamma(gamma_from_a(a)) == pytest.approx(a, rel=1e-12)


def test_language_like_exponents_map_to_very_different_numbers():
    """a = 1.0 is g = 2.0. Quoting one as the other is a factor-of-two error, and it is
    the reason every estimate in this project is converted before it is tabulated."""
    assert gamma_from_a(1.0) == pytest.approx(2.0)
    assert a_from_gamma(2.0) == pytest.approx(1.0)


def test_gamma_of_one_has_no_finite_a():
    assert a_from_gamma(1.0) == float("inf")
    assert a_from_gamma(0.5) == float("inf")


# --- counts -------------------------------------------------------------------------------


def test_counts_are_descending():
    freqs = counts(["a"] * 5 + ["b"] * 3 + ["c"])
    assert freqs.tolist() == [5, 3, 1]


def test_counts_of_empty_stream():
    assert counts([]).tolist() == []


# --- OLS recovers a slope it was given ----------------------------------------------------


def test_ols_recovers_an_exact_power_law():
    """f(r) = r ** -1.2 exactly, with no sampling noise. The estimator has no excuse."""
    ranks = np.arange(1, 5001)
    freqs = np.round(1e6 * ranks**-1.2).astype(np.int64)
    freqs = freqs[freqs > 0]
    assert ols_rank_frequency(freqs).a == pytest.approx(1.2, abs=0.02)


def test_binning_barely_matters_on_a_noiseless_curve():
    """On an exact power law with no singleton shelf, binning is nearly a no-op. Asserting
    this is what makes the next test about sampled data rather than about binning itself."""
    ranks = np.arange(1, 20001)
    freqs = np.round(1e6 * ranks**-1.1).astype(np.int64)
    freqs = freqs[freqs > 0]
    assert abs(ols_rank_frequency(freqs).a - ols_log_binned(freqs).a) < 0.01


def test_binning_changes_the_answer_on_sampled_text():
    """Where it is actually applied - sampled text, with thousands of types tied at a count
    of one - binning is a reweighting, and it returns a materially different exponent from
    the same data."""
    rng = np.random.default_rng(15)
    freqs = counts(zipf_stream(1.0, n_tokens=200_000, support=100_000, rng=rng))
    assert abs(ols_rank_frequency(freqs).a - ols_log_binned(freqs).a) > 0.05


def test_ols_reports_how_many_types_it_used():
    freqs = counts([c for i, c in enumerate("abcdefghijkl") for _ in range(12 - i)])
    fit = ols_rank_frequency(freqs)
    assert fit.n_used == 12
    assert fit.n_discarded == 0


# --- the discrete power-law sampler and the MLE -------------------------------------------


def test_sampler_recovers_gamma():
    """Clauset's appendix-D sampler is an approximation. This measures its error rather
    than trusting the documentation that says it is small for x_min >= 6."""
    rng = np.random.default_rng(0)
    for gamma_true in (2.0, 2.5, 3.0):
        draws = sample_discrete_powerlaw(gamma_true, x_min=6, size=200_000, rng=rng)
        assert mle_gamma(draws, 6) == pytest.approx(gamma_true, abs=0.05)


def test_mle_needs_the_half_integer_offset():
    """Without the -0.5, the continuous estimator applied to counts is biased upward.
    Asserting the offset matters keeps anyone from 'simplifying' it away."""
    rng = np.random.default_rng(1)
    draws = sample_discrete_powerlaw(2.5, x_min=2, size=100_000, rng=rng)
    tail = draws.astype(np.float64)
    without_offset = 1.0 + len(tail) / np.sum(np.log(tail / 2.0))
    with_offset = mle_gamma(draws, 2)
    assert abs(with_offset - 2.5) < abs(without_offset - 2.5)


def test_mle_on_empty_tail_is_nan():
    assert np.isnan(mle_gamma(np.array([1, 1, 2]), x_min=100))


# --- KS distance --------------------------------------------------------------------------


def test_ks_is_small_for_data_from_the_model():
    rng = np.random.default_rng(2)
    draws = sample_discrete_powerlaw(2.5, x_min=5, size=50_000, rng=rng)
    assert ks_distance(draws, 2.5, 5) < 0.02


def test_ks_separates_a_power_law_from_a_geometric():
    """Regression test for a bug that made this impossible.

    The first version compared the left limit of the empirical CDF against F(v) rather
    than F(v-1). Counts are heavily tied, so that measured the height of the atom at
    x_min - a constant of the model - and returned *the same* distance, 0.2581, for a
    power-law sample and a geometric one. The ratio is asserted rather than an absolute
    threshold, because the ratio is what a usable statistic has to deliver.
    """
    rng = np.random.default_rng(3)
    in_model = ks_distance(sample_discrete_powerlaw(2.5, 5, 50_000, rng), 2.5, 5)
    off_model = ks_distance(rng.geometric(p=0.2, size=50_000) + 4, 2.5, 5)
    assert off_model > 10 * in_model


def test_ks_grows_when_the_exponent_is_wrong():
    rng = np.random.default_rng(14)
    draws = sample_discrete_powerlaw(2.5, x_min=5, size=50_000, rng=rng)
    assert ks_distance(draws, 3.5, 5) > ks_distance(draws, 2.5, 5)


def test_cdf_survives_a_large_exponent():
    """The KS search reaches fitted exponents in the hundreds on data that is not a power
    law. Formed naively the survival ratio underflows to 0/0 and the search silently
    compares NaNs, so the scaled-zeta path is asserted here rather than assumed."""
    values = np.arange(190.0, 200.0)
    cdf = _discrete_cdf(values, gamma=200.0, x_min=190)
    assert np.all(np.isfinite(cdf))
    assert np.all(np.diff(cdf) >= 0)
    assert 0.0 <= cdf.min() and cdf.max() <= 1.0


def test_xmin_search_prefers_the_true_cutoff():
    """A power-law tail above 20 glued onto a uniform body below it. The KS search should
    land near 20 rather than fitting the whole thing."""
    rng = np.random.default_rng(4)
    tail = sample_discrete_powerlaw(2.5, x_min=20, size=20_000, rng=rng)
    body = rng.integers(1, 20, size=20_000)
    x_min, gamma, ks = fit_xmin(np.concatenate([tail, body]))
    assert 10 <= x_min <= 40
    assert gamma == pytest.approx(2.5, abs=0.2)
    assert ks < 0.05


def test_xmin_search_respects_the_tail_floor():
    """With fewer than min_tail observations everywhere, the search must not invent a fit
    on three points."""
    x_min, _, _ = fit_xmin(np.array([1, 2, 3, 4, 5] * 4), min_tail=50)
    assert x_min == 1


# --- Piantadosi's split ------------------------------------------------------------------


def test_split_half_discards_the_types_it_cannot_rank():
    """The residual bias Pilgrim & Hills name, made visible: a type seen in the ranking
    half but not in the frequency half is dropped, and those are the rarest types."""
    rng = np.random.default_rng(5)
    tokens = zipf_stream(1.0, n_tokens=50_000, support=20_000, rng=rng)
    fit = split_half(tokens)
    assert fit.n_discarded > 0
    assert fit.n_used > 0


def test_split_half_survives_a_seed_shared_with_the_data_generator():
    """Regression test for the worst bug in this project.

    ``Generator.choice(p=...)`` draws one uniform per sample and maps it through the
    cumulative distribution. A split mask of ``u < 0.5`` taken from a generator seeded
    identically therefore selects exactly those tokens whose cumulative probability is
    under one half - the head of the distribution, and nothing else. The ranking half held
    12 types instead of eleven thousand, one point survived the pairing, and the estimator
    returned a confident -2.44 with no error and no warning.

    The stream and the split are seeded the same way here on purpose.
    """
    rng = np.random.default_rng(0)
    tokens = zipf_stream(1.2, n_tokens=100_000, support=50_000, rng=rng)
    fit = split_half(tokens, seed=0)
    assert fit.n_used > 1_000
    assert 0.3 < fit.a < 2.0


def test_a_slope_is_refused_rather_than_invented():
    """Two points give lstsq an exact fit and one point a minimum-norm solution. Both look
    like answers, which is how the bug above stayed silent."""
    with pytest.raises(ValueError, match="refusing to fit"):
        ols_rank_frequency(np.array([5, 3, 1]))


def test_split_half_is_deterministic_given_a_seed():
    rng = np.random.default_rng(6)
    tokens = zipf_stream(1.1, n_tokens=20_000, support=5_000, rng=rng)
    assert split_half(tokens, seed=7).a == split_half(tokens, seed=7).a


def test_split_half_and_plain_ols_disagree_on_the_same_tokens():
    """If they agreed, the correlated-error problem would not be worth a method."""
    rng = np.random.default_rng(8)
    tokens = zipf_stream(1.0, n_tokens=100_000, support=50_000, rng=rng)
    plain = ols_rank_frequency(counts(tokens)).a
    split = split_half(tokens).a
    assert abs(plain - split) > 0.01


# --- goodness of fit ----------------------------------------------------------------------


def test_gof_does_not_reject_data_drawn_from_a_power_law():
    rng = np.random.default_rng(9)
    draws = sample_discrete_powerlaw(2.5, x_min=6, size=5_000, rng=rng)
    result = goodness_of_fit(draws, n_synthetic=30, seed=9)
    assert result["p_value"] >= 0.1
    assert result["power_law_plausible"] is True


def test_gof_rejects_counts_that_are_obviously_not_a_power_law():
    """Uniform counts have no tail at all. If the test cannot reject these it can reject
    nothing."""
    rng = np.random.default_rng(10)
    result = goodness_of_fit(rng.integers(1, 200, size=20_000), n_synthetic=30, seed=10)
    assert result["power_law_plausible"] is False


def test_gof_has_limited_power_against_a_geometric():
    """An honest limit, asserted rather than left for a reader to discover.

    ``x_min`` is chosen freely, so against a near-miss distribution the search can retreat
    into a short far tail where a power law does fit, and the test then fails to reject.
    That is a property of the procedure, and it means a large p-value here is never
    evidence *for* a power law - only a failure to rule one out.
    """
    rng = np.random.default_rng(11)
    draws = rng.geometric(p=0.05, size=5_000) + 1
    result = goodness_of_fit(draws, n_synthetic=30, seed=11)
    assert result["x_min"] > 1
    assert 0.0 <= result["p_value"] <= 1.0


def test_gof_reports_the_fit_it_tested():
    rng = np.random.default_rng(11)
    draws = sample_discrete_powerlaw(2.3, x_min=6, size=3_000, rng=rng)
    result = goodness_of_fit(draws, n_synthetic=10, seed=11)
    assert set(result) >= {"x_min", "gamma", "ks", "p_value", "power_law_plausible"}
    assert result["gamma"] > 1.0


# --- the panel ----------------------------------------------------------------------------


def test_all_estimates_returns_five_commensurable_numbers():
    rng = np.random.default_rng(12)
    tokens = zipf_stream(1.1, n_tokens=60_000, support=30_000, rng=rng)
    fits = all_estimates(tokens)
    assert len(fits) == 5
    assert len({f.estimator for f in fits}) == 5
    assert all(np.isfinite(f.a) for f in fits)


def test_the_five_estimators_do_not_agree():
    """The finding, asserted as a test: on one token stream with one true exponent, the
    five numbers people all call 'the Zipf exponent' span a visible range."""
    rng = np.random.default_rng(13)
    tokens = zipf_stream(1.1, n_tokens=100_000, support=50_000, rng=rng)
    values = [f.a for f in all_estimates(tokens)]
    assert max(values) - min(values) > 0.1
