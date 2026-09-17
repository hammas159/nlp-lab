"""Tests for vocabulary growth, the exponent's drift, and the cost of a fixed vocabulary.

No dataset and no network. Where a curve is supposed to have an exponent, the curve is
constructed with that exponent first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from heaps import (
    beta_asymptotic,
    beta_drift,
    beta_simulated,
    fit_beta,
    irreducible_oov,
    oov_cost,
    tokens_for_vocabulary,
    vocabulary_curve,
)

# --- the curve ----------------------------------------------------------------------------


def test_vocabulary_curve_is_monotone():
    tokens = [f"w{i % 500}" for i in range(20_000)]
    sizes, types = vocabulary_curve(tokens)
    assert np.all(np.diff(sizes) > 0)
    assert np.all(np.diff(types) >= 0)


def test_vocabulary_curve_ends_at_the_full_type_count():
    tokens = [f"w{i % 500}" for i in range(20_000)]
    sizes, types = vocabulary_curve(tokens)
    assert sizes[-1] == len(tokens)
    assert types[-1] == len(set(tokens))


def test_a_stream_of_all_new_tokens_grows_linearly():
    """Every token unique means V = N, so beta must come out at 1."""
    tokens = [f"w{i}" for i in range(5_000)]
    sizes, types = vocabulary_curve(tokens)
    assert fit_beta(sizes, types)["beta"] == pytest.approx(1.0, abs=1e-9)


def test_a_closed_vocabulary_saturates():
    """Ten types repeated: V stops growing, so beta collapses towards 0."""
    tokens = [f"w{i % 10}" for i in range(50_000)]
    sizes, types = vocabulary_curve(tokens)
    assert fit_beta(sizes, types)["beta"] < 0.15


# --- the fit ------------------------------------------------------------------------------


def test_fit_beta_recovers_a_constructed_exponent():
    sizes = np.geomspace(100, 10_000_000, 80).astype(np.int64)
    types = (7.0 * sizes**0.62).astype(np.int64)
    fit = fit_beta(sizes, types)
    assert fit["beta"] == pytest.approx(0.62, abs=0.005)
    assert fit["K"] == pytest.approx(7.0, rel=0.05)
    assert fit["r_squared"] > 0.999


def test_fit_beta_reports_the_range_it_used():
    sizes = np.geomspace(100, 1_000_000, 50).astype(np.int64)
    types = (3.0 * sizes**0.5).astype(np.int64)
    fit = fit_beta(sizes, types, n_min=10_000)
    assert fit["fitted_from"] >= 10_000
    assert fit["n_points"] < len(sizes)


def test_tokens_for_vocabulary_inverts_the_fit():
    sizes = np.geomspace(100, 1_000_000, 60).astype(np.int64)
    types = (5.0 * sizes**0.55).astype(np.int64)
    fit = fit_beta(sizes, types)
    target = 20_000
    n = tokens_for_vocabulary(fit, target)
    assert fit["K"] * n ** fit["beta"] == pytest.approx(target, rel=1e-6)


# --- drift --------------------------------------------------------------------------------


def test_drift_is_flat_for_a_true_power_law():
    """A curve that really is N ** beta has the same exponent in every window. Anything
    else in a real corpus is the finite-size effect, not an artefact of the windowing."""
    sizes = np.geomspace(100, 10_000_000, 90).astype(np.int64)
    types = (7.0 * sizes**0.6).astype(np.int64)
    betas = [w["beta"] for w in beta_drift(sizes, types)]
    assert max(betas) - min(betas) < 0.01


def test_drift_returns_labelled_windows():
    sizes = np.geomspace(100, 1_000_000, 60).astype(np.int64)
    types = (4.0 * sizes**0.5).astype(np.int64)
    windows = beta_drift(sizes, types, window=12)
    assert len(windows) > 1
    assert windows[0]["from_tokens"] < windows[-1]["from_tokens"]
    assert all(w["to_tokens"] > w["from_tokens"] for w in windows)


# --- the Zipf-Heaps relation --------------------------------------------------------------


def test_asymptotic_relation_is_capped_at_one():
    """beta = min(1, 1/a). Below a = 1 the reciprocal exceeds 1, and a vocabulary cannot
    grow faster than the text."""
    assert beta_asymptotic(2.0) == pytest.approx(0.5)
    assert beta_asymptotic(1.0) == pytest.approx(1.0)
    assert beta_asymptotic(0.5) == pytest.approx(1.0)


def test_simulation_agrees_with_the_asymptotic_form_for_a_steep_exponent():
    """Where the asymptotic result is supposed to hold - a comfortably above 1, so the
    distribution is dominated by its head - the finite-size simulation should land near it."""
    fit = beta_simulated(a=2.0, n_tokens=200_000, support=50_000, seed=0)
    assert fit["beta"] == pytest.approx(beta_asymptotic(2.0), abs=0.12)


def test_simulation_departs_from_the_asymptotic_form_near_a_equals_one():
    """The finite-size regime Lu et al. describe: at a ~ 1 the asymptotic formula says
    beta = 1, and a corpus of finite length does not deliver that."""
    fit = beta_simulated(a=1.0, n_tokens=200_000, support=100_000, seed=0)
    assert fit["beta"] < 0.95


def test_simulation_reports_its_support_choice():
    fit = beta_simulated(a=1.2, n_tokens=20_000, support=5_000, seed=0)
    assert fit["support"] == 5_000
    assert fit["final_types"] <= 5_000


# --- the bill -----------------------------------------------------------------------------


def test_oov_falls_as_the_vocabulary_grows():
    tokens = [f"w{i % 5_000}" for i in range(200_000)]
    rows = oov_cost(tokens, (100, 1_000, 5_000))
    rates = [r["oov_token_rate"] for r in rows]
    assert rates == sorted(rates, reverse=True)


def test_a_vocabulary_covering_everything_costs_nothing():
    tokens = [f"w{i % 50}" for i in range(10_000)]
    rows = oov_cost(tokens, (50,))
    assert rows[0]["oov_token_rate"] == pytest.approx(0.0)


def test_oov_is_measured_on_held_out_text():
    """The vocabulary is cut from the first half and priced on the second. A type that
    appears only in the second half must count as out-of-vocabulary however large the
    vocabulary is - which is exactly what measuring on the training text would hide."""
    tokens = ["a"] * 1_000 + ["b"] * 999 + ["rare"]
    rows = oov_cost(tokens, (10_000,))
    assert rows[0]["oov_token_rate"] > 0


def test_a_vocabulary_larger_than_the_training_half_is_flagged():
    """Past the number of types the training half contains, the vocabulary size parameter
    stops doing anything. The table plateaus, and without the flag that plateau reads as a
    diminishing return rather than as a hard ceiling."""
    tokens = [f"w{i % 300}" for i in range(20_000)]
    rows = oov_cost(tokens, (100, 10_000))
    assert rows[0]["capped_by_training_corpus"] is False
    assert rows[1]["capped_by_training_corpus"] is True


def test_the_floor_is_what_an_unlimited_vocabulary_still_pays():
    """Heaps' law says new types keep arriving, so held-out text contains types no
    training-half vocabulary could have held. That is a floor, not a plateau."""
    tokens = [f"w{i}" for i in range(10_000)]  # every token new: nothing is ever reusable
    floor = irreducible_oov(tokens)
    assert floor["oov_token_rate"] == pytest.approx(1.0)


def test_the_floor_is_zero_for_a_closed_vocabulary():
    tokens = [f"w{i % 20}" for i in range(10_000)]
    assert irreducible_oov(tokens)["oov_token_rate"] == pytest.approx(0.0)


def test_no_vocabulary_size_beats_the_floor():
    tokens = [f"w{(i * 7919) % 4000}" for i in range(60_000)] + [f"new{i}" for i in range(500)]
    floor = irreducible_oov(tokens)["oov_token_rate"]
    for row in oov_cost(tokens, (100, 1_000, 10_000, 100_000)):
        assert row["oov_token_rate"] >= floor - 1e-12


def test_token_rate_and_type_rate_are_different_numbers():
    """A long tail of singletons is most of the types and almost none of the running text.
    Quoting the type rate as though it were the token rate overstates the damage."""
    tokens = (["the"] * 50_000) + [f"rare{i}" for i in range(2_000)] * 2
    rows = oov_cost(tokens, (10,))
    assert rows[0]["oov_type_rate"] > rows[0]["oov_token_rate"] * 2
