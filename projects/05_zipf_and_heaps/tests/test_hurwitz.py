"""The Hurwitz zeta is checked against closed forms, not against another library.

Replacing SciPy with forty lines is only justified if the forty lines are right, and
"it returns a number near the one SciPy returns" is not a check when SciPy is the thing
being removed. Every assertion here is against a value with an independent definition:
pi^2/6, pi^4/90, Apery's constant, and the function's own recurrence.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hurwitz import zeta

APERY = 1.2020569031595942854


def test_basel_problem():
    """zeta(2, 1) = pi^2 / 6."""
    assert float(zeta(2.0, 1.0)) == pytest.approx(math.pi**2 / 6, rel=1e-12)


def test_zeta_four():
    """zeta(4, 1) = pi^4 / 90."""
    assert float(zeta(4.0, 1.0)) == pytest.approx(math.pi**4 / 90, rel=1e-12)


def test_aperys_constant():
    assert float(zeta(3.0, 1.0)) == pytest.approx(APERY, rel=1e-12)


def test_shifted_start():
    """zeta(2, 2) drops the n = 0 term, so it is pi^2/6 - 1."""
    assert float(zeta(2.0, 2.0)) == pytest.approx(math.pi**2 / 6 - 1.0, rel=1e-12)


def test_recurrence_holds_across_a_range():
    """zeta(s, q) - zeta(s, q+1) = q ** -s, by definition. An implementation that drifts
    will fail this even where it happens to match a single tabulated value.

    The range deliberately includes s just above 1, where the direct sum converges far too
    slowly to check by brute force and the Euler-Maclaurin correction is doing all of the
    work, and s in the hundreds, which the KS search reaches on data that is not a power
    law at all.
    """
    for s in (1.05, 1.5, 2.0, 2.7, 4.0, 200.0):
        for q in (1.0, 3.0, 17.0, 250.0):
            left = float(zeta(s, q)) - float(zeta(s, q + 1))
            assert left == pytest.approx(q**-s, rel=1e-9)


def test_scaled_matches_the_unscaled_value():
    """`scaled=True` returns zeta(s, q) * q ** s. Where both are representable they must
    agree; the scaled path exists for the exponents where the unscaled one underflows."""
    for s in (1.5, 2.5, 6.0):
        for q in (1.0, 7.0, 40.0):
            assert float(zeta(s, q, scaled=True)) == pytest.approx(
                float(zeta(s, q)) * q**s, rel=1e-12
            )


def test_scaled_stays_finite_where_unscaled_underflows():
    """s = 200 at q = 190 is inside the range the KS search visits. Unscaled, q ** -s is
    zero to double precision and a ratio of two zetas becomes 0/0."""
    assert float(zeta(200.0, 190.0)) == 0.0
    scaled = float(zeta(200.0, 190.0, scaled=True))
    assert np.isfinite(scaled) and scaled >= 1.0


def test_vectorised_over_q():
    q = np.array([1.0, 2.0, 5.0, 100.0])
    out = zeta(2.5, q)
    assert out.shape == q.shape
    for i, value in enumerate(q):
        assert out[i] == pytest.approx(float(zeta(2.5, float(value))), rel=1e-12)


def test_monotone_decreasing_in_q():
    q = np.arange(1.0, 50.0)
    out = zeta(3.0, q)
    assert np.all(np.diff(out) < 0)


def test_rejects_divergent_s():
    with pytest.raises(ValueError, match="diverges"):
        zeta(1.0, 1.0)
    with pytest.raises(ValueError, match="diverges"):
        zeta(0.5, 1.0)


def test_rejects_non_positive_q():
    with pytest.raises(ValueError, match="q > 0"):
        zeta(2.0, 0.0)
    with pytest.raises(ValueError, match="q > 0"):
        zeta(2.0, np.array([1.0, -1.0]))


def test_more_direct_terms_do_not_change_the_answer():
    """If the asymptotic correction is right, moving the cut between the direct sum and
    the correction must not move the result."""
    for s in (1.3, 2.0, 3.5):
        a = float(zeta(s, 1.0, n_terms=12))
        b = float(zeta(s, 1.0, n_terms=60))
        assert a == pytest.approx(b, rel=1e-11)
