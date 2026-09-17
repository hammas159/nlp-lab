"""Tests for the five association measures.

Each is checked against arithmetic written out in the test, on a contingency table small
enough to verify by hand. A measure that is off by a normalising constant still produces a
plausible ranking, so agreement between two of them is not evidence that either is right.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from association import (
    MEASURES,
    chi2,
    expected_cell_below_five,
    llr,
    pmi,
    ppmi,
    score,
    t_score,
)


def table(a: float, b: float, c: float, d: float):
    """One contingency table as four length-1 arrays."""
    return (np.array([a]), np.array([b]), np.array([c]), np.array([d]))


# --- PMI ----------------------------------------------------------------------------------


def test_pmi_of_an_independent_pair_is_zero():
    """a = 25, and the marginals are 50 and 50 out of 100, so expected is exactly 25."""
    assert pmi(*table(25, 25, 25, 25))[0] == pytest.approx(0.0)


def test_pmi_matches_hand_arithmetic():
    a, b, c, d = 10, 40, 90, 860
    expected = (a + b) * (a + c) / (a + b + c + d)
    assert pmi(*table(a, b, c, d))[0] == pytest.approx(math.log(a / expected))


def test_pmi_is_maximal_for_a_pair_that_never_occurs_apart():
    """The whole problem with PMI in one test. A pair seen once, whose words are seen
    nowhere else, scores higher than a pair seen a thousand times that is genuinely
    associated - because PMI measures surprise, not evidence."""
    once_exclusive = pmi(*table(1, 0, 0, 999_999))[0]
    frequent_associated = pmi(*table(1_000, 1_000, 1_000, 997_000))[0]
    assert once_exclusive > frequent_associated


def test_ppmi_only_clips_the_floor():
    """It cannot change any ranking among pairs that were already positive, which is why
    swapping PMI for PPMI changes nothing about a top-N list."""
    a, b, c, d = table(50, 10, 10, 930)
    assert ppmi(a, b, c, d)[0] == pytest.approx(pmi(a, b, c, d)[0])
    assert ppmi(*table(1, 100, 100, 799))[0] == 0.0


def test_pmi_of_an_absent_pair_is_negative_infinity():
    assert pmi(*table(0, 10, 10, 980))[0] == -np.inf


# --- t score ------------------------------------------------------------------------------


def test_t_score_matches_hand_arithmetic():
    a, b, c, d = 100, 400, 400, 99_100
    expected = (a + b) * (a + c) / (a + b + c + d)
    assert t_score(*table(a, b, c, d))[0] == pytest.approx((a - expected) / math.sqrt(a))


def test_t_score_prefers_frequency_where_pmi_prefers_exclusivity():
    """The two measures disagree by construction, which is why reporting one without the
    other decides the answer in advance."""
    rare = table(1, 0, 0, 999_999)
    frequent = table(1_000, 1_000, 1_000, 997_000)
    assert pmi(*rare)[0] > pmi(*frequent)[0]
    assert t_score(*rare)[0] < t_score(*frequent)[0]


# --- log-likelihood ratio and chi-squared -------------------------------------------------


def test_llr_of_a_perfectly_independent_table_is_zero():
    assert llr(*table(25, 25, 25, 25))[0] == pytest.approx(0.0, abs=1e-12)


def test_chi2_of_a_perfectly_independent_table_is_zero():
    assert chi2(*table(25, 25, 25, 25))[0] == pytest.approx(0.0, abs=1e-12)


def test_chi2_matches_the_closed_form():
    a, b, c, d = 30, 20, 20, 30
    total = a + b + c + d
    expected = total * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    assert chi2(*table(a, b, c, d))[0] == pytest.approx(expected)


def test_llr_handles_zero_cells_without_a_nan():
    """`0 * log(0)` is 0 by convention and NaN in floating point. A NaN here would sort to
    an arbitrary position rather than erroring."""
    out = llr(*table(5, 0, 0, 95))
    assert np.isfinite(out[0])


def test_llr_and_chi2_disagree_on_sparse_tables():
    """Dunning wrote the log-likelihood ratio because chi-squared's normal approximation
    fails on the counts text produces. If they agreed there would have been no paper."""
    sparse = table(3, 1, 1, 99_995)
    assert abs(llr(*sparse)[0] - chi2(*sparse)[0]) > 1.0


def test_both_grow_with_evidence():
    """Ten times the counts in the same proportions is ten times the evidence, and unlike
    PMI these measures must notice."""
    small = table(10, 10, 10, 970)
    large = table(100, 100, 100, 9_700)
    assert llr(*large)[0] > llr(*small)[0]
    assert chi2(*large)[0] > chi2(*small)[0]


def test_pmi_does_not_grow_with_evidence():
    """The same proportions at ten times the scale give PMI the identical score. It is a
    ratio, so it cannot distinguish one observation from a hundred."""
    small = pmi(*table(10, 10, 10, 970))[0]
    large = pmi(*table(100, 100, 100, 9_700))[0]
    assert small == pytest.approx(large)


# --- admissibility ------------------------------------------------------------------------


def test_a_sparse_table_is_flagged_as_inadmissible_for_chi_squared():
    assert expected_cell_below_five(*table(1, 1, 1, 99_997))[0]


def test_a_dense_table_is_admissible():
    assert not expected_cell_below_five(*table(50, 50, 50, 50))[0]


# --- the dispatcher -----------------------------------------------------------------------


@pytest.mark.parametrize("measure", MEASURES)
def test_every_measure_returns_one_score_per_row(measure):
    a, b, c, d = (np.array([10.0, 20.0, 5.0]) for _ in range(4))
    assert score(measure, a, b, c, d).shape == (3,)


def test_an_unknown_measure_is_rejected():
    with pytest.raises(ValueError, match="unknown measure"):
        score("mutual_information", *table(1, 1, 1, 1))
