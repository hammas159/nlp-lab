"""Five association measures over the same bigram counts.

A collocation is a word pair that occurs together more than chance would allow. Turning
that into a number needs a statistic, and the statistics disagree about which pairs win -
not slightly, but at the top of the list, which is the only part anyone reads.

All five are computed from one contingency table per bigram:

                     w2 present        w2 absent
    w1 present       a = #(w1, w2)     b = #(w1, *) - a
    w1 absent        c = #(*, w2) - a  d = N - a - b - c

``pmi``       log( P(w1, w2) / (P(w1) P(w2)) ). Unbounded below, and maximal for pairs that
              occur once, together, and never apart - which is why it is nearly always used
              with a frequency cutoff that is doing more work than the statistic.
``ppmi``      the positive part of PMI. Changes nothing about the ranking, only the floor.
``t_score``   the difference between observed and expected, scaled by the observed count's
              own standard error. Dominated by frequency, so it favours pairs like "of the".
``llr``       Dunning's log-likelihood ratio, the measure written specifically because
              chi-squared is unreliable on the sparse counts that text produces.
``chi2``      Pearson's, computed on the same table, so the disagreement with ``llr`` is a
              property of the statistics rather than of the data they were given.

Every measure is a function of (a, b, c, d) and nothing else, so `test_association.py` can
check each against arithmetic done by hand on a table written out in the test.
"""

from __future__ import annotations

import numpy as np

MEASURES = ("pmi", "ppmi", "t_score", "llr", "chi2")


def _expected(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    total = a + b + c + d
    return (a + b) * (a + c) / total


def pmi(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(a / _expected(a, b, c, d))
    return np.where(a > 0, out, -np.inf)


def ppmi(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.maximum(pmi(a, b, c, d), 0.0)


def t_score(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """(observed - expected) / sqrt(observed).

    Not a t statistic in any defensible sense - the denominator uses the observed count as
    its own variance estimate - but it is the measure that appears in the corpus-linguistics
    literature under that name, and it behaves very differently from PMI, which is the point.
    """
    expected = _expected(a, b, c, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - expected) / np.sqrt(a)
    return np.where(a > 0, out, 0.0)


def llr(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Dunning's (1993) log-likelihood ratio, as ``2 * sum(observed * log(observed/expected))``.

    Written specifically because chi-squared's normal approximation fails on the counts text
    produces - most bigrams occur once or twice, and the approximation needs expected counts
    of five or more in every cell, which almost no bigram satisfies.
    """
    total = a + b + c + d
    cells = np.stack([a, b, c, d])
    expected = np.stack(
        [
            (a + b) * (a + c) / total,
            (a + b) * (b + d) / total,
            (c + d) * (a + c) / total,
            (c + d) * (b + d) / total,
        ]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = cells * np.log(cells / expected)
    return 2.0 * np.nansum(np.where(cells > 0, terms, 0.0), axis=0)


def chi2(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Pearson's chi-squared on the same 2x2 table."""
    total = a + b + c + d
    numerator = total * (a * d - b * c) ** 2
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    return np.where(denominator > 0, out, 0.0)


FUNCTIONS = {
    "pmi": pmi,
    "ppmi": ppmi,
    "t_score": t_score,
    "llr": llr,
    "chi2": chi2,
}


def score(name: str, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    try:
        return FUNCTIONS[name](a, b, c, d)
    except KeyError:
        raise ValueError(f"unknown measure {name!r}; expected one of {MEASURES}") from None


def expected_cell_below_five(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Does this table violate chi-squared's own validity condition?

    The normal approximation behind Pearson's test needs every expected cell at five or
    more. Reporting the share of bigrams that fail it is how this project shows that the
    chi-squared column is not merely different from the log-likelihood column but
    inadmissible on most of the rows.
    """
    total = a + b + c + d
    expectations = np.stack(
        [
            (a + b) * (a + c) / total,
            (a + b) * (b + d) / total,
            (c + d) * (a + c) / total,
            (c + d) * (b + d) / total,
        ]
    )
    return expectations.min(axis=0) < 5.0
