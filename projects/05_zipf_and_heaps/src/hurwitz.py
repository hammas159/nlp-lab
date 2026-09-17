"""The Hurwitz zeta function, implemented rather than imported.

The discrete power law has normalising constant ``zeta(s, q) = sum_{n>=0} (n + q) ** -s``,
and the Kolmogorov-Smirnov test in `zipf.py` needs its exact CDF. That is the only thing
this project wanted from SciPy, and forty lines of Euler-Maclaurin is a better trade than
a forty-megabyte wheel on a slow connection - the same reasoning that put BM25 in
`shared/bm25.py` instead of a dependency.

Euler-Maclaurin, summing the first ``n_terms`` directly and correcting the remainder:

    zeta(s, q) ~ sum_{n<N} (n+q)^-s
               + (N+q)^(1-s) / (s-1)
               + (N+q)^-s / 2
               + sum_k  B_2k / (2k)! * (s)_(2k-1) * (N+q)^(-s-2k+1)

where ``(s)_(2k-1) = s(s+1)...(s+2k-2)`` is the rising factorial and the Bernoulli numbers
carry their own signs.

The Bernoulli correction converges quickly for ``s > 1`` and ``q >= 1``, which is the only
region a power-law fit ever uses. `test_hurwitz.py` checks the implementation against
closed forms (pi^2/6, Apery's constant) rather than against another library.
"""

from __future__ import annotations

import numpy as np

#: B_2, B_4, ... B_14. Only even-index Bernoulli numbers appear; the odd ones vanish.
BERNOULLI_EVEN = (
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
)


def _factorial(n: int) -> float:
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out


def zeta(
    s: float,
    q: np.ndarray | float,
    n_terms: int = 24,
    n_bernoulli: int = 6,
    scaled: bool = False,
) -> np.ndarray:
    """``zeta(s, q)`` for real ``s > 1`` and ``q > 0``, vectorised over ``q``.

    ``n_terms`` direct terms before the asymptotic correction takes over. Raising it costs
    almost nothing and buys accuracy at small ``q``, which is where the direct sum is
    slowest to settle.

    With ``scaled=True`` the result is ``zeta(s, q) * q ** s`` instead. Every term is then
    a ratio of comparable magnitudes, so nothing underflows. The KS search visits fitted
    exponents in the twenties on data that is not power-law at all, where ``q ** -s``
    underflows to zero and an unscaled ratio of two zetas becomes 0/0.
    """
    if s <= 1.0:
        raise ValueError(f"zeta(s, q) diverges for s <= 1; got s={s}")

    q = np.asarray(q, dtype=np.float64)
    if np.any(q <= 0):
        raise ValueError("zeta(s, q) requires q > 0")

    # Direct part: sum over n = 0 .. n_terms-1, written as (1 + n/q) ** -s so that the
    # scaled and unscaled paths share one expression.
    offsets = np.arange(n_terms, dtype=np.float64)
    total = np.sum((1.0 + offsets / q[..., None]) ** -s, axis=-1)

    # Remainder, expanded about N + q, also relative to q.
    ratio = 1.0 + n_terms / q
    total += q * ratio ** (1.0 - s) / (s - 1.0)
    total += 0.5 * ratio**-s

    # Bernoulli correction. (s)_(2k-1) is the rising factorial s(s+1)...(s+2k-2).
    rising = s
    for k in range(1, min(n_bernoulli, len(BERNOULLI_EVEN)) + 1):
        if k > 1:
            for j in range(2 * k - 3, 2 * k - 1):
                rising *= s + j
        total += (
            BERNOULLI_EVEN[k - 1]
            / _factorial(2 * k)
            * rising
            * q ** (1 - 2 * k)
            * ratio ** (-s - 2 * k + 1)
        )

    return total if scaled else total * q**-s
