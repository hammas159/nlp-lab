"""The SMART weighting triples, implemented from the definitions rather than assumed.

"TF-IDF" is written as though it were one algorithm. It is a family, and Salton & Buckley
(1988) already named the family's members with a three-letter code per side of the
comparison - one letter each for the term-frequency component, the document-frequency
component and the normalisation - written ``ddd.qqq`` for the document and query halves.

    term frequency          n natural       tf
                            l logarithm     1 + log tf
                            a augmented     0.5 + 0.5 * tf / max_tf(d)
                            b boolean       1 if tf > 0
                            L log average   (1 + log tf) / (1 + log avg_tf(d))

    document frequency      n none          1
                            t idf           log(N / df)
                            p prob idf      max(0, log((N - df) / df))

    normalisation           n none          1
                            c cosine        1 / sqrt(sum of squared weights)
                            u pivoted       1 / ((1 - slope) * pivot + slope * unique(d))

Five by three by three is forty-five schemes per side. The textbook default, ``lnc.ltc``,
is one of them, and nothing in the notation says it is the best one - which is the question
this project exists to answer.

Zobel & Moffat (1998) searched a far larger space of similarity formulations and reported
that effectiveness varied widely across it. This is the small, legible version of that
search on a corpus with exact ground truth.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

TF_CODES = ("n", "l", "a", "b", "L")
DF_CODES = ("n", "t", "p")
NORM_CODES = ("n", "c", "u")

#: Singhal, Buckley & Mitra (1996) fit the slope empirically; 0.2 is their reported value
#: and is used here as a stated constant rather than tuned on this corpus.
PIVOT_SLOPE = 0.2


def _row_max(matrix: sparse.csr_matrix) -> np.ndarray:
    out = np.asarray(matrix.max(axis=1).todense()).ravel()
    out[out == 0] = 1.0
    return out


def _row_mean_of_nonzeros(matrix: sparse.csr_matrix) -> np.ndarray:
    sums = np.asarray(matrix.sum(axis=1)).ravel()
    counts = np.diff(matrix.indptr).astype(np.float64)
    counts[counts == 0] = 1.0
    out = sums / counts
    out[out == 0] = 1.0
    return out


def term_frequency(counts: sparse.csr_matrix, code: str) -> sparse.csr_matrix:
    """Apply one TF variant to a raw count matrix, in place on the stored values.

    Every variant is a function of the stored nonzeros alone except ``a`` and ``L``, which
    need a per-row statistic. Both are computed from nonzeros only: a zero count is an
    absent term, not a term that occurred zero times, and averaging over the whole
    vocabulary would make every document's average essentially zero.
    """
    out = counts.copy().astype(np.float64)
    if code == "n":
        return out
    if code == "b":
        out.data = np.ones_like(out.data)
        return out
    if code == "l":
        out.data = 1.0 + np.log(out.data)
        return out

    rows = np.repeat(np.arange(out.shape[0]), np.diff(out.indptr))
    if code == "a":
        out.data = 0.5 + 0.5 * out.data / _row_max(counts)[rows]
        return out
    if code == "L":
        out.data = (1.0 + np.log(out.data)) / (1.0 + np.log(_row_mean_of_nonzeros(counts)[rows]))
        return out
    raise ValueError(f"unknown term-frequency code {code!r}; expected one of {TF_CODES}")


def document_frequency(counts: sparse.csr_matrix, code: str) -> np.ndarray:
    """The per-term collection factor. Computed from the document matrix, never the query.

    An idf estimated on the queries would be estimated on a handful of words and would
    leak the query distribution into the weighting.
    """
    n_docs = counts.shape[0]
    df = np.diff(counts.tocsc().indptr).astype(np.float64)

    if code == "n":
        return np.ones(counts.shape[1], dtype=np.float64)
    if code == "t":
        with np.errstate(divide="ignore"):
            out = np.log(n_docs / np.maximum(df, 1.0))
        return np.where(df > 0, out, 0.0)
    if code == "p":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.log((n_docs - df) / np.maximum(df, 1.0))
        # Robertson's probabilistic idf goes negative for terms in over half the
        # collection. Clamping at zero is the standard remedy and is what makes `p`
        # different from `t` rather than merely shifted.
        return np.where(df > 0, np.maximum(out, 0.0), 0.0)
    raise ValueError(f"unknown document-frequency code {code!r}; expected one of {DF_CODES}")


def normalise(
    matrix: sparse.csr_matrix, code: str, slope: float = PIVOT_SLOPE
) -> sparse.csr_matrix:
    """Scale each row by the chosen normalisation."""
    out = matrix.tocsr().astype(np.float64)
    n_rows = out.shape[0]

    if code == "n":
        return out
    if code == "c":
        lengths = np.sqrt(np.asarray(out.multiply(out).sum(axis=1)).ravel())
    elif code == "u":
        unique = np.diff(out.indptr).astype(np.float64)
        pivot = unique.mean() or 1.0
        lengths = (1.0 - slope) * pivot + slope * unique
    else:
        raise ValueError(f"unknown normalisation code {code!r}; expected one of {NORM_CODES}")

    lengths[lengths == 0] = 1.0
    return sparse.diags(1.0 / lengths, shape=(n_rows, n_rows)) @ out


def weight(
    counts: sparse.csr_matrix,
    scheme: str,
    idf: np.ndarray | None = None,
    slope: float = PIVOT_SLOPE,
) -> sparse.csr_matrix:
    """Apply a three-letter SMART code to a raw count matrix.

    ``idf`` must be supplied when weighting queries, so that both sides use the collection
    statistics of the *documents*. Omitting it silently computes idf over the query set,
    which is the most common way to get a query-side weighting wrong.
    """
    if len(scheme) != 3:
        raise ValueError(f"a SMART code is three letters, got {scheme!r}")
    tf_code, df_code, norm_code = scheme

    weighted = term_frequency(counts, tf_code)
    factor = document_frequency(counts, df_code) if idf is None else idf_for(df_code, idf)
    if factor is not None:
        weighted = weighted @ sparse.diags(factor)
    return normalise(weighted, norm_code, slope=slope)


def idf_for(df_code: str, precomputed: dict[str, np.ndarray]) -> np.ndarray | None:
    """Pick the document-side collection factor matching this code."""
    if df_code not in DF_CODES:
        raise ValueError(f"unknown document-frequency code {df_code!r}")
    return precomputed[df_code]


def all_schemes() -> list[str]:
    return [f"{t}{d}{n}" for t in TF_CODES for d in DF_CODES for n in NORM_CODES]
