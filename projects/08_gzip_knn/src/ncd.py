"""Normalised compression distance, and the compressors it can be built from.

Cilibrasi & Vitanyi (2005) define, for a compressor ``C``:

    NCD(x, y) = ( C(xy) - min(C(x), C(y)) ) / max(C(x), C(y))

The intuition is that if ``y`` is similar to ``x`` then compressing them together costs
little more than compressing ``x`` alone, because the compressor can reuse what it already
learned. It needs no training, no vocabulary and no features, which is what makes it
interesting as a classifier baseline - and what makes it expensive, because ``C(xy)`` has
to be recomputed for every pair.

Only the concatenation is pairwise. ``C(x)`` depends on one document and is cached, which
takes the work from three compressions per pair down to one.

**gzip's window is 32 KB.** A concatenation longer than that cannot share references across
the join, so NCD silently degrades towards a constant for long documents. The Devign
functions here average two kilobytes, so a pair is comfortably inside the window; the check
is reported by `window_pressure` rather than assumed.
"""

from __future__ import annotations

import bz2
import gzip
import lzma
from collections.abc import Callable

#: gzip's DEFLATE window. Content further apart than this cannot be matched.
GZIP_WINDOW = 32 * 1024

COMPRESSORS: dict[str, Callable[[bytes], bytes]] = {
    "gzip": lambda raw: gzip.compress(raw, compresslevel=9),
    "bz2": lambda raw: bz2.compress(raw, compresslevel=9),
    "lzma": lambda raw: lzma.compress(raw, preset=9),
}


def compressed_length(text: str, algorithm: str = "gzip") -> int:
    try:
        compress = COMPRESSORS[algorithm]
    except KeyError:
        raise ValueError(
            f"unknown compressor {algorithm!r}; expected one of {sorted(COMPRESSORS)}"
        ) from None
    return len(compress(text.encode("utf-8")))


class CompressedLengths:
    """Memoised ``C(x)`` for a fixed document set."""

    def __init__(self, texts: list[str], algorithm: str = "gzip") -> None:
        self.algorithm = algorithm
        self.lengths = [compressed_length(t, algorithm) for t in texts]

    def __getitem__(self, i: int) -> int:
        return self.lengths[i]

    def __len__(self) -> int:
        return len(self.lengths)


def ncd(x: str, y: str, c_x: int, c_y: int, algorithm: str = "gzip") -> float:
    """NCD from precomputed singleton lengths.

    The concatenation is joined with a single space, as the original implementation does.
    The separator is not cosmetic: joining with nothing lets the compressor match a suffix
    of ``x`` against a prefix of ``y`` across the seam and invent similarity that neither
    document has.
    """
    c_xy = compressed_length(f"{x} {y}", algorithm)
    return (c_xy - min(c_x, c_y)) / max(c_x, c_y)


def window_pressure(texts: list[str]) -> dict:
    """How many pairwise concatenations would exceed gzip's 32 KB window.

    Beyond it the second document cannot reference the first at all and NCD stops measuring
    similarity. Reported rather than assumed, because it is invisible in the output - the
    distances stay in range, they just stop meaning anything.
    """
    sizes = sorted(len(t.encode("utf-8")) for t in texts)
    largest_pair = sizes[-1] + sizes[-2] if len(sizes) > 1 else sizes[-1]
    median = sizes[len(sizes) // 2]
    return {
        "median_bytes": median,
        "largest_pair_bytes": largest_pair,
        "median_pair_bytes": 2 * median,
        "window_bytes": GZIP_WINDOW,
        "worst_case_exceeds_window": largest_pair > GZIP_WINDOW,
        "median_case_exceeds_window": 2 * median > GZIP_WINDOW,
    }
