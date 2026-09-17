"""TF-IDF vectors for the Devign functions, and the Devign loader.

Written out rather than imported so that the weighting is visible: log term frequency, idf,
cosine normalisation - `ltc` in the SMART notation project 07 measures. Project 07 shows
that choice is worth 0.4 of recall on a retrieval task, so it is stated here rather than
left to a library default.
"""

from __future__ import annotations

import glob
import re
from collections import Counter
from pathlib import Path

import numpy as np

HUB = Path.home() / ".cache" / "huggingface" / "hub"
DATASET = "datasets--google--code_x_glue_cc_defect_detection"

TOKEN = re.compile(r"[A-Za-z_]\w*|\d+|[^\s\w]")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text)


def load_devign(
    split: str = "test", limit: int | None = None
) -> tuple[list[str], np.ndarray, list[str]]:
    """Return (texts, provenance labels, label names).

    The label is Devign's ``project`` column - qemu or FFmpeg - not its vulnerability
    target. Provenance is the structure a clustering of source code would actually be
    expected to find, and it is strongly imbalanced (roughly 64/36), which is the point:
    an internal metric with no access to labels has no way to prefer an uneven split.
    """
    import pandas as pd

    hits = sorted(
        glob.glob(
            str(HUB / DATASET / "snapshots" / "*" / "**" / f"{split}-*.parquet"), recursive=True
        )
    )
    if not hits:
        raise FileNotFoundError(
            "Devign not found in the Hugging Face cache. Fetch it with:\n"
            '  python -c "from datasets import load_dataset; '
            "load_dataset('google/code_x_glue_cc_defect_detection')\""
        )
    frame = pd.read_parquet(hits[0])
    if limit:
        frame = frame.head(limit)

    names = sorted(frame["project"].unique())
    index = {name: i for i, name in enumerate(names)}
    return (
        [str(f) for f in frame["func"]],
        np.array([index[p] for p in frame["project"]], dtype=np.int64),
        names,
    )


def tfidf(
    texts: list[str], min_df: int = 3, max_features: int = 5_000
) -> tuple[np.ndarray, list[str]]:
    """Dense ``ltc`` vectors.

    Dense on purpose: the silhouette coefficient needs every pairwise distance, so the
    matrix is materialised anyway and a sparse one would only add conversions. That is also
    why `max_features` exists - the full Devign vocabulary is far larger than the clustering
    needs, and the pairwise distance matrix is what actually bounds the corpus size here.
    """
    counts = Counter(t for text in texts for t in set(tokenize(text)))
    vocabulary = [w for w, c in counts.most_common(max_features) if c >= min_df]
    position = {w: i for i, w in enumerate(vocabulary)}

    matrix = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for i, text in enumerate(texts):
        for token, n in Counter(tokenize(text)).items():
            j = position.get(token)
            if j is not None:
                matrix[i, j] = 1.0 + np.log(n)

    document_frequency = (matrix > 0).sum(axis=0)
    idf = np.log(len(texts) / np.maximum(document_frequency, 1))
    matrix *= idf

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms, vocabulary
