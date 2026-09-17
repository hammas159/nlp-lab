"""Devign C functions, from the local Hugging Face cache.

Two splits are available here - `test` and `validation`, 2,732 functions each. The `train`
split has never been obtainable on this machine, the same 17.85 MB download that blocked
`devign-leakage` and projects 04 and 05. So `validation` is used as the reference set and
`test` as the evaluation set, which is a legitimate pairing as long as it is stated: they
are disjoint, and neither was used to tune anything here.

The task is binary vulnerability detection, and it is chosen deliberately. Compression
distance needs no tokenizer, so it can be pointed at source code without the preprocessing
argument that dominates comparisons on prose. And a binary task with k = 2 is precisely the
setting where the tie-breaking rule decides the most cases.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HUB = Path.home() / ".cache" / "huggingface" / "hub"
DATASET = "datasets--google--code_x_glue_cc_defect_detection"


@dataclass(frozen=True)
class Split:
    texts: list[str]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.texts)

    def subsample(self, n: int, seed: int = 0) -> Split:
        """A seeded random subset, not a prefix.

        Devign is ordered by project, so the first n functions come from a handful of
        codebases and share far more vocabulary than a random draw would. A nearest
        neighbour method would look much better on that, for a reason that has nothing to
        do with compression.
        """
        if n >= len(self.texts):
            return self
        rng = np.random.default_rng([seed, 0xDE7])
        idx = rng.choice(len(self.texts), size=n, replace=False)
        return Split([self.texts[i] for i in idx], self.labels[idx])


def _parquet(split: str) -> Path:
    hits = sorted(
        glob.glob(
            str(HUB / DATASET / "snapshots" / "*" / "**" / f"{split}-*.parquet"), recursive=True
        )
    )
    if not hits:
        raise FileNotFoundError(
            "Devign not found in the Hugging Face cache. Fetch it with:\n"
            '  python -c "from datasets import load_dataset; '
            "load_dataset('google/code_x_glue_cc_defect_detection')\"\n"
            f"Looked under: {HUB / DATASET}"
        )
    return Path(hits[0])


def load(split: str) -> Split:
    import pandas as pd

    frame = pd.read_parquet(_parquet(split))
    return Split(
        texts=[str(f) for f in frame["func"]],
        labels=np.array([bool(t) for t in frame["target"]], dtype=bool),
    )


def load_pair(
    n_reference: int = 1_000, n_evaluation: int = 500, seed: int = 0
) -> tuple[Split, Split]:
    reference = load("validation").subsample(n_reference, seed=seed)
    evaluation = load("test").subsample(n_evaluation, seed=seed)
    return reference, evaluation
