"""The neural side of the comparison: gensim's skip-gram with negative sampling.

Hyperparameters are matched to project 01's `Word2Vec` retriever exactly - 300 dimensions,
window 5, min_count 2, 5 negative samples, 5 epochs, seed 0 - so that the neural side of
this comparison is a model somebody would actually build, rather than one weakened to make
the counting side look good.

The recall figures here are **not** comparable with project 01's. That project scores on
`build(300)`, 2,964 paragraphs; this one needs a far larger corpus to train embeddings on
at all and uses all 66,581. Recall@10 depends on how many documents the gold two are hiding
among, so only the numbers within this project may be compared with each other.

This is the only file in the project that trains anything. Everything it is compared
against is counting, matrix weighting and one SVD.
"""

from __future__ import annotations

import numpy as np

from factorize import Embedding

DIM = 300
WINDOW = 5
MIN_COUNT = 2
NEGATIVE = 5
EPOCHS = 5
SEED = 0


def available() -> bool:
    try:
        import gensim  # noqa: F401
    except ImportError:
        return False
    return True


def train(
    sentences: list[list[str]],
    dim: int = DIM,
    window: int = WINDOW,
    min_count: int = MIN_COUNT,
    negative: int = NEGATIVE,
    epochs: int = EPOCHS,
    sample: float = 1e-5,
    seed: int = SEED,
    workers: int = 4,
) -> Embedding:
    """Train SGNS and return it in the same container the SVD side uses.

    ``workers=4`` with a fixed seed is not bit-reproducible - gensim's threads interleave
    updates - and that is stated rather than papered over with ``workers=1``, which would
    make a five-epoch run over six million tokens slow enough to discourage re-running it.
    The comparison here is between methods separated by tens of points of recall, not by
    the third decimal.
    """
    from gensim.models import Word2Vec

    model = Word2Vec(
        sentences,
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,
        negative=negative,
        sample=sample,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
    words = list(model.wv.index_to_key)
    vectors = np.vstack([model.wv[w] for w in words]).astype(np.float64)
    return Embedding(words, vectors, label="SGNS (gensim)")
