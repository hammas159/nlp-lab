"""Seven retrievers over one corpus, with everything held constant that can be.

The comparison this repo exists to make is between *methods*, so every knob that is not
the method itself is fixed: the same tokenizer, the same corpus, the same queries, the
same metric. Where a method has a dimensionality, it is set to 300 - word2vec's original
size - so that a win is not simply a wider vector.

Two of the seven are pretrained on billions of words elsewhere. They are included on
purpose, and labelled, because the gap between them and the corpus-trained ones is the
result.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import tokenize
from shared.bm25 import BM25

DIM = 300


@dataclass
class Ranking:
    """Scores for one query against every document, highest first."""

    order: np.ndarray  # doc indices, best first
    name: str


# --- lexical -------------------------------------------------------------------------


class TfIdf:
    name = "TF-IDF"
    trained_on = "this corpus"

    def fit(self, docs: list[str]) -> TfIdf:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.matrix = self.vec.fit_transform(docs)
        return self

    def score(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return np.asarray((self.matrix @ q.T).todense()).ravel()


class LSA:
    """TF-IDF followed by truncated SVD - the 1990s answer to the same problem."""

    name = "LSA (SVD)"
    trained_on = "this corpus"

    def fit(self, docs: list[str]) -> LSA:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tfidf = self.vec.fit_transform(docs)
        # TruncatedSVD cannot produce more components than the vocabulary has terms, and
        # raises rather than clamping. A small corpus is a legitimate input, so clamp here.
        self.n_components = min(DIM, tfidf.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=0)
        self.matrix = _unit(self.svd.fit_transform(tfidf))
        return self

    def score(self, query: str) -> np.ndarray:
        q = _unit(self.svd.transform(self.vec.transform([query])))
        return (self.matrix @ q.T).ravel()


# --- static embeddings, trained on this corpus ----------------------------------------


class _Averaged:
    """Mean of the word vectors in a document.

    Crude, and deliberately so: averaging is what the tutorials do, and the question here
    is what the *method* buys, not what a better pooling strategy buys.
    """

    trained_on = "this corpus"

    def _vectors(self, tokens: list[str]) -> np.ndarray:
        vecs = [self.kv[t] for t in tokens if t in self.kv]
        if not vecs:
            return np.zeros(DIM, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def _embed_all(self, docs: list[str]) -> np.ndarray:
        return _unit(np.vstack([self._vectors(tokenize(d)) for d in docs]))

    def score(self, query: str) -> np.ndarray:
        q = _unit(self._vectors(tokenize(query))[None, :])
        return (self.matrix @ q.T).ravel()


class Word2Vec(_Averaged):
    name = "word2vec (corpus)"

    def fit(self, docs: list[str]) -> Word2Vec:
        from gensim.models import Word2Vec as GW2V

        sentences = [tokenize(d) for d in docs]
        model = GW2V(
            sentences,
            vector_size=DIM,
            window=5,
            min_count=2,
            sg=1,
            negative=5,
            epochs=5,
            workers=4,
            seed=0,
        )
        self.kv = model.wv
        self.matrix = self._embed_all(docs)
        return self


class FastText(_Averaged):
    name = "fastText (corpus)"

    def fit(self, docs: list[str]) -> FastText:
        from gensim.models import FastText as GFT

        sentences = [tokenize(d) for d in docs]
        model = GFT(
            sentences,
            vector_size=DIM,
            window=5,
            min_count=2,
            sg=1,
            negative=5,
            epochs=5,
            workers=4,
            seed=0,
        )
        self.kv = model.wv
        self.matrix = self._embed_all(docs)
        return self


# --- pretrained, trained on billions of words elsewhere -------------------------------


class BGE:
    name = "BGE-small (pretrained)"
    trained_on = "billions of words, elsewhere"

    def fit(self, docs: list[str]) -> BGE:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self.matrix = _unit(
            np.asarray(self.model.encode(docs, batch_size=64, show_progress_bar=False))
        )
        return self

    def score(self, query: str) -> np.ndarray:
        # BGE is trained with an instruction prefix on the query side only.
        prefix = "Represent this sentence for searching relevant passages: "
        q = _unit(np.asarray(self.model.encode([prefix + query], show_progress_bar=False)))
        return (self.matrix @ q.T).ravel()


class NomicEmbed:
    name = "nomic-embed (pretrained)"
    trained_on = "billions of words, elsewhere"
    URL = "http://localhost:11434/api/embeddings"

    def _embed(self, text: str) -> np.ndarray:
        body = json.dumps({"model": "nomic-embed-text", "prompt": text}).encode()
        request = urllib.request.Request(self.URL, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return np.asarray(json.loads(response.read())["embedding"], dtype=np.float32)

    #: Embeddings arrive one HTTP round-trip at a time - a few hundred milliseconds each, so
    #: a 3,000-document corpus is a twenty-minute silence. Checkpoint often enough that an
    #: interrupted run resumes rather than restarts, and say where it is while it works.
    CHECKPOINT_EVERY = 200

    def fit(self, docs: list[str], cache: dict | None = None, checkpoint=None) -> NomicEmbed:
        rows = []
        computed = 0
        for i, d in enumerate(docs):
            if cache is not None and str(i) in cache:
                rows.append(np.asarray(cache[str(i)], dtype=np.float32))
                continue
            v = self._embed(d)
            rows.append(v)
            computed += 1
            if cache is not None:
                cache[str(i)] = v.tolist()
                if checkpoint and computed % self.CHECKPOINT_EVERY == 0:
                    checkpoint(cache)
                    print(f"[{i + 1}/{len(docs)}]", end=" ", flush=True)
        if cache is not None and checkpoint and computed:
            checkpoint(cache)
        self.matrix = _unit(np.vstack(rows))
        return self

    def score(self, query: str) -> np.ndarray:
        q = _unit(self._embed(query)[None, :])
        return (self.matrix @ q.T).ravel()


# --- helpers ---------------------------------------------------------------------------


def _unit(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot product is a cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-10)


LEXICAL = [BM25, TfIdf, LSA]
CORPUS_TRAINED = [Word2Vec, FastText]
PRETRAINED = [BGE, NomicEmbed]
