<h1 align="center">nlp-lab</h1>
<p align="center"><i>Classic NLP techniques, measured against each other on the same corpus</i></p>

<p align="center">
  <a href="#the-projects">The projects</a> &middot;
  <a href="#the-through-line">The through-line</a> &middot;
  <a href="#running-a-project">Running a project</a> &middot;
  <a href="#stack">Stack</a>
</p>

<p align="center">
  <a href="https://github.com/hammas159/nlp-lab/actions/workflows/ci.yml"><img src="https://github.com/hammas159/nlp-lab/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/data-real%20public%20benchmarks-orange" alt="data">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="license"></a>
</p>

---

Classic NLP, done as **comparisons rather than tutorials**. Each project takes a family of
techniques, runs them over the same corpus with everything held constant that can be, and
reports what actually separates them.

The rule throughout: **if two methods differ in more than one way, the comparison is not
about the method.**

---

## The through-line

Most technique comparisons in NLP are not comparisons at all. The usual shape is:

| What is compared | What is actually different |
|---|---|
| TF-IDF *fitted on your corpus* | corpus size, vocabulary, fitting procedure |
| pretrained word2vec | **100 billion words** of training data |
| pretrained BERT | different data, different objective, different dimensionality |

Three methods, but also three training sets, three vocabularies and three vector widths.
The conclusion "the neural one is better" cannot be separated from "the neural one saw more
text".

**Every project here fixes everything except the one variable under test**, and reports the
variable it could not fix as a column rather than a footnote.

---

## The projects

| # | Project | Question | Status |
|---|---|---|---|
| **01** | [Embedding fair comparison](projects/01_embedding_fair_comparison) | Is a neural embedding's advantage the *method*, or the 100 billion words it was trained on? | 🟡 4 of 7 methods measured |

### 01 · Embedding fair comparison

Seven retrievers over one corpus — BM25, TF-IDF, LSA, word2vec and fastText **trained on
this corpus**, and BGE and nomic-embed **trained on billions of words elsewhere**.

Measured so far on HotpotQA (2,964 documents, 300 queries, exactly 2 gold each):

| Method | Trained on | **recall@10** | Index time |
|---|---|---:|---:|
| BGE-small | billions of words | **0.943** | 27.8 s |
| **BM25** | this corpus | **0.865** | **0.1 s** |
| TF-IDF | this corpus | 0.842 | 1.1 s |
| LSA (SVD) | this corpus | 0.735 | 3.3 s |

**BM25 — zero parameters, zero training, a tenth of a second to index — lands within 8
points of a pretrained neural embedding at recall@10, and within 3.4 at recall@20, for
1/270th of the indexing cost.**

🔴 **word2vec and fastText trained on this corpus are still pending.** They are the point of
the project: until a static embedding trained on *893,000 tokens* sits beside one trained on
billions, this is a lexical-vs-pretrained result, not the full experiment.

---

## Planned

Ideas that fit the same shape — each one a family of techniques where the usual comparison
confounds something:

- **Tokenisation** — BPE vs WordPiece vs Unigram on the same corpus, measured by downstream
  retrieval rather than by intrinsic vocabulary statistics
- **Pooling** — mean vs max vs attention-weighted vs `[CLS]`, holding the encoder fixed. The
  embedding comparison above deliberately uses the crudest option; this would measure what
  that costs.
- **Classical topic models** — LSA vs LDA vs NMF on the same corpus, scored on a task rather
  than on coherence
- **Reranking** — how much a cross-encoder recovers on top of each first-stage retriever,
  and whether the ordering of first stages survives it
- **Low-resource morphology** — where subword methods earn their keep, using Urdu, which
  connects to [urdu-nlp-toolkit](https://github.com/hammas159/urdu-nlp-toolkit)

---

## Running a project

```bash
cd projects/01_embedding_fair_comparison
python src/corpus.py 1000       # build and inspect the benchmark
python src/evaluate.py 300      # score every available method
streamlit run ui/app.py         # dashboard
pytest -q                       # tests - no dataset, no network
```

Each project is self-contained: its own `README.md`, `src/`, `tests/`, `ui/` and `results/`.

### Requirements

Python 3.11+. Project 01 reads its dataset from the **local Hugging Face cache** and needs
no download; optional extras (`sentence-transformers`, `gensim`) are only required for the
pretrained and static-embedding methods.

## Stack

`Python 3.11+` &middot; `scikit-learn` &middot; `NumPy` &middot; `pandas` &middot;
`sentence-transformers` &middot; `gensim` &middot; `Ollama` &middot; `Streamlit` &middot;
`Altair` &middot; `pytest` &middot; `ruff` &middot; `GitHub Actions`

## Keywords

NLP &middot; information retrieval &middot; word embeddings &middot; word2vec &middot;
GloVe &middot; fastText &middot; BM25 &middot; TF-IDF &middot; LSA &middot; LSI &middot;
SVD &middot; sentence embeddings &middot; BGE &middot; nomic-embed &middot;
lexical vs dense retrieval &middot; sparse retrieval &middot; recall@k &middot; MRR &middot;
HotpotQA &middot; ablation &middot; fair comparison &middot; training data vs method &middot;
reproducible evaluation &middot; tokenisation &middot; topic models &middot; reranking

## Licence

MIT — see [LICENSE](LICENSE).
