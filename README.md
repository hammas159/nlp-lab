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
| **02** | [Preprocessing ablation](projects/02_preprocessing_ablation) | Which parts of the standard NLP pipeline actually help - and do they compose? | ✅ complete |
| **03** | [The reranker ceiling](projects/03_reranker_ceiling) | Does a reranker rescue a weak first stage, or only reorder it? | ✅ complete |

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

### 02 · Preprocessing ablation

Every tutorial teaches lowercase → strip punctuation → remove stopwords → stem → drop
short tokens as one step called "preprocessing". It is several independent decisions, and
**they are not additive.**

| Variant | recall@10 | Δ | p | Verdict |
|---|---:|---:|---:|---|
| baseline (lowercase only) | 0.865 | — | — | *baseline* |
| **stopwords + stemming** | **0.888** | **+0.023** | **0.005** | ✅ **REAL** |
| + Porter stemming | 0.887 | +0.022 | 0.012 | ✅ REAL |
| + remove stopwords | 0.877 | +0.012 | 0.034 | ✅ REAL |
| **the full tutorial pipeline** | 0.875 | +0.010 | **0.360** | ❌ **within noise** |
| + drop tokens < 3 chars | 0.862 | −0.003 | 0.676 | ❌ within noise |

**Stemming works. Stopword removal works. Add a third step that does nothing on its own,
and the significant +2.3 point gain becomes indistinguishable from noise.**

The mechanism: Porter stemming *produces* short stems (`aging` → `ag`), and the length
filter then deletes exactly the tokens stemming just created. That interaction is invisible
if "preprocessing" is evaluated as one block — which is how it is almost always taught.

Significance is a **paired bootstrap over queries**, because a table of six numbers two
points apart invites a ranking that the sample size may not support. Three of six
differences here are real; the other three are reported as noise rather than ranked.

### 03 · The reranker ceiling

A cross-encoder can only **reorder** what the first stage handed it, so the first stage's
recall@50 is a hard ceiling on anything achievable at k ≤ 10.

| First stage | r@10 before | r@10 after | Ceiling (r@50) | Converted |
|---|---:|---:|---:|---:|
| **BM25** | 0.865 | **0.922** | 0.967 | 95.3% |
| TF-IDF | 0.842 | 0.918 | 0.955 | 96.1% |
| BGE-small | 0.943 | 0.942 | 0.978 | 96.3% |
| random (control) | 0.003 | 0.022 | 0.022 | 100% |

**The spread between real retrievers collapses from 0.102 to 0.023.** BGE gains nothing
(−0.001) because it was already ordering well; the cheap retrievers are the ones the
reranker rescues.

Every first stage converts **about 96% of its ceiling**, whichever one it is. So what
separates retrievers is not how well they rank, but **what they fetch at all** — the first
stage's job is recall@50, not precision@10.

The **random control** is what makes the ceiling a measurement rather than an assertion: it
converts 100% of its ceiling and still scores 0.022, because a reranker cannot retrieve a
document that was never fetched.

**This reframes project 01.** If a reranker is in the pipeline — and in any serious RAG
system it is — BM25's 8-point deficit becomes 2, for 1/270th of the indexing cost.

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
- **Retrieval depth** — project 03 fixes the reranker's window at 50. The whole finding
  is a function of that number, and sweeping it is the obvious follow-up
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
