<h1 align="center">07 · The SMART weighting grid</h1>
<p align="center"><i>"TF-IDF" names forty-five different schemes. They span 0.400 of recall — five times the gap between BM25 and a pretrained neural embedding on the same corpus.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#which-decision-carries-the-weight">Which decision</a> &middot;
  <a href="#a-third-of-the-query-code-does-nothing">The inert third</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-101%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/runtime-12%20seconds-success" alt="runtime">
</p>

---

> ### The best weighting scheme beats the worst by 0.400 recall@10. Project 01's headline gap — BM25 against a pretrained neural embedding — is 0.078 on this same corpus.

Salton & Buckley (1988) named the family in a three-letter code per side: one letter for the
term-frequency component, one for the document-frequency component, one for the
normalisation, written `ddd.qqq`.

| | | |
|---|---|---|
| **term frequency** | `n` natural | `tf` |
| | `l` logarithm | `1 + log tf` |
| | `a` augmented | `0.5 + 0.5 · tf / max_tf(d)` |
| | `b` boolean | `1 if tf > 0` |
| | `L` log average | `(1 + log tf) / (1 + log avg_tf(d))` |
| **document frequency** | `n` none | `1` |
| | `t` idf | `log(N / df)` |
| | `p` prob idf | `max(0, log((N − df) / df))` |
| **normalisation** | `n` none | `1` |
| | `c` cosine | `1 / √Σw²` |
| | `u` pivoted | `1 / ((1 − slope)·pivot + slope·unique(d))` |

Five by three by three is **forty-five schemes per side**. Every one of them is "TF-IDF".

---

## The result

HotpotQA, 2,964 paragraphs, 300 questions, two gold paragraphs each — the same corpus
projects 01, 02 and 03 score on. Query weighting held at the textbook `ltc`.

| Rank | Scheme | recall@10 |
|---:|---|---:|
| 1 | `atc` | **0.893** |
| 2 | `apc` | 0.893 |
| 3 | `bpc` | 0.893 |
| 4 | `btc` | 0.892 |
| 5 | `lpc` | 0.890 |
| … | | |
| 17 | **`lnc`** — the textbook default | **0.872** |
| … | | |
| 43 | `npn` | 0.703 |
| 44 | `nnu` | 0.645 |
| 45 | `nnn` | 0.493 |

**Spread: 0.400.** The textbook scheme ranks **17th of 45**.

Against the baselines already in this lab:

| | recall@10 | |
|---|---:|---|
| best scheme `atc` | **0.893** | |
| BM25 (`shared/bm25.py`) | 0.865 | `atc` beats it by **+0.028** [+0.012, +0.045], p = 0.000 |
| textbook `lnc` | 0.872 | `atc` beats it by **+0.022** [+0.002, +0.040], p = 0.047 |
| worst scheme `nnn` | 0.493 | |

Both differences survive a paired bootstrap over the 300 queries. The BM25 figure here is
**0.865, identical to the number project 01 reports**, which is the check that these two
tables are measuring the same thing on the same corpus.

So a paper reporting "we replaced TF-IDF with *X* and gained 3 points" has reported a
number smaller than the distance between two things both called TF-IDF — and in most cases
has not said which one it started from.

---

## Which decision carries the weight

Mean recall across all 45 schemes, grouped by one letter at a time. The group mean is an
average over fifteen schemes, which is a steadier summary than naming the single best one.

**Document side:**

| Component | | | | | | Spread |
|---|---|---|---|---|---|---:|
| term frequency | `n` 0.734 | `l` 0.830 | `L` 0.861 | `b` 0.866 | `a` 0.880 | **0.146** |
| normalisation | `n` 0.792 | `u` 0.833 | `c` 0.877 | | | **0.085** |
| document frequency | `n` 0.800 | `t` 0.851 | `p` 0.852 | | | **0.053** |

**The term-frequency component matters roughly three times as much as the idf component.**
That inverts the usual emphasis: "TF-IDF" is named after the idf, tutorials explain the
idf, and the idf is the smallest of the three decisions here. Raw counts (`n`) are the
single worst choice on either axis — any squashing of term frequency, log or augmented or
even boolean, is worth more than adding idf.

**Query side** (document weighting held at `lnc`):

| Component | | | | Spread |
|---|---|---|---|---:|
| document frequency | `n` 0.654 | `t` 0.872 | `p` 0.873 | **0.219** |
| term frequency | `n` 0.791 | … | `b` 0.803 | **0.012** |
| normalisation | `n` 0.799 | `c` 0.799 | `u` 0.799 | **0.000** |

The two sides are not mirror images. On the query side the idf is nearly everything
(**0.219**) and the term-frequency component is almost irrelevant (**0.012**) — queries are
short, so a term rarely repeats and there is little frequency to weight.

---

## A third of the query code does nothing

The query normalisation spread is not small. It is **exactly zero**, for all 15
term-weighting families.

Normalising a query vector multiplies every document's score *for that query* by one
constant. A constant cannot reorder a list. So the third letter of the query code is inert
for every rank-based metric — recall, precision, MAP, NDCG — and **the 45 query schemes are
15 distinct rankings wearing 45 names.**

`ltc` and `ltn` and `ltu` are the same retrieval system. The notation gives no hint of it,
and a grid search over query schemes spends two thirds of its budget re-measuring rankings
it has already seen.

`test_query_normalisation_cannot_change_the_ranking` asserts it, and
`test_document_normalisation_does_change_the_ranking` asserts the opposite for the document
side — which is what makes the first a finding rather than a broken scorer.

---

## Method

One index of raw counts, built once. Every weighting is a transformation of that matrix, so
evaluating ninety schemes costs ninety sparse matrix products rather than ninety indexing
passes — the whole study runs in **12 seconds**.

**One axis at a time.** Forty-five document schemes against a fixed query scheme, then
forty-five query schemes against a fixed document scheme. Varying both would give 2,025
numbers and no attribution.

**idf always comes from the documents.** Weighting queries with an idf estimated over the
query set would estimate it from a handful of short texts and leak the query distribution
into the weighting. `test_queries_are_weighted_with_document_collection_statistics` pins
that down.

**Cosine is not applied in the scorer.** In SMART notation cosine *is* the `c`
normalisation. Normalising again at scoring time would silently convert every scheme in the
grid into its cosine variant and collapse the thing being measured.

**The corpus is project 01's**, `build(300)` — 2,964 paragraphs, 300 questions, 25,295
terms. Recall@10 depends on how many documents the gold two are hiding among, so a
different corpus size would have made the comparison against project 01's 0.078 gap
meaningless.

---

## Limitations

- **One collection, one query set.** Zobel & Moffat (1998) reported that the best
  formulation is not stable across collections. Nothing here tests that, so `atc` is the
  best scheme *on this corpus*, not a recommendation.
- **The pivot slope is fixed at 0.2**, the value Singhal, Buckley & Mitra (1996) report. It
  is not tuned here, and tuning it would improve `u` by an unknown amount — so the `u`
  column is a lower bound on what pivoted normalisation can do.
- **Recall@10 only.** A metric that weights rank position, such as NDCG, could order the
  schemes differently. The query-normalisation result is the exception: it holds for any
  rank-based metric, because the ranking itself is unchanged.
- **Forty-five is not the whole family.** SMART has more letters than the five, three and
  three used here, and Zobel & Moffat's space was far larger. The spread reported is
  therefore a lower bound on the family's spread.
- **BM25 is not in the grid.** It is a different parameterisation with its own `k1` and `b`,
  shown as a reference point rather than as a 46th scheme.
- **Significance is over 300 queries.** The `atc` versus `lnc` difference has p = 0.047 —
  real by the usual threshold, but only just, and a different query sample could move it.

## Run it

```bash
python src/run.py            # every table above, 12 seconds
python src/run.py --quick    # a smaller corpus
pytest -q                    # 101 tests, no dataset, no network
```

## Layout

```
src/smart.py    the five TF, three DF and three normalisation components
src/index.py    raw count matrices for documents and queries, one vocabulary
src/search.py   inner-product scoring, recall@k, paired bootstrap, component summary
src/run.py      the two axes, BM25, and the significance tests
results/        grid.json
```

## Keywords

TF-IDF · term weighting · SMART notation · Salton Buckley · lnc.ltc · cosine normalisation ·
pivoted length normalisation · inverse document frequency · probabilistic idf · BM25 ·
vector space model · sparse retrieval · recall@k · paired bootstrap · ablation ·
hyperparameter sensitivity · information retrieval · HotpotQA
