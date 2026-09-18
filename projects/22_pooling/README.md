<h1 align="center">22 · Pooling</h1>
<p align="center"><i>Five poolings, three distinct vectors, and one of them is another one wearing a different name.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#two-of-the-five-are-not-different-poolings">Not five poolings</a> &middot;
  <a href="#what-that-leaves">What that leaves</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-37%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/cosine(cls%2C%20last)-1.000000-critical" alt="identical">
  <img src="https://img.shields.io/badge/second%20encoder-pending%20download-yellow" alt="pending">
</p>

---

A bi-encoder produces one vector per **token**. Turning that into one vector per **sentence**
needs a pooling step, and it is usually presented as a free hyperparameter — mean is the
default, `[CLS]` is "what BERT does", max is occasionally tried.

[Project 01](../01_embedding_fair_comparison) flagged this as a gap: it used the crudest
option for its static embeddings and noted that measuring the cost was a separate project.
This is that project. The encoder is held fixed and only the pooling changes.

---

## The result

HotpotQA · 2,964 documents · 300 queries · `bge-small-en-v1.5`, which ships
`pooling_mode_cls_token: true`. BM25 on the same corpus scores 0.865.

| Pooling | recall@10 | vs trained |
|---|---:|---:|
| **cls** *(as trained)* | **0.945** | — |
| mean | 0.945 | +0.000 |
| last | 0.945 | +0.000 |
| idf_mean | 0.938 | −0.007 |
| max | 0.908 | −0.037 |

The spread is **0.037**, and three of the five are identical to three decimals. That looks
like "pooling barely matters" — a tidy, forgettable result.

It is the wrong reading, and the way to find out is to stop looking at the scores and look
at the vectors.

---

## Two of the five are not different poolings

Mean cosine between the document vectors each pooling produces, over all 2,964 documents:

| | cls | mean | max | last | idf_mean |
|---|---:|---:|---:|---:|---:|
| **cls** | 1.000 | 0.932 | 0.513 | **1.000** | 0.918 |
| **mean** | 0.932 | 1.000 | 0.537 | 0.932 | **0.996** |
| **max** | 0.513 | 0.537 | 1.000 | 0.513 | 0.539 |
| **last** | **1.000** | 0.932 | 0.513 | 1.000 | 0.918 |
| **idf_mean** | 0.918 | **0.996** | 0.539 | 0.918 | 1.000 |

**`cls` and `last` are the same vector.** Not similar — identical, to six decimal places:

```
first token: [CLS]   last real token: [SEP]
  cosine(CLS, last) = 1.000000   max abs diff = 0.000046
```

"Last-token pooling" means the last *unmasked position*, and on a BERT-style encoder the
sequence is `[CLS] w1 w2 … [SEP]`. So it reads `[SEP]`, not the final content word — and on
this model `[SEP]`'s representation has converged onto `[CLS]`'s during contrastive training.
`last` is `cls` under another name here, and its row in the results table carries no
information at all.

Nothing about the arithmetic is wrong. The *name* is what misleads: on a decoder-style model
with no suffix token, `last` would mean something entirely different.

**`mean` and `idf_mean` are nearly the same too**, at 0.996. Weighting a mean by inverse
document frequency barely moves it, because the mean over a few hundred tokens is dominated
by the bulk of them and IDF only reweights the tail.

So the five-way comparison is really a **three-way** one: `{cls, last}`, `{mean, idf_mean}`,
and `max` — and `max` is the only one that is genuinely far from the others, at cosine
0.51–0.54 with everything.

---

## What that leaves

Once the duplicates are collapsed, the actual finding is sharper than the score table
suggested:

| Distinct vector | recall@10 | cosine to `cls` |
|---|---:|---:|
| `cls` / `last` *(as trained)* | **0.945** | 1.000 |
| `mean` / `idf_mean` | 0.945 / 0.938 | 0.932 / 0.918 |
| `max` | **0.908** | 0.513 |

**Mean pooling produces a measurably different vector from the one BGE was trained to
produce — cosine 0.932, not 0.99 — and retrieves exactly as well.** That is the interesting
part. The model was optimised end to end for `[CLS]`, and reading it a different way costs
nothing at this task. Whatever the training put in the `[CLS]` position, it also put in the
average of the token positions.

**Max is the exception that shows the rule.** It is the one pooling that departs sharply from
the trained geometry, and it is the one that loses — 0.037, which is about half of BM25's
entire distance from BGE. A pooling can be swapped freely up to a point, and `max` is past it.

---

## Method

**Every pooling comes from the same forward pass.** The model runs once per corpus; all five
poolings read the same hidden states. Re-encoding per pooling would let batch composition and
non-determinism leak into the comparison.

**Pooled per batch, not stored.** 2,964 documents at 256 tokens and 384 dimensions is about
4.6 GB in float64, and none of it is needed once pooled.

**Padding is masked in every pooling, and it is tested.** A mean that averages over pad
positions scores a short document differently depending on what else was in its batch — a
property of the batching, not the pooling, that would show up as one.
`test_extra_padding_does_not_change_the_answer` pins it for all five.

**`max` uses a large negative sentinel, not zero.** With zeros, a dimension whose real values
are all negative would take 0 from a pad position, so an all-negative feature would read as 0
exactly when the batch happened to contain padding.

**BGE keeps its query prefix**, as its authors intend. Removing it would measure the prefix.

---

## Limitations

- **One encoder, and it is the one trained for `[CLS]`.** The design calls for a second
  encoder trained with **mean** pooling — `all-MiniLM-L6-v2`, also 384-dimensional, so the
  comparison could not be a width comparison. Its config downloaded; its weights are still
  at zero bytes behind a throttled connection. The run skips it **by name** rather than
  silently, and the table grows when the file lands.
- **That missing arm is the interesting one.** With one encoder, "the trained pooling wins"
  and "cls happens to be best" are indistinguishable. Two encoders with opposite trained
  poolings would separate them — and if the ranking flipped, it would show that *which
  pooling is best* has no answer independent of the encoder.
- **One task.** Retrieval by cosine similarity. A classification head on the same vectors
  could rank the poolings differently, and `max` in particular is a more natural fit for
  tasks that hinge on a single strong feature.
- **`idf_mean` is a stand-in for attention-weighted pooling**, not the thing itself. Real
  attention pooling learns its weights; this uses corpus statistics, which is the closest
  thing available without training anything.
- **No significance testing.** 0.945 against 0.938 on 300 queries is not established as a
  real difference. The 0.037 gap to `max` is large enough not to need it; the rest is not.

## Run it

```bash
python src/run.py            # 2,964 documents, ~75 seconds on a GPU
python src/run.py --quick    # 600 documents
pytest -q                    # 37 tests, no model, no dataset, no network
```

## Layout

```
src/pooling.py   the five poolings and the idf table - pure numpy, fully tested offline
src/run.py       the tables above
results/         pooling.json
```

## Keywords

pooling · mean pooling · CLS pooling · max pooling · last-token pooling · sentence
embeddings · bi-encoder · sentence-transformers · BGE · attention pooling · anisotropy ·
SEP token · masking · retrieval · HotpotQA
