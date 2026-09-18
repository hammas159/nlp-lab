<h1 align="center">22 · Pooling</h1>
<p align="center"><i>Each encoder's trained pooling wins, and the two encoders rank the five poolings differently. "Best pooling" has no answer on its own.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-ranking-flips">The ranking flips</a> &middot;
  <a href="#two-of-the-five-are-not-different-poolings-on-bge">Not five poolings</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-37%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/cosine(cls%2C%20last)-1.000000-critical" alt="identical">
  <img src="https://img.shields.io/badge/encoders-2%20with%20opposite%20training-informational" alt="two encoders">
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

HotpotQA · 2,964 documents · 300 queries. Two encoders, both 384-dimensional so a difference
cannot be a difference in vector width, with **opposite trained poolings**. BM25 on the same
corpus scores 0.865.

| Pooling | **BGE-small** (trained CLS) | **MiniLM-L6** (trained mean) |
|---|---:|---:|
| cls | **0.945** *(as trained)* | 0.743 |
| mean | 0.945 | **0.833** *(as trained)* |
| last | 0.945 | 0.748 |
| idf_mean | 0.938 | 0.828 |
| max | 0.908 | 0.667 |
| **spread** | **0.037** | **0.167** |

**Each encoder's own trained pooling wins.** Both rows marked *as trained* are the top score
for their column — which is the result the single-encoder version of this project could not
distinguish from "CLS happens to be best".

---

## The ranking flips

```
BGE-small    cls  >  mean  >  last  >  idf_mean  >  max
MiniLM-L6    mean >  idf_mean >  last  >  cls   >  max
```

**The two encoders rank the five poolings differently.** `cls` is first for one and fourth
for the other; `mean` is the reverse. The only thing they agree on is that `max` is worst.

So **"which pooling is best" has no answer independent of the encoder.** It is decided by
what the model was trained with, not by a property of the pooling operation. A paper
reporting "mean pooling outperforms CLS" has reported a fact about its checkpoint.

The cost is asymmetric, which is the practically useful part:

| | reading it the other way costs |
|---|---:|
| BGE (trained CLS), read with mean | **0.000** |
| MiniLM (trained mean), read with CLS | **0.090** |

BGE tolerates being read as a mean; MiniLM does not tolerate being read as a CLS. That makes
sense in one direction only — a model trained to put everything in `[CLS]` still has that
content available in the average of its tokens, while a model trained on the average has no
reason to have put anything in `[CLS]` at all.

**And the spread itself is a fact about the encoder**: 0.037 on BGE against **0.167** on
MiniLM. The single-encoder version of this project would have concluded "pooling barely
matters" — a conclusion that is 4.5 times wrong on the very next model.

---

## Two of the five are not different poolings, on BGE

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

So on BGE the five-way comparison is really a **three-way** one: `{cls, last}`,
`{mean, idf_mean}`, and `max`.

**This does not happen on MiniLM.** There `cls` and `last` sit at cosine **0.463**, not
1.000 — genuinely different vectors that score 0.743 and 0.748. The `[SEP]`-equals-`[CLS]`
collapse is a property of *BGE's contrastive training*, not of the BERT architecture, and
one encoder alone could not have told those apart:

| | cls vs last | mean vs idf_mean | max vs mean |
|---|---:|---:|---:|
| BGE-small | **1.000** | 0.996 | 0.537 |
| MiniLM-L6 | **0.463** | 0.970 | 0.219 |

---

## What that leaves

Once the BGE duplicates are collapsed, the finding that survives on both encoders is about
`max`:

**`max` is the only pooling both encoders agree is worst**, and the only one that departs
sharply from the trained geometry on both — cosine 0.537 to BGE's mean, 0.219 to MiniLM's. It
loses 0.037 on one and 0.167 on the other. Taking the largest value per dimension discards
the rest of the sequence, and no amount of training makes that a good idea for retrieval.

Everything else is contingent on the checkpoint.

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

- **Two encoders is two.** They have opposite trained poolings, which is what the argument
  needs, but "the trained pooling wins" is established on a sample of two checkpoints. A
  third trained with `max` would be the sharp test, and no such sentence encoder is common
  enough to be worth the download.
- **Both are small English models of similar size.** A larger encoder, or a multilingual
  one, could behave differently — in particular the `[SEP]`-equals-`[CLS]` collapse may be
  more or less common than one example suggests.
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
python src/run.py            # 2,964 documents, both encoders, ~40 seconds on a GPU
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
