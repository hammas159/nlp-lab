<h1 align="center">08 · gzip-kNN</h1>
<p align="center"><i>Compression-based classification, scored as a classifier rather than as a metric that needs the answer.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#why-k--2-and-only-k--2">Why k = 2</a> &middot;
  <a href="#does-the-compressor-matter">The compressor</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/compressors-stdlib%20only-informational" alt="stdlib">
</p>

---

`gzip` plus k-nearest-neighbours, with no training and no parameters, was reported to beat
BERT on low-resource text classification. The method is real and genuinely elegant:
normalised compression distance, from Cilibrasi & Vitányi (2005),

```
NCD(x, y) = ( C(xy) − min(C(x), C(y)) ) / max(C(x), C(y))
```

If `y` resembles `x`, compressing them together costs little more than compressing `x`
alone, because the compressor reuses what it already learned. No tokenizer, no vocabulary,
no features.

What this project measures is not the distance. It is **the step that turns k neighbours
into a prediction**, which is where the published number came from.

---

## The result

1,000 Devign functions as the reference set, 500 as the evaluation set, one gzip distance
matrix, three ways of reading it:

| k | tie rate | `oracle_tie` (published) | `nearest_tie` | `random_tie` |
|---:|---:|---:|---:|---:|
| 1 | 0.000 | 0.574 | 0.574 | 0.574 |
| **2** | **0.454** | **0.804** | **0.574** | 0.588 |
| 3 | 0.000 | 0.594 | 0.594 | 0.594 |
| 5 | 0.000 | 0.594 | 0.594 | 0.594 |
| 11 | 0.000 | 0.584 | 0.584 | 0.584 |

**At k = 2 the published rule reports 0.804. Scored as a classifier the same neighbours give
0.574** — a difference of 0.230, decided entirely on the 45.4% of documents where the two
neighbours disagree and the true label breaks the tie.

## Why k = 2, and only k = 2

Look down the tie-rate column. **It is zero at every k except 2.**

With two classes, an odd number of neighbours always has a majority — three neighbours are
2–1, five are 3–2, eleven are 6–5. A tie needs an even k, and at k = 2 a tie means the two
neighbours simply disagree, which on this task happens almost half the time.

So the rule is inert at k = 1, inert at k = 3, 5 and 11, and decides 45% of the test set at
k = 2. **The published configuration is the single choice of k at which the tie-breaking
rule does anything at all.** Every other row of that table is a perfectly ordinary kNN
result in the high 0.57s.

## Against baselines that were actually configured

| | accuracy | cost |
|---|---:|---|
| **best gzip-kNN (any k, honest rule)** | **0.594** | **360 s** of compression |
| nearest centroid (TF-IDF) | 0.566 | 0.9 s to train and predict |
| multinomial Naive Bayes | 0.560 | 0.8 s to train and predict |
| **majority class** | **0.528** | free |

Compression distance does win — by 2.8 points over a TF-IDF centroid — and it wins for 400
times the compute, because NCD needs one compression per (reference, evaluation) pair and a
trained model needs none at prediction time.

It is also **6.6 points above always guessing the majority class.** The published-style
number, 0.804, sits 27.6 points above that floor, which is the difference between "a
parameter-free method is competitive here" and "a parameter-free method beats BERT".

## Does the compressor matter

Measured on a smaller 150 × 80 subsample, because lzma at its default preset cannot afford
the full matrix:

| compressor | k = 2 | best k | best accuracy | ms per pair |
|---|---:|---:|---:|---:|
| gzip | 0.600 | 1 | 0.600 | **0.23** |
| bz2 | 0.500 | 5 | **0.625** | 0.63 |
| lzma | 0.575 | 1 | 0.575 | **20.17** |

**lzma costs 88 times gzip per pair and is less accurate.** A better compressor is a better
model of the data in the compression sense, and that does not translate into a better
similarity measure — NCD divides by `max(C(x), C(y))`, so a compressor that shrinks
everything also shrinks the differences it is being asked to detect.

bz2 is the most accurate of the three at its best k, by 2.5 points, at three times gzip's
cost. None of the three separates from the others by more than the majority-class floor is
from chance.

---

## Method

The published implementation used **k = 2**. When the two neighbours carried different
labels there was no majority, and the tie was resolved by checking whether the true label
was among them — scoring the prediction correct if it was.

That is not a classifier. It cannot be deployed, because at prediction time there is no
answer to consult. It is a legitimate metric — roughly top-k accuracy, "is the right label
anywhere in the shortlist" — but it is not the quantity it was compared against.

Three rules are implemented against **one distance matrix**, so nothing between them can be
sampling noise:

| Rule | Ties resolved by |
|---|---|
| `oracle_tie` | the true label, when it is among the tied ones — the published behaviour |
| `nearest_tie` | the closest neighbour among the tied labels — deterministic |
| `random_tie` | a seeded coin — the floor, and what an implementation does with no rule |

Where the k neighbours have a clear majority, all three agree. **The gap between them is
bounded by the tie rate**, which is reported alongside every row.

`predict` raises rather than running when `oracle_tie` is asked for without the labels. A
rule that cannot execute without the answer should say so.

**The task is Devign** — binary vulnerability detection over 2,732 C functions per split.
Compression distance needs no tokenizer, so it can be pointed at source code without the
preprocessing argument that dominates comparisons on prose; and a **binary** task at
**k = 2** is exactly the setting where the tie-breaking rule decides the most cases. The
`validation` split is the reference set and `test` the evaluation set: Devign's `train`
split has never been obtainable here, the same 17.85 MB download that blocked
`devign-leakage` and projects 04 and 05.

**The baselines were actually configured.** Multinomial Naive Bayes and a TF-IDF nearest
centroid, both written out in `src/baselines.py` rather than imported, because a comparison
against a method nobody set up is not a comparison. Both train in under a second — which is
itself part of the argument, since the headline claim for compression-based classification
is that it needs no training.

---

## Limitations

- **One dataset, and a hard one.** Devign is a binary task where strong learned models
  reach the high 60s. Nothing here says what compression distance does on the topic
  classification tasks it was originally reported on, where classes are lexically far
  apart and a nearest-neighbour method has much more to work with.
- **Subsampled.** NCD needs one compression per (reference, evaluation) pair, so the full
  2,732 × 2,732 matrix is 7.5 million compressions. The reported matrix is smaller and its
  size is stated with every number.
- **The compressor comparison uses a smaller subsample still**, because at default settings
  lzma costs about ten times gzip per pair and the full matrix would take hours.
- **`oracle_tie` is a reconstruction.** It reproduces the behaviour the critique describes
  rather than running the original code, so it is a demonstration of what that rule does to
  a score, not a re-execution of the original experiment.
- **Devign contains near-duplicates.** Project 04 found 30 near-duplicate pairs in the test
  split, 20 of them carrying conflicting labels. A nearest-neighbour method is precisely
  the kind that benefits from duplicates and is confused by conflicting ones; that
  interaction is not separated here.
- **No BERT.** The comparison in the original paper was against learned neural models. The
  baselines here are classical and cheap, which makes them a floor rather than a ceiling.

## Run it

```bash
python src/run.py            # the full study
python src/run.py --quick    # a smaller matrix
pytest -q                    # tests, no dataset, no network
```

## Layout

```
src/ncd.py         compressed lengths, normalised compression distance, gzip's window
src/knn.py         the three tie-breaking rules, the tie rate, the majority floor
src/data.py        Devign from the local Hugging Face cache
src/baselines.py   multinomial Naive Bayes and a TF-IDF nearest centroid, from scratch
src/run.py         the tables below
results/           classification.json
```

## Keywords

gzip · compression · normalised compression distance · NCD · Kolmogorov complexity ·
parameter-free classification · k-nearest neighbours · tie-breaking · top-k accuracy ·
evaluation bug · text classification · Devign · vulnerability detection · Naive Bayes ·
nearest centroid · baselines · reproduction
