<h1 align="center">21 · Reranking depth</h1>
<p align="center"><i>Project 03's whole finding rests on the number 50. This sweeps it, and the bottleneck moves.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-bottleneck-flips">The bottleneck flips</a> &middot;
  <a href="#the-better-the-first-stage-the-shallower-the-window">Optimal depth</a> &middot;
  <a href="#what-survives-of-project-03">What survives of project 03</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-15%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/GPU-one%20pass%2C%20six%20depths-informational" alt="one pass">
  <img src="https://img.shields.io/badge/450%2C000-pairs%20scored-orange" alt="pairs">
</p>

---

[Project 03](../03_reranker_ceiling) asked whether a cross-encoder rescues a weak first stage
or only reorders it, and found that **every first stage converts about 96% of its recall@50
into recall@10**. It measured that at one depth: 50. Nothing in the project justified the
number, and the whole conclusion is a function of it.

This sweeps the depth from 10 to 500 over the same corpus, the same queries and the same
reranker.

---

## The result

HotpotQA · 2,964 documents · 300 queries · `cross-encoder/ms-marco-MiniLM-L-6-v2`.

| First stage | depth | ceiling (r@depth) | r@10 after | change | converted |
|---|---:|---:|---:|---:|---:|
| **BM25** (0.865 before) | 10 | 0.865 | 0.865 | **+0.000** | 100.0% |
| | 20 | 0.933 | 0.910 | +0.045 | 97.5% |
| | 50 | 0.967 | 0.922 | +0.057 | 95.3% |
| | **100** | 0.980 | **0.932** | **+0.067** | 95.1% |
| | 200 | 0.982 | 0.932 | +0.067 | 94.9% |
| | 500 | 0.990 | 0.932 | +0.067 | 94.1% |
| **TF-IDF** (0.842 before) | 10 | 0.842 | 0.842 | **+0.000** | 100.0% |
| | 20 | 0.923 | 0.907 | +0.065 | 98.2% |
| | 50 | 0.955 | 0.918 | +0.077 | 96.2% |
| | 100 | 0.977 | 0.928 | +0.087 | 95.1% |
| | **200** | 0.983 | **0.932** | **+0.090** | 94.7% |
| | 500 | 0.987 | 0.930 | +0.088 | 94.3% |
| **BGE-small** (0.943 before) | 10 | 0.943 | 0.943 | **+0.000** | 100.0% |
| | **20** | 0.967 | **0.947** | **+0.003** | 97.9% |
| | 50 | 0.978 | 0.942 | −0.002 | 96.3% |
| | 100 | 0.988 | 0.940 | −0.003 | 95.1% |
| | 200 | 0.992 | 0.937 | −0.007 | 94.5% |
| | 500 | 0.998 | 0.937 | −0.007 | 93.8% |

*(The depth-50 rows reproduce project 03 exactly — 0.922, 0.918, 0.942.)*

**Reranking at depth 10 changes nothing, provably.** The top 10 reranked is the top 10
reordered: the same set of documents, so recall@10 cannot move no matter what the
cross-encoder thinks. A reranking window has to exceed the evaluation cutoff to do anything
at all, and `test_reranking_at_the_evaluation_cutoff_cannot_change_recall` pins it.

---

## The bottleneck flips

Project 03 concluded:

> the first stage's job is **recall@50, not ranking.** Optimising a first-stage retriever
> for precision@10 is optimising the wrong metric when a reranker sits behind it.

That is true at depth 50. Follow the ceiling column past it and it stops being true.

At depth 500, BGE-small hands the reranker a candidate set containing **99.8%** of all gold
documents — very nearly everything there is — and the final answer is **0.937**. Six points
of reachable gold are sitting in the window, and the reranker does not bring them up.

| | ceiling | delivered | left behind |
|---|---:|---:|---:|
| BM25 @ 500 | 0.990 | 0.932 | 0.058 |
| TF-IDF @ 500 | 0.987 | 0.930 | 0.057 |
| BGE-small @ 500 | **0.998** | 0.937 | **0.061** |

**Past roughly depth 100, the first stage has stopped being the constraint and the reranker
has become it.** Fetching more does not help, because what is missing is not candidates but
the ability to rank them. The advice "optimise the first stage for recall@K" is correct at
K=50 and pointless at K=500 — you can already have essentially all the gold and still land
in the same place.

---

## The better the first stage, the shallower the window

| First stage | r@10 before | best depth | r@10 there | change |
|---|---:|---:|---:|---:|
| TF-IDF | 0.842 | **200** | 0.932 | +0.090 |
| BM25 | 0.865 | **100** | 0.932 | +0.067 |
| BGE-small | 0.943 | **20** | 0.947 | +0.003 |

The ordering is exact and it runs backwards to intuition: **the weakest first stage wants
the deepest window, and the strongest wants the shallowest.** TF-IDF needs ten times the
reranking compute of BGE to reach a worse result.

And BGE is **actively harmed** by depth. Beyond 20 every number is negative, reaching −0.007
at depth 200. Handing a good ranker's output to the cross-encoder in bulk makes it worse,
because each extra candidate is another chance to rank a distractor above a gold document
that BGE had already placed correctly.

This is the practical form of project 03's finding that its fixed depth could not show:
**a reranker is a repair for a weak first stage, and the repair has a dose.** If your
retriever is already good, the correct window is small, and "more candidates" is a
regression.

---

## What survives of project 03

Project 03's headline was that every first stage converts **about 96%** of its ceiling. Read
across the sweep, that number is a reading of a declining curve at one point:

| depth | BM25 | TF-IDF | BGE-small | **spread** |
|---:|---:|---:|---:|---:|
| 10 | 100.0% | 100.0% | 100.0% | 0.0% |
| 20 | 97.5% | 98.2% | 97.9% | 0.7% |
| **50** | **95.3%** | **96.2%** | **96.3%** | **0.9%** |
| 100 | 95.1% | 95.1% | 95.1% | 0.1% |
| 200 | 94.9% | 94.7% | 94.5% | 0.5% |
| 500 | 94.1% | 94.3% | 93.8% | 0.4% |

**The constant is not 96%. Conversion falls monotonically with depth**, for all three first
stages, from 100% down to about 94%. Project 03 reported a property of depth 50.

But the claim underneath it survives, and it is the more interesting one. **At every depth,
the three first stages convert within 0.9% of each other** — they agree far more closely
with each other than any of them agrees with itself across depths. So conversion is not a
constant; it is a function of depth alone, and **essentially independent of which retriever
fed it.** That is a stronger statement than the original, and it needed the sweep to say.

---

## Method

**One pass of the cross-encoder, six depths.** Each query's top 500 is scored once and every
depth reads a prefix. A cross-encoder scores pairs independently — the score of (query,
document) does not depend on what else is in the candidate set — so a prefix gives exactly
the ranking a fresh pass at that depth would, at a sixth of the cost. 450,000 pairs, about
four minutes per first stage on the GPU.

That property is not universal, and the code says so: a listwise reranker, or anything that
normalises scores over the candidate set, would need a fresh pass per depth, and reusing
scores would silently report numbers for an experiment that was never run.
`test_a_prefix_equals_a_fresh_pass_at_that_depth` pins the assumption.

**The tail is kept.** Reranking at depth *d* reorders the first *d* and leaves everything
below in first-stage order, because that is what a deployed reranker does. Truncating
instead would make recall@k undefined for k > d and would flatter every shallow depth by
deleting exactly the documents it failed to reach.

**Ties keep first-stage order**, so a measurement never picks up an arbitrary tiebreak
instead of the reranker's opinion.

---

## Limitations

- **One reranker.** MiniLM-L-6 is small and fast; a larger cross-encoder would convert more
  of the ceiling and might move the depth at which the bottleneck flips. The shape of the
  curve is the finding, not the exact crossover.
- **One evaluation cutoff.** Everything is recall@10. The depth-10 row is zero *because*
  the cutoff is 10; at recall@20 the zero row would be depth 20.
- **Cost is described, not tabulated.** Reranking is linear in depth, so depth 500 costs 25×
  depth 20 — but the seconds here are one machine's GPU and would mislead as a benchmark.
  Project 01 reports index time with the same caveat.
- **300 queries.** Enough to separate 0.842 from 0.943, not enough to call −0.002 a real
  regression. The BGE decline is monotone across four depths, which is stronger evidence
  than any single pair, but no significance test was run.
- **HotpotQA questions have exactly two gold documents**, so recall@10 saturates easily.
  A task with more relevant documents per query would push the useful depth higher.

## Run it

```bash
python src/run.py            # 300 queries, depths 10-500, ~13 minutes on a GPU
python src/run.py --quick    # 60 queries, depths to 100
pytest -q                    # 15 tests, no model, no dataset, no network
```

## Layout

```
src/depth.py   prefix reranking and the metrics - no model, fully tested offline
src/run.py     the sweep
results/       depth.json
```

## Keywords

reranking · cross-encoder · retrieval depth · candidate window · recall ceiling ·
two-stage retrieval · first-stage retrieval · BM25 · TF-IDF · BGE · MiniLM · HotpotQA ·
monoBERT · precision-recall tradeoff · ablation
