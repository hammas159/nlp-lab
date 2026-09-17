<h1 align="center">10 · Pseudo-relevance feedback</h1>
<p align="center"><i>Every setting tested loses to the first stage it was meant to improve — and the per-query breakdown says exactly why.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#what-the-mean-hides">What the mean hides</a> &middot;
  <a href="#which-queries-it-hurts">Which queries</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-20%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
</p>

---

Run the query, assume the top few results are relevant, harvest terms from them, add those
terms to the query, run it again. RM3 (Lavrenko & Croft, 2001) is old, standard, and
reliably improves retrieval when reported as a mean over a query set.

**Nothing in the method checks whether the feedback documents are relevant.** That is the
whole mechanism and the whole risk. This project reports the per-query distribution instead
of the mean.

---

## The result

HotpotQA, 2,964 paragraphs, 300 questions — the same corpus projects 01, 02, 03 and 07 use.
BM25 first stage at **recall@10 = 0.865**.

| feedback docs | terms | α | recall@10 | Δ | p | |
|---:|---:|---:|---:|---:|---:|---|
| 10 | 50 | 0.5 | **0.852** | −0.013 | 0.130 | within noise |
| 10 | 10 | 0.3 | 0.842 | −0.023 | 0.003 | worse |
| 5 | 10 | 0.3 | 0.837 | −0.028 | 0.001 | worse |
| 5 | 20 | 0.5 | 0.828 | −0.037 | 0.000 | worse |
| 10 | 20 | 0.5 | 0.827 | −0.038 | 0.000 | worse |
| 20 | 20 | 0.5 | 0.823 | −0.042 | 0.000 | worse |
| 10 | 20 | 0.8 | 0.767 | −0.098 | 0.000 | worse |

**Not one setting beats the baseline.** Six of the seven are significantly worse under a
paired bootstrap over the 300 queries; the best is indistinguishable from doing nothing.

That is not the published behaviour of RM3, and the reason is in the next two tables.

---

## What the mean hides

| feedback docs | terms | α | improved | hurt | unchanged | worst drop |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 10 | 0.3 | 5 | 22 | 273 | −0.500 |
| 10 | 10 | 0.3 | 5 | 19 | 276 | −0.500 |
| **10** | **50** | **0.5** | **7** | **15** | **278** | **−0.500** |
| 10 | 20 | 0.5 | 4 | 27 | 269 | −0.500 |
| 20 | 20 | 0.5 | 6 | 31 | 263 | −0.500 |
| 10 | 20 | 0.8 | 7 | **65** | 228 | **−1.000** |

Even at the best setting, **more than twice as many queries get worse as get better** (15
against 7). The mean of −0.013 is not a small uniform effect; it is a small number of large
losses averaged against a smaller number of gains, with 93% of queries untouched.

At α = 0.8 — letting the expansion dominate the query — **65 queries degrade and at least one
loses every gold document it had.**

---

## Which queries it hurts

Splitting the 300 queries by how well the first stage did, at the best setting:

| First stage | n | mean change | hurt |
|---|---:|---:|---:|
| already perfect (recall 1.0) | 221 | **−0.034** | 15 |
| partly right (0 < r < 1) | 77 | **+0.039** | **0** |
| found nothing (recall 0.0) | 2 | **+0.250** | **0** |

**The effect is perfectly monotone in how much room the first stage left.** Feedback helps
every query it could possibly help and harms only queries that were already right — where
there was nothing to gain and a query to drift away from.

So pseudo-relevance feedback is a bet on the first stage being *mediocre*. On this benchmark
BM25 already answers 74% of queries perfectly, so the bet loses: the 221 queries with
nothing to gain outweigh the 79 with something.

That reframes the usual reporting. A mean gain on a collection where the first stage is weak
is a real result, but it does not transfer to a collection where the first stage is strong —
and the direction of the per-stratum effect, not the average, is what tells you which case
you are in.

---

## Method

RM3 in its interpolated form:

```
P(t | R) = Σ over feedback documents d of  P(t | d) · P(d | q)
q'       = (1 − α) · q  +  α · P(t | R)
```

**Feedback documents are weighted by their first-stage score**, normalised over the feedback
set, so a document the retriever was confident about contributes more terms. No softmax —
that would add a temperature, which is another knob nobody reports.

**Both sides are normalised before interpolation**, so α means the same thing regardless of
query length or how peaked the relevance model is. Without it, a two-word query and a
ten-word query get different amounts of expansion from the same α.

**BM25 is scored over a weighted bag of terms.** `shared.bm25.BM25.score` gives every query
term weight 1, which is correct for an unexpanded query and wrong for an expanded one — an
expansion term harvested with weight 0.01 would otherwise count as much as a term the user
typed. `test_uniform_weights_reproduce_plain_bm25` pins the weighted scorer to the original
at uniform weights, so every expanded result stays comparable with its baseline.

**A query whose first run matched nothing gets no expansion.** With all scores zero there is
no evidence to build a model from, and an arbitrary top-k would be feedback from documents
chosen by tie-breaking.

---

## Limitations

- **One collection, and an easy one.** HotpotQA questions are long and lexically rich, and
  BM25 already reaches 0.865. The result here is evidence that PRF does not help *when the
  first stage is strong* — it is not evidence that PRF does not work.
- **Recall@10 only.** PRF is often evaluated on MAP or precision at depth, where reordering
  within the top 10 counts and this metric is blind to it.
- **Seven settings is a coarse grid.** α, feedback depth and term budget interact; a finer
  sweep might find a setting that beats the baseline, though the per-stratum breakdown
  suggests the ceiling is low.
- **RM3 only.** No Rocchio, no term-selection by idf or by a χ²-style criterion, no
  re-ranking variants.
- **Two gold documents per query** means recall@10 moves in steps of 0.5, so "worst drop
  −0.500" is one gold document lost, not a continuous slide.
- **No relevance judgements beyond the gold two.** A retrieved document that is genuinely
  useful but not one of HotpotQA's two supporting paragraphs counts as a miss, which is
  harsher on expansion than on the baseline — expansion is what brings in related documents.

## Run it

```bash
python src/run.py            # the full sweep, 90 seconds
python src/run.py --quick    # one setting
pytest -q                    # 20 tests, no dataset, no network
```

## Layout

```
src/prf.py   weighted BM25 scoring, the relevance model, query expansion
src/run.py   the sweep, the per-query breakdown, and the stratified analysis
results/     feedback.json
```

## Keywords

pseudo-relevance feedback · PRF · RM3 · relevance model · Lavrenko Croft · query expansion ·
query drift · BM25 · information retrieval · recall@k · paired bootstrap · per-query
analysis · variance · HotpotQA
