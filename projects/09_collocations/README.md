<h1 align="center">09 · Collocations</h1>
<p align="center"><i>Five association measures over one set of bigram counts. Four of the five top-20 lists share not a single pair.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-cutoff-is-the-model">The cutoff is the model</a> &middot;
  <a href="#is-chi-squared-even-admissible">Is chi-squared admissible</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-40%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/runtime-15%20seconds-success" alt="runtime">
</p>

---

A collocation is a word pair that co-occurs more than chance allows. Turning that into a
ranked list needs a statistic, and the choice is usually made by whichever one the toolkit
defaults to.

66,581 HotpotQA paragraphs, 5,929,439 tokens, **1,690,372 distinct bigrams of which 73.1%
occur exactly once.** That last number is the whole problem: the modal bigram is a hapax,
and the measures disagree most violently about what to do with it.

---

## The result

Top 20 pairs by each measure, from identical counts and no frequency cutoff:

| Measure | median frequency of its top 20 | what it picks |
|---|---:|---|
| `pmi` | **1** | publica ianuensis, ommegang ommegeddon, ryudo uzaki |
| `ppmi` | **1** | identical to PMI — clipping the floor cannot reorder |
| `t_score` | **10,561** | of the, is a, in the, it was |
| `llr` | **7,589** | is a, of the, united states, in the |
| `chi2` | **5** | iwo jima, djimon hounsou, hynden walch |

Jaccard overlap between the top-20 lists:

| | pmi | ppmi | t_score | llr | chi2 |
|---|---:|---:|---:|---:|---:|
| **pmi** | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **ppmi** | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| **t_score** | 0.00 | 0.00 | 1.00 | **0.48** | 0.00 |
| **llr** | 0.00 | 0.00 | 0.48 | 1.00 | 0.00 |
| **chi2** | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |

**Every off-diagonal entry is zero except one.** Only the two frequency-driven measures,
`t_score` and `llr`, agree at all, and they agree on under half. PMI and chi-squared share
nothing with anything, including each other — and the two of them are the measures most
often described interchangeably as "statistical association".

PMI's behaviour is not a defect in the implementation, it is the definition. PMI is a
**ratio**: `test_pmi_does_not_grow_with_evidence` asserts that ten times the counts in the
same proportions gives the identical score. A pair seen once, whose two words appear
nowhere else, is maximally surprising and carries no evidence whatever — and PMI cannot
tell those apart.

---

## The cutoff is the model

Every toolkit exposes a minimum-frequency cutoff. It is set without comment in every
tutorial. Here is what it does, measured as the overlap of each measure's top-20 with *its
own* list at the previous cutoff:

| min count | bigrams kept | pmi | ppmi | t_score | llr | chi2 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 455,554 | **0.00** | 0.00 | 1.00 | 1.00 | 0.33 |
| 5 | 142,814 | **0.00** | 0.00 | 1.00 | 1.00 | 0.38 |
| 10 | 66,964 | **0.00** | 0.00 | 1.00 | 1.00 | 0.08 |
| 25 | 25,223 | **0.00** | 0.00 | 1.00 | 0.82 | 0.33 |
| 50 | 11,536 | **0.03** | 0.03 | 1.00 | 0.90 | 0.38 |

**PMI's top-20 is completely replaced by every single change of the cutoff.** Not reordered
— replaced, with an overlap of zero, four times in a row. The median frequency of its top 20
tracks the cutoff exactly: 1, 2, 5, 11, 29, 60. PMI is not ranking the corpus; it is
returning whatever sits at the threshold it was handed.

`t_score` is perfectly stable at 1.00 throughout, for the opposite reason: it never looked
at rare pairs, so removing them changes nothing.

So the honest description of a PMI collocation list is **"the rarest pairs above the cutoff
you chose"**, and the cutoff is doing more work than the statistic. Reporting one without
the other is reporting half the method.

---

## Is chi-squared even admissible

Pearson's chi-squared needs every expected cell at five or more for its normal
approximation to hold. On this corpus:

| | share of bigram types failing the condition |
|---|---:|
| no cutoff | **96.1%** |
| min count 5 | 71.7% |
| min count 25 | 40.6% |
| min count 50 | **30.5%** |

**At no usable cutoff does chi-squared become admissible on most of its own input.** Dunning
(1993) wrote the log-likelihood ratio for precisely this reason, and the two measures do in
fact disagree here — not marginally, but to a Jaccard overlap of 0.00 on the top 20.

The chi-squared column is therefore not merely a different opinion from the log-likelihood
column. On 96% of the rows it is a statistic being read outside the range where it means
anything.

---

## Method

All five measures are computed from one 2×2 contingency table per bigram, so the
disagreement is a property of the statistics rather than of different inputs:

```
                 w2 present        w2 absent
w1 present       a = #(w1, w2)     b = #(w1, ·) − a
w1 absent        c = #(·, w2) − a  d = N − a − b − c
```

**Marginals come from first and second positions separately.** In `a b a`, the token `a`
appears twice but only once as the first element of a bigram. Using one unigram count for
both marginals is wrong at the edges of the stream and invisible in the output — the tables
still sum correctly. `test_marginals_use_first_and_second_positions_separately` pins it.

**Bigrams are counted within documents, never across them.** Concatenating the corpus into
one stream invents a bigram at every document boundary from two words that never appeared
together — 66,581 of them, every one a hapax pair, which is exactly the population PMI ranks
highest. Both the guard and the counterfactual are asserted, so the guard cannot be removed
silently.

**The cutoff filters what is scored, not what it is scored against.** Marginals are
properties of the whole corpus; recomputing them after filtering would make every score
depend on the cutoff twice.

---

## Limitations

- **One corpus, one register.** Encyclopaedic English prose. The hapax share drives the
  whole result, and it differs by register — code and speech transcripts would not give
  these proportions.
- **Adjacent bigrams only.** No window, no gaps, no syntactic filtering. Real collocation
  extraction usually applies a part-of-speech pattern first, which removes most of what
  `t_score` ranks highest before any statistic is applied.
- **Top-20 is a small sample of each ranking.** The overlaps would rise at top-1000; the
  claim is about the part of the list anyone reads.
- **No human judgement of what is actually a collocation.** This measures disagreement
  between the measures, not which of them is right. "of the" is high-frequency and not a
  collocation in any useful sense; "iwo jima" is. Nothing here scores that.
- **`t_score` is not a t statistic** in any defensible sense — the denominator uses the
  observed count as its own variance estimate. It is included because it appears in the
  corpus-linguistics literature under that name and behaves unlike the others.
- **The chi-squared admissibility rule is a convention**, not a theorem. Five is the number
  textbooks give; the approximation degrades continuously rather than failing at a
  threshold.

## Run it

```bash
python src/run.py            # every table above, 15 seconds
python src/run.py --quick    # a tenth of the corpus
pytest -q                    # 40 tests, no dataset, no network
```

## Layout

```
src/association.py  PMI, PPMI, t-score, log-likelihood ratio, chi-squared, admissibility
src/bigrams.py      contingency tables from a token stream, and the frequency cutoff
src/run.py          the tables above
results/            collocations.json
```

## Keywords

collocations · pointwise mutual information · PMI · PPMI · log-likelihood ratio · Dunning ·
chi-squared · t-score · association measures · contingency table · bigrams · hapax
legomena · frequency threshold · corpus linguistics · n-grams · statistical significance
