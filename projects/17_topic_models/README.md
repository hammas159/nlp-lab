<h1 align="center">17 · Classical topic models</h1>
<p align="center"><i>Coherence and the task disagree at every setting tested, and the disagreement is systematic</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-disagreement-is-not-noise">Not noise</a> &middot;
  <a href="#the-input-representation-is-worth-as-much-as-the-model">The input representation</a> &middot;
  <a href="#an-expectation-that-did-not-survive">A refuted expectation</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-36%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/NPMI-implemented%2C%20not%20imported-informational" alt="npmi">
</p>

---

Topic models are compared on **coherence** — how often a topic's top words co-occur. It is
intrinsic, it is what papers report, and NPMI is the variant Lau, Newman & Baldwin (EACL
2014) found tracks human judgement best.

Chang et al. (NIPS 2009) had already shown that the *other* standard measure, held-out
likelihood, can run **opposite** to human interpretability. This asks the same question of
coherence against a task: fit LSA, NMF and LDA on one corpus at matched *k*, score each by
coherence and by handing its document vectors to a retriever, and see whether the two
rankings ever coincide.

---

## The result

HotpotQA · 2,964 documents · 10,930 terms (min_df 2, max_df 0.5, stopwords removed) ·
300 queries. **BM25 on the same corpus and queries scores 0.865**, which is the line every
row below is read against.

| k | model | coherence | recall@10 |
|---:|---|---:|---:|
| 10 | LSA | 0.078 | **0.168** |
| 10 | **NMF** | **0.297** | 0.080 |
| 10 | LDA | 0.115 | 0.083 |
| 50 | LSA | −0.233 | **0.442** |
| 50 | **NMF** | **0.332** | 0.233 |
| 50 | LDA | −0.044 | 0.205 |
| 200 | LSA | −0.447 | **0.678** |
| 200 | **NMF** | **0.301** | 0.338 |
| 200 | LDA | −0.075 | 0.257 |

*(k = 20 and 100 omitted for width; both follow the same pattern and are in `results/`.)*

**Coherence picks NMF at 5 of 5 settings. The task picks LSA at 5 of 5 settings. They never
once agree.** Across all fifteen fits, coherence and recall@10 correlate at **r = −0.687** —
not uninformative, but pointing the wrong way.

---

## The disagreement is not noise

The cleanest evidence is inside a single model, where nothing varies but *k*:

| k | LSA coherence | LSA recall@10 |
|---:|---:|---:|
| 10 | 0.078 | 0.168 |
| 20 | −0.014 | 0.290 |
| 50 | −0.233 | 0.442 |
| 100 | −0.378 | 0.575 |
| 200 | **−0.447** | **0.678** |

**LSA's coherence falls monotonically while its usefulness rises monotonically.** The
configuration with the worst topics by the standard metric is the one that does the job
best, and the relationship is not noisy — every step moves both numbers in opposite
directions.

The mechanism is visible in the topics themselves. At k=50:

```
NMF  (coherence  0.332)   film directed stars starring written produced american drama
                          basketball season men tournament ncaa team conference division
                          album released studio records cosby songs single release singer

LSA  (coherence -0.233)   s film american album series season song released football band
                          film album song released band directed music written single studio
                          film series directed television drama films stars starring festival
```

NMF's topics are genuinely better *as topics* — crisp, separable, exactly what you would
print in a paper. LSA's are redundant and overlapping: three variations on entertainment,
sharing most of their words.

And LSA retrieves nearly twice as well. Its components are not trying to be readable topics;
they are orthogonal directions that reconstruct the matrix, and overlap between them is not
a defect but how a basis spans a space. Coherence penalises exactly the property that makes
the representation work.

**So coherence is not a broken metric. It is a faithful measure of something else** — how
much a topic looks like a list a person would write — and that is only the goal when a
human is going to read the topics. If the model feeds a downstream system, coherence is
selecting against the thing you want.

---

## The input representation is worth as much as the model

LDA is defined over **counts**: it models how many times a word was drawn. LSA and NMF are
conventionally given **TF-IDF**. Feeding LDA TF-IDF is a common tutorial shortcut, and it is
not what the model says. At k=50:

| Model | representation | coherence | recall@10 |
|---|---|---:|---:|
| LSA | **TF-IDF** *(as defined)* | −0.233 | **0.442** |
| LSA | counts | −0.101 | 0.212 |
| NMF | **TF-IDF** *(as defined)* | 0.332 | 0.233 |
| NMF | counts | 0.209 | 0.158 |
| LDA | TF-IDF | −0.348 | 0.118 |
| LDA | **counts** *(as defined)* | −0.044 | **0.205** |

**Giving LDA TF-IDF costs 0.087 recall and 0.304 coherence.** It is a one-word change in a
constructor and it is worse than choosing a different model.

More strikingly: swapping LSA's representation moves recall by **0.230**, while the gap
between the best and worst *model* at this k is **0.237**. The knob nobody reports is worth
as much as the choice everybody argues about.

---

## An expectation that did not survive

Stopwords co-occur in nearly every document, so a topic made of function words should have
near-perfect co-occurrence statistics and coherence should reward it. The prediction was
that removing stopwords would lower coherence across the board.

| Model | filtered | stopwords left in | change |
|---|---:|---:|---:|
| LSA | −0.233 | −0.145 | **+0.088** |
| NMF | 0.332 | 0.248 | **−0.084** |
| LDA | −0.044 | 0.030 | **+0.074** |

**It moves in both directions, by about the same amount.** The prediction was wrong, and
recording it is more useful than quietly deleting the section.

Why it fails is visible in the NPMI definition. A word pair appearing in *every* document
gives P(a,b) = P(a)P(b) = 1, so the ratio is 1, its logarithm is 0 — and so is the
denominator. NPMI does not reward a perfectly ubiquitous pair with 1.0; it is a genuine 0/0,
and the implementation floors it to 0, meaning "no information". Stopwords are therefore not
free coherence. `test_a_word_pair_in_every_document_scores_zero_rather_than_one` pins that
boundary, which is what turned a plausible story into a measured non-effect.

---

## Method

**One interface, three models.** LSA, NMF and LDA expose the same `fit` / `transform` /
`top_words`, so nothing differs between rows but the algorithm and the *k*.

**Coherence is scored in-corpus.** Papers often score against Wikipedia, which measures
whether topics match general English rather than whether they describe this collection.
In-corpus is the harder reading and is stated rather than buried.

**Co-occurrence is boolean per document.** A word appearing nine times in a paragraph counts
once, or coherence rewards repetition instead of association.

**Undefined pairs take the floor, not exclusion.** A pair that never co-occurs scores −1
rather than being dropped — dropping it would raise a bad topic's average by deleting its
worst evidence.

**LDA runs batch with 100 iterations and NMF to 2,000.** Both were under-converged at
sklearn's defaults, and a comparison where one model is stopped early is measuring the
iteration budget.

---

## Limitations

- **Retrieval is one task, and it flatters LSA.** LSA is a *retrieval* method by ancestry —
  latent semantic indexing — so scoring topic models by retrieval is arguably scoring them
  on LSA's home ground. The finding that coherence and utility diverge would be stronger
  with a second, unrelated task; classification on the same corpus is the obvious next one.
- **Every topic model loses badly to BM25.** The best here is 0.678 against BM25's 0.865.
  None of this is an argument for using topic models to retrieve; the retrieval score is
  being used as a *yardstick* for representation quality, not as a recommendation.
- **One seed.** LDA is the only stochastic model of the three, and its numbers are from
  `random_state=0`. Project 11 measured how far a seed can move a result on this corpus;
  that caveat applies here and was not repeated across seeds.
- **NPMI is one coherence measure.** Röder et al. (2015) catalogue many; C_v is the other
  common choice, and it is not implemented here.
- **LDA does not improve smoothly with k.** Its recall runs 0.083, 0.107, 0.205, 0.180,
  0.257 — a dip at k=100 that the other two do not have. With one seed and a fixed iteration
  budget, that is as likely to be under-convergence at high k as a property of the model,
  and this design cannot tell the two apart.

## Run it

```bash
python src/run.py            # k in {10, 20, 50, 100, 200}, ~11 minutes
python src/run.py --quick    # k in {10, 50}
pytest -q                    # 36 tests, no dataset, no network
```

## Layout

```
src/models.py      LSA, NMF and LDA behind one interface
src/coherence.py   NPMI, implemented rather than imported
src/run.py         the tables above
results/           topic_models.json
```

## Keywords

topic models · LSA · LSI · latent semantic analysis · LDA · latent Dirichlet allocation ·
NMF · non-negative matrix factorisation · truncated SVD · topic coherence · NPMI ·
intrinsic vs extrinsic evaluation · TF-IDF · HotpotQA · Blei · Lee & Seung · Deerwester ·
Chang · Lau
