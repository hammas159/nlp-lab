<h1 align="center">11 · Text clustering</h1>
<p align="center"><i>Silhouette picks k = 10 when the truth is 2, and the implementation everyone thinks they are running is the worse of the two.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#cosine-k-means-is-two-algorithms">Two algorithms</a> &middot;
  <a href="#how-much-does-the-seed-decide">The seed</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-39%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/k--means-implemented%2C%20not%20imported-informational" alt="kmeans">
</p>

---

1,500 Devign C functions, labelled by which codebase they came from — **qemu 63.2%, FFmpeg
36.8%**. The true number of clusters is 2, and provenance is exactly the kind of structure a
clustering of source code would be expected to find.

Two questions: does the internal metric used to choose *k* find it, and is "cosine k-means"
one algorithm?

---

## The result

Spherical k-means, mean of three seeds:

| k | silhouette | ARI (vs labels) | purity |
|---:|---:|---:|---:|
| **2** *(truth)* | 0.0152 | 0.3251 | 0.7789 |
| 3 | 0.0169 | **0.3356** | 0.8816 |
| 4 | 0.0163 | 0.2649 | 0.8720 |
| 5 | 0.0173 | 0.2676 | 0.9002 |
| 6 | 0.0180 | 0.1911 | 0.8833 |
| 8 | 0.0193 | 0.1281 | 0.8436 |
| **10** | **0.0218** | 0.1203 | 0.8620 |

**Silhouette is maximised at k = 10. Agreement with the labels is maximised at k = 3. The
truth is k = 2.** Silhouette rises monotonically across the whole range tested, so it has not
found a maximum — it would presumably keep going.

And **purity rises monotonically too**, 0.779 to 0.862. That is not a property of this
dataset; it is a property of purity, which reaches 1.0 when every point is its own cluster.
`test_purity_is_one_when_every_point_is_its_own_cluster` asserts it. Any k chosen by purity
is the largest k you were willing to try.

Note the silhouette values themselves: **0.015 to 0.022**. On a scale where 1 is perfectly
separated and 0 is no structure at all, this corpus has essentially none by that measure —
and the metric still has an argmax, and people still read it.

---

## Cosine k-means is two algorithms

Text is clustered with cosine similarity so that document length does not decide membership.
The standard way to get that from k-means is **spherical k-means**: normalise every vector,
assign by maximum dot product, and **re-normalise the centroids after each update**.

The last step is the one that gets skipped. On the unit sphere, minimising squared Euclidean
distance and maximising dot product give the same *assignment* — so normalising the input and
then calling ordinary k-means looks equivalent. It is not, because **the mean of a set of
unit vectors is not itself a unit vector.** Without re-normalisation the centroids drift
inward and the decision boundary moves.

Same input, same seeds, the only difference being that one line:

| k | ARI spherical | ARI euclidean | agreement between them |
|---:|---:|---:|---:|
| **2** | **0.3251** | **0.7542** | **0.2959** |
| 3 | 0.3356 | 0.4656 | 0.5069 |
| 4 | 0.2649 | 0.2841 | 0.4511 |
| 5 | 0.2676 | 0.3288 | 0.4626 |
| 6 | 0.1911 | 0.2346 | 0.4140 |
| 8 | 0.1281 | 0.1763 | 0.4232 |
| 10 | 0.1203 | 0.1694 | 0.4289 |

**At k = 2 the two partitions agree with each other only 0.296** — less than either agrees
with the labels. These are not two runs of one algorithm; they are two algorithms.

And the direction is the uncomfortable one. **The variant people run by accident scores
0.754 against the true labels; the one they meant to run scores 0.325.** More than twice as
good, at every k.

That is not a recommendation to skip the re-normalisation — it is one corpus, and the reason
is visible in the class sizes. The classes here are 63/37, and letting centroid norms vary
lets one centroid claim more of the space, which is what an uneven split needs. Spherical
k-means, by forcing every centroid onto the sphere, is biased towards clusters of similar
size. The point is that **"we used cosine k-means" does not identify what was run**, and the
difference is larger than most results being reported.

---

## How much does the seed decide

| k | ARI mean | ARI sd | silhouette sd |
|---:|---:|---:|---:|
| **2** | 0.3251 | **0.1562** | 0.0002 |
| 3 | 0.3356 | 0.0449 | 0.0006 |
| 4 | 0.2649 | 0.0726 | 0.0013 |
| 10 | 0.1203 | 0.0020 | 0.0004 |

At k = 2, **the standard deviation across three seeds is 0.156 against a mean of 0.325** —
nearly half the signal. A single-seed comparison between two clustering methods on this data
would be measuring the seed.

The silhouette standard deviation is two orders of magnitude smaller. **The metric used to
choose k is stable; the thing being chosen is not.** So a k selected by silhouette is
reproducible and the clustering it selects is not, which is the worst combination — it looks
settled and is not.

---

## Method

**k-means is implemented, not imported**, with k-means++ seeding. Uniform seeding on text
routinely picks two points from the same dense region and leaves a whole topic unseeded, so
run-to-run variance swamps whatever is being compared.

Both variants share everything except the centroid step, so the comparison is that one line
and nothing else. `test_spherical_centroids_are_unit_length` and
`test_euclidean_centroids_are_not_unit_length` pin the difference.

**Features are `ltc`** — log term frequency, idf, cosine normalisation — written out rather
than taken from a library default, because project 07 measures that choice as worth 0.4 of
recall on a retrieval task.

**Three metrics, deliberately.** Silhouette is the one used when there are no labels; ARI is
chance-corrected agreement with the labels; purity is included specifically to show that it
cannot be used for this. `test_putting_everything_in_one_cluster_scores_zero` checks that the
chance correction is doing its job.

---

## Limitations

- **One corpus and a two-class label.** Provenance in Devign is qemu against FFmpeg, which is
  a topical distinction as much as a stylistic one — the two codebases use different
  identifiers, so a bag-of-words clustering has an easy signal available.
- **The Euclidean result is not a recommendation.** It wins here, on a 63/37 split, for a
  reason (centroid norms can vary, so an uneven split is reachable) that will not hold on a
  balanced corpus.
- **Three seeds is a small sample** for a standard deviation. It is enough to show the seed
  matters and not enough to quantify it precisely.
- **Only k-means.** No agglomerative clustering, no HDBSCAN, no spectral methods — all of
  which would choose differently and none of which is tested here.
- **Silhouette on 4,530 dimensions.** Distance concentration in high dimensions is a known
  problem for every internal validity metric, and it is part of why the values are so small.
  Reducing dimensions first would change the numbers and add another undocumented choice.
- **The true k is an assumption.** Devign functions could reasonably cluster by subsystem,
  by author or by age rather than by project, and any of those would be a defensible "truth"
  that this scores as error.

## Run it

```bash
python src/run.py            # 1,500 functions, ~4 minutes
python src/run.py --quick    # 500
pytest -q                    # 39 tests, no dataset, no network
```

## Layout

```
src/cluster.py   spherical and Euclidean k-means, silhouette, adjusted Rand, purity
src/features.py  Devign loader and ltc TF-IDF vectors
src/run.py       the tables above
results/         clustering.json
```

## Keywords

text clustering · k-means · spherical k-means · cosine similarity · k-means++ · silhouette
coefficient · adjusted Rand index · purity · cluster validity · choosing k · internal
validation · seed variance · TF-IDF · Devign · unsupervised learning
