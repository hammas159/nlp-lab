<h1 align="center">16 · Stylometry</h1>
<p align="center"><i>Authorship attribution on code scores 0.965. Remove the identifiers and 68% of it goes with them — it was topic.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#eighteen-features-with-no-words-in-them">Eighteen features</a> &middot;
  <a href="#burrows-delta-collapses-on-a-full-vocabulary">Delta collapses</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-46%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/Burrows%20Delta-implemented%2C%20not%20imported-informational" alt="delta">
</p>

---

Stylometry claims a writer can be identified from *style* — habits of function words,
punctuation and rhythm — independently of subject. On source code the equivalent claim is
that a codebase has a recognisable house style.

The difficulty is that an identifier carries both at once. `av_frame_alloc` is a naming
**convention** (lower_snake, project prefix) and also a **topic** (video frames). A
classifier handed raw tokens can use either, and reporting its accuracy as "style" assumes
it used the first.

4,000 Devign C functions labelled by codebase — **qemu 64%, FFmpeg 36%.** Four views strip
topic progressively while the classifier stays fixed, so the drop between rows is the
contribution of what was removed.

---

## The result

| View | what survives | features | Naive Bayes | Burrows' Delta |
|---|---|---:|---:|---:|
| **lexical** | everything, identifiers included | 16,285 | **0.965** | 0.659 |
| **masked** | shape only — `if ( ID ) { ID = ID ( ID , NUM ) ; }` | 60 | 0.760 | 0.767 |
| **structural** | keywords, operators, punctuation | 57 | 0.756 | 0.751 |
| **layout** | line lengths, indents, braces — **no tokens at all** | 18 | **0.767** | 0.732 |
| *majority class* | — | — | *0.659* | *0.659* |

**Removing identifiers costs 0.209 of accuracy. Of everything the lexical view had above
the floor, the structural view keeps 32%.**

So roughly **two thirds of the "authorship" signal was topic**, not style. A paper reporting
0.965 for code authorship on raw tokens has measured, mostly, that qemu functions are about
virtualisation and FFmpeg functions are about codecs.

The remaining third is real. Every style-only view sits 0.09–0.11 above the majority floor,
and that gap does not come from subject matter — it cannot, because the subject matter has
been deleted.

---

## Eighteen features with no words in them

The `layout` view contains **no source content whatsoever**. No identifiers, no keywords,
no operators. Only: which length bin each line falls in, which indent bin, whether a line is
blank, a comment, a preprocessor directive, whether it ends in a semicolon, whether the
brace sits on its own line or trails, whether the file uses tabs.

Eighteen features. It scores **0.767** — **the best of the three style-only views**, ahead
of the 57 structural tokens.

That is the strongest form of the stylometric claim available here: a codebase is
identifiable from **how its code is set on the page**, with every word removed. Brace
convention and indent width are house style in the most literal sense, and they survive
deletion of everything else.

It also says something about the other views. If 18 typographic features beat 57 grammatical
ones, then what `structural` was measuring was substantially formatting too — reached
indirectly, through which punctuation appears and how often.

---

## Burrows' Delta collapses on a full vocabulary

Look down the Delta column. On the lexical view it scores **0.659 — exactly the majority
floor.** It has stopped classifying and is answering "qemu" every time.

On all three small views it works, and on two of them it **beats** Naive Bayes.

The mechanism is the method's defining step. Delta z-scores every feature across the corpus
and compares documents by mean absolute difference in z units. That is deliberate: it puts a
habit appearing in a tenth of a percent of tokens on equal footing with one appearing in
five percent, which is exactly right if style lives in small consistent preferences.

With 16,285 features, almost all of them near-zero and sparse, that same equality is fatal.
Every rare identifier gets full weight, the distance is dominated by thousands of noisy
dimensions, and the centroids become indistinguishable.

**Delta is not a general-purpose classifier that happens to be used in stylometry. It is a
method for a few hundred curated function words, and it degenerates outside that regime** —
silently, to the floor, with no error. `test_delta_weights_a_rare_feature_as_heavily_as_a_common_one`
pins the property that causes it.

Naive Bayes shows the opposite profile: excellent with 16,285 features, unremarkable with
18. The two attributors are not better and worse; they are suited to different feature
counts, and reporting one without the other hides that.

---

## Method

**Four views, one classifier.** The comparison is only meaningful if nothing changes between
rows except what was removed, so both attributors see every view and the split is fixed.

**Masked keeps identifier *density* while destroying identifier *content*.** A function with
five identifiers still shows five `ID` tokens, so naming frequency remains measurable while
naming vocabulary does not. `test_masked_preserves_identifier_density` asserts it.

**Layout is binned, not continuous.** Line lengths and indent widths are bucketed so the
same multinomial classifier can read them. Switching classifier between views would make
the comparison a comparison of classifiers.

**The floor is 0.659, not 0.5.** Devign's provenance split is 64/36, so a model that always
answers qemu scores 0.659. Every number here is read against that — which is what reveals
the Delta collapse, since 0.659 looks like a result until you notice it *is* the floor.

**Frequencies, not counts, for Delta.** A 400-line function and a 20-line one must be
comparable; Delta is defined over relative frequencies.

---

## Limitations

- **Provenance is not authorship.** qemu and FFmpeg are codebases with many contributors
  each. What is being attributed is a project's conventions — enforced partly by linters and
  review — not an individual's hand.
- **Two classes only**, and imbalanced. Real authorship attribution is many-way, where the
  problem is much harder and the floor much lower.
- **Topic and project are confounded by construction.** These two codebases genuinely do
  different things. A pair of projects in the *same* domain would separate style from topic
  far more sharply, and the lexical advantage would shrink — this design can show that topic
  contributes, not exactly how much it would contribute elsewhere.
- **`masked` still leaks a little.** Identifier *lengths* and *positions* survive masking as
  structure, and both correlate weakly with naming convention.
- **No feature selection for Delta.** Stylometry normally hands Delta the top *n* function
  words. Giving it a full vocabulary is the realistic misuse being demonstrated, not the
  method's intended configuration — the small-view rows show it working properly.
- **One split, one seed.** Project 11 measured how much a seed can move a result on this
  same corpus; the numbers here carry the same caveat and were not repeated across seeds.

## Run it

```bash
python src/run.py            # 4,000 functions, ~32 seconds
python src/run.py --quick    # 800
pytest -q                    # 46 tests, no dataset, no network
```

## Layout

```
src/views.py      the four views: lexical, masked, structural, layout
src/attribute.py  Burrows' Delta and multinomial Naive Bayes, both written out
src/run.py        the table above
results/          stylometry.json
```

## Keywords

stylometry · authorship attribution · Burrows' Delta · function words · z-score ·
Naive Bayes · code provenance · topic confound · feature ablation · Devign · qemu · FFmpeg ·
house style · formatting · code fingerprinting
