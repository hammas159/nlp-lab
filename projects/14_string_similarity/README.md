<h1 align="center">14 · String similarity</h1>
<p align="center"><i>Six of seven ranking positions change when you swap the error model. The ranking is a property of the corruption, not of the measures.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-measure-that-knows-the-error-model-does-not-win">The error-model metric</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#problems-hit-while-building-this">Problems hit</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-72%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/metrics-implemented%2C%20not%20imported-informational" alt="metrics">
</p>

---

The standard way to choose a string metric: corrupt a word list, see which measure recovers
the originals, report the ranking. The corruption gets a sentence in the methods section;
the ranking gets the table.

**The ranking belongs to the pair.** This project makes the error model an explicit variable
and measures what happens when it changes.

Seven measures in three families — edit distance (`levenshtein`, `damerau`, `keyboard`),
phonetic codes (`soundex`, `metaphone`), character overlap (`jaro_winkler`, `bigram`) — all
written out rather than imported. Two error models:

| | |
|---|---|
| **typing** | keyboard-adjacent substitution, adjacent transposition, deletion, doubling |
| **phonetic** | pronunciation-preserving rewrites — `ph`/`f`, `c`/`k`, `ie`/`y`, silent letters |

2,000 candidate words from HotpotQA, 600 corrupted words per model, one edit each. A measure
ranks the whole candidate list against each corruption; accuracy is how often the original
comes back first.

---

## The result

| Measure | typing | phonetic | gap |
|---|---:|---:|---:|
| `damerau` | **0.909** | 0.871 | +0.039 |
| `jaro_winkler` | 0.897 | 0.834 | +0.064 |
| `keyboard` | 0.827 | 0.811 | +0.015 |
| `levenshtein` | 0.807 | **0.871** | −0.064 |
| `bigram` | 0.655 | 0.691 | −0.036 |
| `metaphone` | 0.429 | 0.745 | **−0.316** |
| `soundex` | 0.195 | 0.235 | −0.040 |

**Best under typing noise: `damerau`. Best under phonetic noise: `levenshtein`.**

Ranked side by side:

| rank | under typing | under phonetic |
|---:|---|---|
| 1 | `damerau` | `levenshtein` |
| 2 | `jaro_winkler` | `damerau` |
| 3 | `keyboard` | `jaro_winkler` |
| 4 | `levenshtein` | `keyboard` |
| 5 | `bigram` | `metaphone` |
| 6 | `metaphone` | `bigram` |
| 7 | `soundex` | `soundex` ← same |

**Six of seven positions hold a different measure.** Only the worst one stays put.

`metaphone` shows the mechanism most plainly: **0.429 under typing, 0.745 under phonetic** —
a swing of 0.316 from changing nothing but how the words were corrupted. A phonetic code is
built to be blind to spelling variation that preserves sound, so it is nearly useless when
the corruption is a slipped finger and strong when the corruption is a misheard name. That
is not a defect. It is the measure doing exactly what it says, evaluated against a
distribution it was not designed for.

So a table of string metrics ranked on one corrupted dataset is a description of that
dataset. The useful question is never "which measure is best" but **"what does my noise
actually look like"**, and that is an empirical question about the data, not a choice from a
menu.

---

## The measure that knows the error model does not win

`keyboard` is Levenshtein with a substitution cost that depends on how far apart two keys
are: replacing `a` with `s` costs 0.4, replacing it with `p` costs 1.0. It is the only
measure here that encodes *how* text gets corrupted rather than *how much*.

The design expectation was that it would win under typing noise. **It ranks third of seven,
at 0.827 — below plain `damerau` at 0.909.**

The reason is visible once stated: making near-key substitutions cheap forgives the
corruption, and it equally forgives **every wrong candidate** that differs by a near-key
substitution. Against a list of two thousand words, the second effect is larger than the
first. A metric that encodes the error model buys tolerance and pays for it in
discrimination.

`damerau` wins instead for a narrower reason: it charges **one** operation for an adjacent
transposition where Levenshtein charges two. Transposition is one of the four corruptions in
the typing model, and that single rule is worth more than a whole cost matrix.

---

## Method

**Two rows, not a full matrix**, for Levenshtein — the recurrence only reads the previous
row, and this runs about eight million times per sweep.

**A tie is not a hit.** The phonetic measures return exactly 1 or 0, so they tie constantly;
counting a tie as correct would score them on the size of the equivalence class rather than
on whether they found the word. `run.py` requires a unique argmax.

**Words shorter than four characters are excluded.** A three-letter word corrupted once is
usually a different three-letter word, and the task stops being retrieval.

**The phonetic model only applies rules that fit.** A word with no applicable rewrite is
returned unchanged rather than corrupted by a rule that does not apply — so the two models
corrupt different *numbers* of words (5 of 600 unchanged under typing, 59 under phonetic),
and the run reports that rather than hiding it. Unchanged words are excluded from scoring.

---

## Problems hit while building this

**The phonetic model could return an empty string.** Several rewrites delete characters —
`gh` → nothing, trailing `e` → nothing. Applied repeatedly to a short word, `ee` became `e`
became `""`, and the early-return path skipped the non-empty guard at the end of the
function. An empty query scores equally against every candidate, which would have quietly
handed every measure the same wrong answer on a handful of words.

A rewrite that would empty the string is now not applicable at all.
`test_phonetic_never_returns_an_empty_string` is the regression test, and it is the test that
found it.

---

## Limitations

- **Synthetic corruption.** Neither model is a claim about real-world error rates. They are
  two clearly different distributions, and the finding is that the ranking does not survive
  swapping them — which does not require either to be realistic.
- **One edit per word.** Two or three edits would favour the overlap family, which degrades
  more gracefully than edit distance as corruption accumulates.
- **English, lowercased, alphabetic.** `soundex` and `metaphone` are built for English names
  in particular, and this vocabulary is ordinary English prose — it is not the population
  either was designed for, which is part of why both score poorly.
- **`metaphone` here is a reduced implementation** — the common digraph rules and vowel
  dropping, not the hundred context-sensitive cases of the original. It is labelled as such
  in the source.
- **Accuracy at rank 1 only.** A measure that reliably puts the original second would score
  zero here and be perfectly useful as a candidate generator feeding a reranker.
- **The candidate list is closed.** Every corrupted word's original is in the list, so this
  measures ranking and never the decision of whether a word is a typo at all.

## Run it

```bash
python src/run.py            # the full sweep, ~3 minutes
python src/run.py --quick    # fewer words
pytest -q                    # 72 tests, no dataset, no network
```

## Layout

```
src/metrics.py  seven measures: edit distance, phonetic codes, character overlap
src/corrupt.py  the two error models, stated explicitly
src/run.py      the sweep and the ranking comparison
results/        similarity.json
```

## Keywords

string similarity · edit distance · Levenshtein · Damerau · transposition · Jaro-Winkler ·
Soundex · Metaphone · phonetic matching · n-gram overlap · fuzzy matching · record linkage ·
spelling correction · error model · noisy channel · evaluation design
