<h1 align="center">13 · Language identification</h1>
<p align="center"><i>0.997 on a paragraph, 0.788 on ten characters. The number is reported on documents and the method is used on queries.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-ranking-flips-with-length">The ranking flips</a> &middot;
  <a href="#the-hard-pair-is-not-the-one-that-looks-hard">The hard pair</a> &middot;
  <a href="#problems-hit-while-building-this">Problems hit</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-26%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/runtime-34%20seconds-success" alt="runtime">
</p>

---

Language identification is described as solved, and on a paragraph it is. The accuracy
quoted is almost always measured on **documents**; the thing the method is used for is a
search query, a tweet, a log line, a code snippet — inputs an order of magnitude shorter.

Three classes, all from the local cache: **English prose** (HotpotQA, 10.3M characters),
**C source** (Devign, 5.4M) and **Python source** (MBPP + HumanEval, 198k). Profiles built
from 70% of each class; every test snippet cut from a document the profile never saw.

Three standard identifiers over the same profiles:

| | |
|---|---|
| `cavnar_trenkle` | Cavnar & Trenkle (1994) rank-order distance — order only, no probabilities |
| `naive_bayes` | multinomial Naive Bayes over character n-grams, in log space |
| `compression` | which class profile compresses the input best |

---

## The result

Accuracy against input length. Chance is 0.333.

| chars | `cavnar_trenkle` | `naive_bayes` | `compression` |
|---:|---:|---:|---:|
| **10** | **0.796** | 0.788 | 0.718 |
| 20 | 0.876 | 0.960 | 0.844 |
| 40 | 0.936 | 0.982 | 0.930 |
| 80 | 0.948 | 0.989 | 0.966 |
| 160 | 0.962 | 0.994 | 0.986 |
| 320 | 0.973 | 0.997 | 0.992 |
| **640** | 0.981 | **0.997** | 0.994 |

**0.997 at 640 characters, 0.788 at ten — a fall of 0.209 for a 64× shorter input.**

At ten characters, one in five short strings is assigned to the wrong language. That is not
a failure of the method; it is the method being asked a question the input cannot answer.
A ten-character window holds perhaps three n-grams the profile has ever seen.

The practical reading: an accuracy quoted for a language identifier is meaningless without
the length it was measured at, and the lengths people measure at are not the lengths people
use.

---

## The ranking flips with length

At 640 characters `naive_bayes` wins by 0.016 over `cavnar_trenkle`. **At ten characters
`cavnar_trenkle` wins** — 0.796 against 0.788.

The mechanism is in how each handles an unseen n-gram. Cavnar-Trenkle charges a fixed
penalty equal to the profile size; Naive Bayes charges `log(alpha / denominator)`, which on
a large profile is a much heavier penalty. With plenty of evidence that severity is an
advantage — it sharpens the separation. With three n-grams it is not, because almost
everything is unseen and the score becomes a count of absences rather than a reading of
presences.

So the oldest and simplest method here is the best one at the length that matters most, and
it would never be chosen on a document-length benchmark. **A method selected on long inputs
is not selected for short ones.**

`compression` is worst at ten characters (0.718) for the same reason amplified: it needs
enough input to find something to reference in the profile, and ten bytes appended to eight
kilobytes of context barely moves the compressed size.

---

## The hard pair is not the one that looks hard

The expectation going in — written into the code before the run — was that **C and Python**
would be the hard pair. They share braces, operators, English keywords and identifier
conventions that neither shares with prose.

Confusion matrix, `naive_bayes` at 40 characters, rows are truth:

| | c | english | python |
|---|---:|---:|---:|
| **c** | 0.99 | 0.00 | 0.00 |
| **english** | 0.00 | 1.00 | 0.00 |
| **python** | 0.01 | 0.03 | 0.96 |

**The two programming languages are essentially never confused with each other.** Both are
mistaken for English, and only in that direction — worst confusion is Python read as English
at 3.0%, against 1.3% for the C/Python pair.

Why, once measured: a forty-character window of source code is often *entirely* identifiers,
keywords and comment text, which is English. What separates C from Python is punctuation
density and indentation, and those survive truncation intact. What separates code from prose
is a vocabulary that a short enough window simply does not contain.

The run reads the worst confusion off the matrix rather than asserting which pair it will
be — because the first version of this project hardcoded the C/Python claim and would have
shipped a sentence the data contradicts.

---

## Problems hit while building this

**A run that should take a minute took over an hour.** `naive_bayes` needs the size of the
union of every profile's n-gram vocabulary for its smoothing denominator. That union was
being rebuilt **on every single classification** — millions of n-grams, 6,300 times. It
depends only on the profiles, so it is now computed once and memoised against their
identities. The full sweep went from unfinished after an hour to **34 seconds**.

Nothing about the output looked wrong. The results were correct; only the clock said
anything, and only because the first row of a seven-row table took longer than the whole run
should have.

**The C/Python claim was written before it was measured.** Both the module docstring and the
closing paragraph asserted that the two programming languages would be the hard pair. They
are not. Both now state the expectation and then the refutation, and the closing text is
derived from the confusion matrix at runtime rather than written by hand.

---

## Limitations

- **Three classes, and two of them are programming languages.** This is not a multilingual
  language identifier. No multilingual corpus is available offline here, so the register
  contrast — prose against two code languages — is what could be measured honestly rather
  than what would be most representative.
- **Python is the smallest class by two orders of magnitude** (198k characters against
  10.3M). Its profile is thinner, and its higher error rate is partly that rather than
  anything about Python.
- **Snippets are cut at random offsets**, so many begin and end mid-token. That is realistic
  for a log line and unrealistic for a search query, and it makes the short-input numbers a
  lower bound.
- **`compression` uses an 8 KB context** per class, truncated so every class gets the same
  amount. A larger context would help it and cost time proportionally.
- **The profile size (400 n-grams) and n-range (1–5) are Cavnar & Trenkle's**, applied
  unchanged to all three methods so they see identical input. Naive Bayes in particular would
  usually be given the full count table rather than a truncated ranking.
- **No confidence threshold.** Every method must answer. A real identifier would decline
  below some margin, and the interesting question at ten characters is what fraction it
  should decline — which is not measured here.

## Run it

```bash
python src/run.py            # the full sweep, 34 seconds
python src/run.py --quick    # fewer samples
pytest -q                    # 26 tests, no dataset, no network
```

## Layout

```
src/identify.py  n-grams, profiles, and the three decision rules
src/run.py       the length sweep and the confusion matrix
results/         language_id.json
```

## Keywords

language identification · language detection · character n-grams · Cavnar Trenkle ·
Naive Bayes · compression distance · text classification · short text · confusion matrix ·
held-out evaluation · profile-based classification
