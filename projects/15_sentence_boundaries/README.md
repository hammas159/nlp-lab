<h1 align="center">15 · Sentence boundaries</h1>
<p align="center"><i>Four splitters span 0.015 of F1. Two thirds of every one's errors are the same construction, and none of them fixes it.</i></p>

<p align="center">
  <a href="#what-the-reference-is-and-is-not">What the reference is</a> &middot;
  <a href="#the-result">Result</a> &middot;
  <a href="#the-one-construction-that-matters">The one construction</a> &middot;
  <a href="#problems-hit-while-building-this">Problems hit</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-31%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/runtime-22%20seconds-success" alt="runtime">
</p>

---

## What the reference is, and is not

HotpotQA ships each paragraph pre-split into sentences. That segmentation was produced by a
**tool, not a person**.

So this project measures **agreement with a reference segmentation, not accuracy.** Nothing
here can say which splitter is right, because nothing here knows. Reporting an F1 against it
as though it were a gold standard would be the exact error project 05 is about — a number
that looks like a measurement and is an artefact of how it was produced.

It is still worth measuring, and the honest framing makes a better question: sentence
splitting is treated as settled preprocessing, so **if reasonable implementations disagree,
the choice of splitter is an unreported parameter of every downstream result.**

---

## The result

66,581 paragraphs, 220,000-odd reference boundaries. Four splitters, each the previous one
plus a restriction:

| Splitter | precision | recall | F1 |
|---|---:|---:|---:|
| naive — split on any `.` `!` `?` | 0.905 | 0.982 | 0.942 |
| + supplied abbreviation list | 0.928 | 0.979 | 0.953 |
| + list *learned* from the corpus | 0.913 | 0.979 | 0.945 |
| + require a sentence-like next token | **0.940** | 0.974 | **0.957** |

**The spread is 0.015.** Four implementations that a methods section would describe
identically as "sentences were split", and the distance between the crudest and the most
careful is a point and a half.

**Learning the abbreviation list makes things worse than supplying one** (0.945 against
0.953). Punkt's idea — that an abbreviation is a token bound to its trailing period — found
only a handful of tokens on this corpus, because encyclopaedic prose uses few abbreviations
and uses them rarely. The unsupervised method needs a corpus with enough abbreviations to
learn from, which is not the same as a large corpus.

---

## The one construction that matters

False boundaries, by what is on either side of them:

| Construction | naive | + list | + learned | + conservative |
|---|---:|---:|---:|---:|
| **single initial** | **13,892** | **13,892** | **13,849** | **11,208** |
| known abbreviation | 5,404 | **0** | 3,571 | **0** |
| other | 1,088 | 1,088 | 1,004 | 1,058 |
| quote or bracket follows | 297 | 297 | 295 | 297 |
| lowercase follows | 118 | 118 | 118 | **0** |
| number before period | 59 | 59 | 59 | 59 |
| exclamation or question | 42 | 42 | 42 | 11 |
| **TOTAL** | **20,900** | 15,496 | 18,938 | **12,633** |

**Single initials are 66.5% of the naive splitter's errors, and every refinement fails on
them.** `J. R. R. Tolkien`, `Ed Wood Jr.`, `Harry S. Truman` — three boundaries invented per
name.

Why each fix misses:

- **The abbreviation list** cannot help: `J` is not an abbreviation of anything, it is a
  name. No list contains every letter of the alphabet, and one that did would refuse to
  split after any single-letter sentence.
- **The learned list** cannot help either, for the same reason and more weakly.
- **Requiring a capital next** cannot help, because the token after `J.` is `R.` — also
  capitalised. It removes 19% of them and no more.

The abbreviation list *does* completely solve the construction it was built for — 5,404
errors to zero. It is simply not the construction that dominates. **Every splitter here
spends its sophistication on the second-largest problem.**

---

## Problems hit while building this

**A guaranteed false positive on every paragraph.** The splitters emitted a boundary at the
end of the text; the reference never contains one, because it is a list of *n* pieces and
*n* pieces have *n*−1 joins. That was one spurious error per paragraph across the corpus,
and it cost about **fifteen points of precision** — the first run reported 0.905 as 0.715.

**It hid behind a second bug.** Those end-of-text errors were being filed under "quote or
bracket follows", because the test was `following[:1] in "\"'(["` and **the empty string is
a substring of every string**, so the slice form returns `True` when nothing follows. The
taxonomy said 5,556 quote errors and 668 number errors; the real figures are 297 and 59.

Neither bug raised an exception, and both produced a plausible table. What exposed them was
printing six actual disputed spans with their surrounding text — at which point every one of
them turned out to end at `|` with nothing after it. `test_the_end_of_the_text_is_not_a_boundary`
and `test_end_of_text_is_named_rather_than_misfiled` are the regression tests.

---

## Limitations

- **Agreement, not accuracy.** Stated above and worth repeating: the reference is a tool's
  output. A splitter that disagreed with it by being *right* would score worse here.
- **One register.** Wikipedia introductions. They are unusually full of names with initials,
  which is exactly the construction that dominates — a legal or clinical corpus would have a
  different leading error, and a news corpus fewer initials and more quotations.
- **The tolerance is one character.** HotpotQA's sentences carry a leading space, so a
  boundary placed before or after it is the same decision written differently, and scoring
  those as errors would measure whitespace convention. A larger tolerance would hide real
  disagreements.
- **Four splitters, none of them modern.** No Punkt proper, no spaCy, no neural segmenter.
  The point is the spread between reasonable simple choices and the shape of what defeats
  them, not a leaderboard.
- **The learned-abbreviation threshold (0.85) and minimum count (4) are parameters** that
  were set once and not tuned. Tuning them on this corpus would make the learned list look
  better than it deserves.
- **`other` is 1,088 errors** and is not broken down further. Some of those are genuine
  disagreements worth a category of their own.

## Run it

```bash
python src/run.py            # the full corpus, 22 seconds
python src/run.py --quick    # a tenth
pytest -q                    # 31 tests, no dataset, no network
```

## Layout

```
src/splitters.py  four splitters, the learned abbreviation list, the error taxonomy
src/run.py        agreement scoring and the breakdown above
results/          boundaries.json
```

## Keywords

sentence boundary detection · sentence segmentation · sentence splitting · Punkt ·
Kiss Strunk · abbreviation detection · tokenisation · preprocessing · precision recall ·
error taxonomy · unsupervised learning · HotpotQA · silver standard
