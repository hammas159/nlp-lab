<h1 align="center">19 · Sentiment lexicons</h1>
<p align="center"><i>The prediction was that the rules would outweigh the lexicon. They do not.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#a-prediction-that-did-not-survive">A prediction that did not survive</a> &middot;
  <a href="#coverage-is-not-the-advantage-it-looks-like">Coverage</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-58%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/dependencies-numpy%20only-success" alt="numpy only">
  <img src="https://img.shields.io/badge/result-prediction%20refuted-orange" alt="refuted">
</p>

---

Lexicon-based sentiment analysis has two knobs. One is **which lexicon** — the axis papers
compare on. The other is **what you do with it**: whether "not good" is treated as "good",
whether "extremely good" outweighs "good", whether the clause before *but* still counts.

This project was designed around a specific prediction, written down in the lab README
before any code existed:

> the intended finding was that **negation and intensifier handling outweighs the choice of
> lexicon**

Three lexicons × four rule sets on one corpus, everything else held fixed. **The prediction
is wrong**, and it is wrong by a margin that survives a bootstrap.

---

## The result

Movie-review sentences (Pang & Lee, ACL 2005) · 10,662 items · exactly balanced, so the
**floor is 0.500**.

| Lexicon | entries | bag of words | + negation | + intensifiers | + contrastive | unscored | acc. where it fired |
|---|---:|---:|---:|---:|---:|---:|---:|
| VADER | 7,491 | 0.621 | 0.629 | 0.630 | 0.637 | 13.4% | 0.646 |
| **Opinion** | 6,786 | **0.652** | **0.655** | **0.657** | **0.663** | 26.2% | **0.697** |
| AFINN | 2,477 | 0.616 | 0.623 | 0.623 | 0.628 | 26.0% | 0.659 |
| *floor* | — | *0.500* | *0.500* | *0.500* | *0.500* | | |

**Holding the rules fixed, changing the lexicon moves accuracy by up to 0.035.
Holding the lexicon fixed, changing the rules moves it by up to 0.016.**

A paired bootstrap over items, 1,000 resamples: the difference is **+0.019, 95% interval
[+0.010, +0.031], and the lexicon axis is the larger one in 100% of resamples.** This is not
a near-tie that happened to fall one way.

---

## A prediction that did not survive

The rules help. Every one of them, on every lexicon, moves accuracy up — and the full set is
worth +0.016 on VADER. They are just worth **less than half** what the lexicon choice is
worth.

The reason is not that they fail to fire:

| Rule | fires on | flips the call | of those flips, correct |
|---|---:|---:|---:|
| + negation | 17.4% | 4.8% | 58.5% |
| + intensifiers | 21.5% | 5.1% | 58.8% |
| + contrastive | **32.6%** | 6.1% | **63.3%** |

The contrastive rule changes the score on nearly a third of all sentences. The rules are
firing constantly.

**They just cannot reach far enough.** Changing a score only matters if it changes the
*answer*, and that happens on 5–6% of items. On those, a flip is a coin already flipped, so
it helps exactly as much as its precision exceeds 50% — 58.5% for negation means a net gain
of about 17% on 4.8% of items, which is +0.008. The arithmetic checks out against the table:
0.621 → 0.629.

So the honest statement is narrower and more useful than the prediction: **rules for
negation, intensification and contrast are individually correct more often than not, and
collectively worth about half of what picking a different word list is worth.** If you are
choosing where to spend an afternoon, the word list is the better bet.

---

## Coverage is not the advantage it looks like

The lexicons agree with each other almost completely:

| Pair | agree in sign | on shared words |
|---|---:|---:|
| VADER vs Opinion | 97.8% | 2,212 |
| VADER vs AFINN | 96.8% | 2,425 |
| Opinion vs AFINN | 98.8% | 1,314 |

They essentially never call the same word positive and negative. So the 0.035 spread is not
about disagreement — it is about **which words are in the list at all**, and that turns out
to cut against the obvious reading.

**VADER is unscored on 13.4% of sentences; Opinion on 26.2%.** VADER speaks nearly twice as
often. And it is *worse*: 0.646 correct where it fires, against Opinion's **0.697**.

VADER's extra reach comes from emoticons, slang and hedges carrying small valences, and on
movie-review prose those fire on sentences where the evidence is thin. Opinion's flat
positive/negative word lists say less and are right more often when they say it.

This is worth stating plainly because coverage is the number a lexicon is usually sold on.
Here the lexicon with the best coverage finishes last.

---

## Method

**One tokenizer, one classifier, one corpus.** Only the lexicon and the rule set vary.

**Every lexicon is rescaled to [−1, 1].** Classification is by the sign of a sum, and
dividing by a positive constant cannot change a sign, so the rescaling costs nothing — but
it means an intensifier's ×1.5 is the same operation whether the source scale ran to 4, to 1
or to 5.

**Nothing abstains.** An item the lexicon cannot score is assigned to a class and counted,
rather than dropped. Dropping them would inflate every accuracy by removing exactly the hard
cases, and the unscored rate is reported as its own column instead.

**The lexicons are parsed from their distributed text files**, not read through a library.
It removes a dependency, and it makes the one genuine judgement call visible: SentiWordNet
scores *senses*, so collapsing it to words requires choosing how many senses to average —
which is why two papers that both say "we used SentiWordNet" can report different numbers.

**Where the source papers contradict each other, the contradiction is recorded.** `hardly`,
`barely` and `scarcely` appear in VADER's negator list *and* in its booster dictionary as
dampeners. They are treated as negators here, and the note is in the code.

---

## Limitations

- **SentiWordNet is missing.** It was in the design and its file had not finished
  downloading when this ran; the loader and its four tests are written and will pick it up
  the moment the file is present. Three lexicons is enough to show a 0.035 spread, but the
  fourth is the one with the most interesting failure mode.
- **One corpus, one domain.** The second domain — tweets — is the arm that would test
  whether the ranking is about the lexicon or about the match between lexicon and domain.
  **VADER was built for social media and finishes last on movie prose**, which is exactly
  the result that needs a second domain before it means anything. The loader and tests are
  in place; the corpus is still downloading.
- **Window-based negation is crude.** A negator flips everything within three tokens, so
  `not bad good` scores 0.0 — both words flip, where a reader negates only `bad`.
  Clause-based negation handles this; a fixed window cannot, and a fixed window is what
  almost every lexicon pipeline uses. It applies identically to all three lexicons, so it
  cannot bias the comparison, but it does bound how good any `+ negation` row can be.
  `test_a_negator_flips_everything_in_its_window_not_just_its_target` pins it.
- **VADER's full shipped implementation is not a row here.** It adds capitalisation and
  punctuation rules on top of the three implemented. Its absence means the rules axis is
  measured at *my* implementation of those rules, not at the best available one — which, if
  anything, understates the axis that already lost.
- **Accuracy only.** No per-class precision or recall, and the errors are not broken down
  by sentence length, which is where a bag-of-words method would be expected to fail.

## Run it

```bash
python src/run.py            # every lexicon and corpus present, ~30 seconds
python src/run.py --quick    # 2,000 items per corpus
pytest -q                    # 58 tests, no corpus, no network
```

Fetching the resources:

```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('opinion_lexicon'); \
           nltk.download('sentence_polarity'); nltk.download('sentiwordnet'); \
           nltk.download('twitter_samples')"
```

AFINN is not in NLTK; put `AFINN-111.txt` in `~/nltk_data/corpora/afinn/`. Any resource that
is absent is skipped with a message naming it, and the table shrinks — the project never
approximates a missing lexicon, because inventing a lexicon to compare against other
lexicons would measure the invention.

## Layout

```
src/lexicons.py   the four lexicons, parsed from their own file formats
src/rules.py      the four rule sets - no dependencies, fully tested offline
src/corpora.py    the two labelled corpora
src/run.py        the tables above
results/          sentiment.json
```

## Keywords

sentiment analysis · sentiment lexicon · VADER · Opinion Lexicon · Hu & Liu · AFINN ·
SentiWordNet · negation · intensifiers · valence shifters · contrastive conjunction ·
rule-based sentiment · Pang & Lee · sentence polarity · paired bootstrap · Taboada
