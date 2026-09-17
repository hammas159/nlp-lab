<h1 align="center">19 · Sentiment lexicons</h1>
<p align="center"><i>The prediction was that the rules would outweigh the lexicon. In both domains, they do not.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#a-prediction-that-did-not-survive">A prediction that did not survive</a> &middot;
  <a href="#coverage-is-not-the-advantage-it-looks-like">Coverage</a> &middot;
  <a href="#the-best-lexicon-depends-on-the-domain">Domain</a> &middot;
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

Four lexicons × four rule sets × two domains, everything else held fixed. **The prediction
is wrong**, in both domains, by a margin that survives a bootstrap in 100% of resamples.

---

## The result

Two corpora, both exactly balanced, so the **floor is 0.500** in each.

**Movie-review sentences** (Pang & Lee, ACL 2005) · 10,662 items

| Lexicon | entries | bag of words | + negation | + intensifiers | + contrastive | abstains | acc. where it fired |
|---|---:|---:|---:|---:|---:|---:|---:|
| VADER | 7,491 | 0.621 | 0.629 | 0.630 | 0.637 | 13.4% | 0.627 |
| **Opinion** | 6,786 | **0.652** | **0.655** | **0.657** | **0.663** | 26.2% | **0.689** |
| SentiWordNet | 21,479 | 0.600 | 0.607 | 0.609 | 0.614 | **1.2%** | 0.600 |
| AFINN | 2,477 | 0.616 | 0.623 | 0.623 | 0.628 | 26.0% | 0.643 |

**Tweets** · 10,000 items

| Lexicon | bag of words | + negation | + intensifiers | + contrastive | abstains | acc. where it fired |
|---|---:|---:|---:|---:|---:|---:|
| **VADER** | 0.636 | 0.645 | 0.644 | **0.646** | 29.8% | 0.679 |
| Opinion | 0.633 | 0.639 | 0.638 | 0.638 | 49.8% | **0.740** |
| SentiWordNet | 0.592 | 0.604 | 0.604 | 0.605 | **9.8%** | 0.606 |
| AFINN | 0.631 | 0.637 | 0.637 | 0.638 | 37.9% | 0.692 |

| | lexicon axis | rules axis | bootstrap difference | lexicon axis larger in |
|---|---:|---:|---|---:|
| movie sentences | **0.052** | 0.016 | **+0.036** [+0.024, +0.046] | **100%** of resamples |
| tweets | **0.044** | 0.013 | **+0.031** [+0.021, +0.043] | **100%** of resamples |

**Both domains agree, and neither is close.**

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

The contrastive rule changes the score on nearly a third of all movie sentences. The rules
are firing constantly. (On tweets they fire about half as often — 8.9% for negation against
17.4% — because tweets are shorter, which is itself a reason the rules axis is smaller there:
0.013 against 0.016.)

**They just cannot reach far enough.** Changing a score only matters if it changes the
*answer*, and that happens on 5–6% of items. On those, a flip is a coin already flipped, so
it helps exactly as much as its precision exceeds 50% — 58.5% for negation means a net gain
of about 17% on 4.8% of items, which is +0.008. The arithmetic checks out against the table:
0.621 → 0.629.

So the honest statement is narrower and more useful than the prediction: **rules for
negation, intensification and contrast are individually correct more often than not, and
collectively worth about a third of what picking a different word list is worth.** If you
are choosing where to spend an afternoon, the word list is the better bet.

---

## Coverage is not the advantage it looks like

Three of the four lexicons agree with each other almost completely. SentiWordNet does not:

| Pair | agree in sign | on shared words |
|---|---:|---:|
| Opinion vs AFINN | **98.8%** | 1,314 |
| VADER vs Opinion | 97.8% | 2,212 |
| VADER vs AFINN | 96.8% | 2,425 |
| SentiWordNet vs AFINN | **81.7%** | 1,554 |
| VADER vs SentiWordNet | **78.4%** | 3,072 |
| Opinion vs SentiWordNet | **77.0%** | 4,666 |

The three hand-built lexicons essentially never call the same word positive and negative.
SentiWordNet disagrees with all of them about **one word in five** — because it is not
hand-built. It is derived from WordNet *senses*, and collapsing a word's senses into one
score means a word whose dominant sense is positive and whose third sense is negative can
come out either way.

So the spread is about which words are listed and how they were scored. And that cuts
directly against the number lexicons are usually sold on:

| | abstains on | correct where it fired |
|---|---:|---:|
| **movie sentences** | | |
| SentiWordNet | **1.2%** | **0.600** |
| VADER | 13.4% | 0.627 |
| AFINN | 26.0% | 0.643 |
| Opinion | **26.2%** | **0.689** |
| **tweets** | | |
| SentiWordNet | **9.8%** | **0.606** |
| VADER | 29.8% | 0.679 |
| AFINN | 37.9% | 0.692 |
| Opinion | **49.8%** | **0.740** |

**In both domains, ordering the lexicons by coverage orders them exactly backwards by
precision — at every step, all four lexicons.** SentiWordNet scores almost every sentence
and is the least accurate on the ones it scores; Opinion abstains on half of all tweets and
is right three quarters of the time when it speaks.

The mechanism is that breadth is bought with weak entries. SentiWordNet reaches 21,479 words
by scoring senses that are only faintly evaluative, and those fire on sentences carrying no
real sentiment. Coverage is not free — it is traded against precision, and neither table
above shows that trade paying off.

---

## The best lexicon depends on the domain

| Corpus | best lexicon | accuracy |
|---|---|---:|
| movie sentences | **Opinion** | 0.663 |
| tweets | **VADER** | 0.646 |

This was the point of running a second domain, and it lands the way the lexicons' origins
predict. **VADER was built for social media**, finishes last of the three hand-built
lexicons on movie prose, and wins on tweets. Hu & Liu's Opinion Lexicon was built from
product reviews and wins on review-like sentences.

It also means the first table's "Opinion is best" is not a fact about Opinion. It is a fact
about Opinion *and movie reviews*, and the lexicon axis being the larger one does not imply
there is a best lexicon to pick — only that the pick matters more than the rules do.

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

- **SentiWordNet's word scores are a choice, not a reading.** It scores senses; this
  averages the first three by rank. Taking only the first sense, or all of them, gives a
  different lexicon with the same name — which is why its numbers vary between papers that
  all say "we used SentiWordNet". Its 77–82% agreement with the hand-built lexicons is
  partly this decision rather than the resource.
- **Two domains, both English and both short-text.** Tweets and review sentences differ
  enough to flip which lexicon wins, which is the point — but neither says anything about
  long documents, where a bag of words has far more evidence to work with.
- **Window-based negation is crude.** A negator flips everything within three tokens, so
  `not bad good` scores 0.0 — both words flip, where a reader negates only `bad`.
  Clause-based negation handles this; a fixed window cannot, and a fixed window is what
  almost every lexicon pipeline uses. It applies identically to all four lexicons, so it
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
python src/run.py            # 4 lexicons x 4 rule sets x 2 domains, ~10 seconds
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
