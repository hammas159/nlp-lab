<h1 align="center">20 · Word sense disambiguation</h1>
<p align="center"><i>Five of five methods beat random. Zero of five beat the most frequent sense.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-baseline-you-report-decides-the-story">Which baseline</a> &middot;
  <a href="#the-oracle-that-was-nearly-a-result">The oracle</a> &middot;
  <a href="#how-circular-is-first-sense">How circular is 'first sense'?</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-52%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/dependencies-numpy%20only-success" alt="numpy only">
  <img src="https://img.shields.io/badge/leakage-found%20and%20fixed-orange" alt="leakage">
</p>

---

Word sense disambiguation is frequently reported against a **random** baseline. That is the
flattering comparison. The baseline that matters is the **most frequent sense** — always
answer the sense WordNet lists first — because it requires no training, no glosses and no
context, and it is very hard to beat.

SemCor, 352 documents of human sense-tagged text, split by document. Five ways to pick a
sense, scored against both baselines.

---

## The result

Polysemous tokens only, 30% of documents held out.

| Method | accuracy | vs random | vs first sense |
|---|---:|---:|---:|
| random | 0.277 | — | −0.350 |
| **first sense (WordNet order)** | **0.628** | **+0.350** | — |
| trained MFS | 0.566 | +0.289 | −0.061 |
| context overlap + discourse | 0.563 | +0.286 | −0.064 |
| context overlap (supervised) | 0.559 | +0.282 | −0.069 |
| **Lesk (gloss overlap)** | **0.499** | **+0.222** | **−0.128** |
| *one sense per discourse (ORACLE)* | *0.675* | *+0.397* | *+0.047* |

**Against random, 5 of 5 methods win. Against the first sense, 0 of 5 win.**

The same six numbers support "every method works" or "nothing works", depending only on
which row you print underneath them.

---

## The baseline you report decides the story

The most frequent sense is **+0.350 above random before any method has done anything at
all**. That single number is larger than every improvement any method here achieved over
anything.

So a paper reporting "our system reaches 0.56, against a random baseline of 0.28" is
reporting a doubling — and the system is losing to a one-line heuristic that needed no
training data, no sense glosses and no context window.

**Lesk is the sharpest case.** It is the classic knowledge-based method and the one most
often shown against random, where it looks like a **+0.222** win — an 80% relative
improvement, from a method that needs only a dictionary. Against the most frequent sense it
is **0.128 behind, the worst of the five.** The same system, the same run, two defensible
baselines, and opposite conclusions.

This is not a claim that the methods are bad. A supervised context model at 0.559 is doing
real work: it is discriminating senses from surrounding words, which random is not. It is a
claim about **what the number is compared against**, and the comparison is not a detail —
it is the entire content of the sentence "this method works".

---

## The oracle that was nearly a result

The first version of this project had a method called `one sense per discourse`, scoring
**0.675** — the only thing to beat the most frequent sense, and by a comfortable margin. It
implemented Gale, Church & Yarowsky's (1992) observation that a word keeps one sense within
a document: look at the lemma's other occurrences in the same document and take the majority.

It was reading the **gold senses** of those other occurrences.

Excluding the token's own vote — which it did — is not enough. The other senses in a test
document are human annotations from the evaluation set, so the method was answering with the
answer key. It is kept in the table, renamed, in italics, as an **oracle**: a ceiling, not
something anyone can run.

The honest version predicts every token with the context model, then forces each lemma in
each document to its own majority *prediction*. No gold label is consulted. It scores
**0.563** — below the baseline, and **0.112 below the oracle.**

That gap is the actual finding about the heuristic: one-sense-per-discourse is a strong
regularity, and nearly all of its apparent value comes from already knowing the sense you
are propagating. `test_the_honest_discourse_method_never_reads_a_gold_sense` pins the
difference by changing the test labels and asserting the predictions do not move.

---

## How circular is 'first sense'?

WordNet orders senses by frequency, and **that ordering was derived from a sense-tagged
corpus — SemCor.** Scoring "always sense 1" on SemCor is therefore not quite the
knowledge-free baseline it appears to be.

Estimating the most frequent sense from the training split instead gives **0.566**, against
WordNet's **0.628** — a difference of **−0.061**.

So the inherited ordering is worth about six points over one estimated from 70% of this
corpus. They are close because they are nearly the same statistic computed twice, on
overlapping evidence. The honest reading is that `first sense` is a *very good* baseline
that carries some corpus knowledge with it, and reporting it as though it required nothing
understates it.

---

## Method

**Split by document, never by token.** Sentences from one document must not appear on both
sides, or a context method sees its own test distribution and the discourse heuristic is
handed most of its answer.

**Only polysemous tokens are scored.** 31.2% of tagged test tokens have a single observed
sense; every method gets them right, so including them would add the same constant to all
six rows and compress exactly the differences the evaluation exists to show.

**Nothing abstains.** Every method returns a sense for every token, so accuracies are
directly comparable and nobody can improve by declining the hard cases.

**SemCor is parsed directly from its SGML.** The files are not well-formed XML, and the
attribute that matters — `wnsn` — is the WordNet sense *rank*, which is what makes the
first-sense baseline computable with no WordNet installation at all.

**Multi-sense tags take the first.** 703 of 226,005 tagged tokens carry more than one sense
tag, where the annotator allowed both readings; the count is reported rather than hidden.

---

## Limitations

- **The sense inventory is the one observed in SemCor, not WordNet's.** A sense that never
  occurs in the corpus is invisible here, so the inventory is a lower bound on polysemy —
  which makes the **random baseline stronger than it should be**. The real random baseline,
  over full WordNet inventories, would be lower, and the gap this project reports would be
  *wider*. The finding is therefore conservative.
- **Lesk here is simple Lesk**, overlapping a sense's own gloss with the sentence. Banerjee
  & Pedersen's extended version also pools the glosses of related synsets — hypernyms,
  hyponyms, meronyms — which gives it far more words to match on and scores materially
  better in the literature. The 0.499 is a floor for the gloss-overlap family, not a ceiling,
  and the published extended results still do not clear MFS.
- **The supervised model is deliberately the cheapest one.** IDF-weighted context bags, no
  embeddings, no sequence model. It is here to show what clearing MFS costs, not to be a
  competitive system — a modern supervised model would clear it.
- **One split, one seed.** Project 11 measured how far a seed can move a result; the same
  caveat applies and was not repeated here.
- **Accuracy over all polysemous tokens** weights frequent words heavily. A per-lemma macro
  average would tell a different and also true story about rare words.

## Run it

```bash
python src/run.py            # all of SemCor, ~8 seconds
python src/run.py --quick    # 60 files
pytest -q                    # 52 tests, no corpus, no network
```

```bash
python -c "import nltk; nltk.download('semcor'); nltk.download('wordnet')"
```

`nltk` fetches the corpora and is never imported by this project. **WordNet is optional**:
without it every row above still runs and Lesk is left out of the table by name, rather than
degrading to its fallback and reporting that number under Lesk's label.

## Layout

```
src/semcor.py        the SGML reader; wnsn is a sense rank
src/wordnet.py       sense-ordered synsets and glosses, read from the database files
src/disambiguate.py  five methods and the oracle, with what each is allowed to read
src/run.py           the tables above
results/             word_sense.json
```

## Keywords

word sense disambiguation · WSD · SemCor · WordNet · most frequent sense · MFS baseline ·
first sense heuristic · one sense per discourse · Gale Church Yarowsky · Lesk ·
knowledge-based WSD · supervised WSD · label leakage · baseline selection · Navigli
