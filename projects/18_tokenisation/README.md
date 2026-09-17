<h1 align="center">18 · Tokenisation</h1>
<p align="center"><i>BPE, WordPiece or Unigram is worth 0.030. The vocabulary size is worth 0.075.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-intrinsic-metrics-always-pick-bpe-the-task-does-not">The intrinsic metrics always pick BPE</a> &middot;
  <a href="#unigram-is-the-odd-one-out">Unigram is the odd one out</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-39%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/tokenizers-trained%20here-informational" alt="trained here">
</p>

---

Tokenizers are compared on **fertility** (pieces per word), vocabulary statistics, and how
morphological their pieces look. All three are intrinsic: they describe the tokenizer, and
assume something about what that means downstream.

This trains **BPE**, **WordPiece** and **Unigram** on the same corpus at matched vocabulary
sizes, with the same normaliser and the same pre-tokenizer, and scores each by handing its
output to BM25. The retriever, corpus, queries and metric never change, so the only variable
is the segmentation.

It also keeps the baseline that a subword comparison usually omits: **plain words**.

---

## The result

HotpotQA · 2,964 documents · 277,259 word tokens · 25,295 word types · 300 queries.

| Tokenizer | vocab | fertility | intact | **recall@10** |
|---|---:|---:|---:|---:|
| **plain words** | 25,295 | 1.000 | 1.000 | **0.865** |
| BPE | 2,000 | 1.668 | 0.618 | 0.797 |
| WordPiece | 2,000 | 1.929 | 0.534 | 0.792 |
| Unigram | 2,000 | 1.721 | 0.603 | 0.793 |
| BPE | 4,000 | 1.374 | 0.760 | 0.845 |
| WordPiece | 4,000 | 1.475 | 0.722 | **0.860** |
| Unigram | 4,000 | 1.496 | 0.663 | 0.830 |
| BPE | 8,000 | 1.188 | 0.867 | 0.862 |
| WordPiece | 8,000 | 1.230 | 0.847 | 0.862 |
| Unigram | 8,000 | 1.385 | 0.689 | 0.847 |
| BPE | 16,000 | 1.080 | 0.939 | 0.858 |
| **WordPiece** | 16,000 | 1.099 | 0.927 | **0.867** |
| Unigram | *13,401* | 1.370 | 0.694 | 0.852 |

**At a fixed vocabulary size, the three algorithms differ by at most 0.030. Changing the
vocabulary size moves the result by 0.075** — two and a half times as much.

The argument about which subword algorithm to use is an argument about the smaller of two
knobs, and the larger one is a number typed into a config.

### The subword apparatus buys nothing here

The best subword configuration in the table scores **0.867** against plain words' **0.865**.
That is +0.002, which is noise, and it took a 16,000-piece vocabulary to get there — large
enough that 93% of words are no longer split at all. Every smaller budget is worse, by up to
0.073.

This is not an argument against subwords in general. It is a specific claim about what they
do to a **lexical** retriever: BM25 scores by term rarity, and splitting a rare
discriminative word into common pieces is precisely the operation that destroys rarity.
Subwords earn their keep where an out-of-vocabulary word would otherwise be unrepresentable
— a neural model with a fixed embedding table. BM25 has no such limit, so the trade has an
upside of zero and a downside that grows as the vocabulary shrinks.

---

## The intrinsic metrics always pick BPE. The task does not.

Across the whole grid, fertility looks like a strong predictor: **r = −0.908** with
recall@10, and intact rate **+0.843**. Both are misleading, because both mostly track the
vocabulary size, which is set rather than discovered.

The question that could actually guide a decision is narrower: *at a size you have already
chosen, does the intrinsic metric pick the algorithm the task picks?*

| vocab | fertility says | intact rate says | **the task says** |
|---|---|---|---|
| 2k | BPE | BPE | BPE ✅ |
| 4k | BPE | BPE | **WordPiece** ❌ |
| 8k | BPE | BPE | BPE / WordPiece ✅ |
| 16k | BPE | BPE | **WordPiece** ❌ |

**Both metrics agree with the task at 2 of 4 sizes — and both name BPE every single time.**
They are not weak predictors, they are constant ones. BPE produces the fewest pieces per
word at every budget, so any metric that rewards fewer pieces has already decided before
the task is run.

---

## Unigram is the odd one out

The three do not fail to agree evenly. At a 8,000-piece vocabulary:

| Pair | identical segmentation |
|---|---:|
| BPE vs WordPiece | **91.5%** |
| BPE vs Unigram | 70.7% |
| WordPiece vs Unigram | 70.3% |

BPE and WordPiece are near-duplicates — unsurprising, since both build bottom-up by merging
pairs and differ only in the merge criterion. **Unigram is a genuinely different tokenizer**,
built top-down by pruning a large candidate set, and it disagrees with both by the same
margin.

You can see it in the pieces:

```
entanglement       BPE  ent | ang | le | ment
              WordPiece  ent | ##ang | ##lement
                Unigram  en | tang | le | ment

microbiologist     BPE  mic | ro | bi | ologist
              WordPiece  micro | ##bi | ##ologist
                Unigram  micro | biolog | is | t

unhappiness        BPE  un | happ | iness
              WordPiece  un | ##ha | ##pp | ##iness
                Unigram  un | happi | ness
```

Bostrom & Durrett (Findings of EMNLP 2020) found Unigram's pieces align more closely with
morphology than BPE's and argued BPE's greedy construction is why. `un | happi | ness` and
`micro | biolog | ...` against `un | happ | iness` and `mic | ro | bi | ologist` is that
claim visible in three words.

**And Unigram is the worst of the three on this task at every vocabulary size above 2,000.**
Its morphological pieces do not help a lexical retriever, because BM25 does not know that
`happi` and `happy` are related — it only knows how rare each string is, and Unigram keeps
splitting when the others have stopped (fertility 1.370 at 13k, where BPE is at 1.080).

Better morphology, worse retrieval. Which is what "intrinsic" means.

---

## Method

**Everything except the algorithm is pinned.** Same corpus, same target vocabulary size,
same NFD + strip-accents + lowercase normaliser, same whitespace pre-tokenizer. Three
tokenizers with different normalisation would be a normalisation comparison.

**Intrinsic metrics are computed on a frequency-weighted sample.** Measuring fertility over
unique types describes the dictionary; what a tokenizer does to a corpus is dominated by the
words that actually occur, so the 4,000-word sample is drawn by frequency.

**`[UNK]` is measured, not assumed.** The three algorithms fail differently: Unigram falls
back to characters, BPE loses only the individual characters it never saw, and **WordPiece
discards the entire word** for one `[UNK]` if any prefix fails to match. On this corpus the
rate is **0.000 for all three** — every character appears in training — so none of the
accuracy differences above are a failure-mode artefact. Three tests pin each behaviour, so
the day it does bite, it will not be silent.

**Continuation markers are stripped before comparing segmentations.** WordPiece writes
`##ing` where BPE writes `ing`. Comparing raw strings would have scored two identical
segmentations as different and produced a striking, wrong agreement number.

---

## Limitations

- **Unigram could not reach 16,000 pieces** — it prunes down from a candidate set and
  stopped at 13,401. So the top row of the matched-size control is not actually matched, and
  Unigram's last result is at a smaller budget than its competitors. Its fertility barely
  moves between 8k and 13k, so this probably costs it little, but "probably" is the honest
  word and the row is marked in the table.
- **A lexical retriever is one downstream task, and an unusually unforgiving one.** The
  conclusion that subwords buy nothing applies to BM25, not to a neural encoder, which is
  where subwords were invented to help. Project 01 shows what a corpus-trained neural model
  does on this data, and it is not good enough for this to be a fair second opinion.
- **One corpus, English, Wikipedia prose.** The case for subwords is strongest in
  morphologically rich languages, which is exactly where this says nothing.
- **Fertility, intact rate and vocabulary size are not independent.** They move together by
  construction, which is why the winner-agreement table exists — the correlations alone
  would overstate how much the intrinsic metrics know.
- **No significance testing on the 0.002 gap.** It is reported as noise rather than as a
  win, but the paired bootstrap that project 02 uses was not run here.

## Run it

```bash
python src/run.py            # 2k, 4k, 8k, 16k vocabularies, ~70 seconds
python src/run.py --quick    # 2k and 8k
pytest -q                    # 39 tests, no dataset, no network
```

## Layout

```
src/segmenters.py   the three algorithms, identically configured, plus the intrinsic metrics
src/run.py          the tables above
results/            tokenisation.json
```

## Keywords

tokenisation · subword · BPE · byte pair encoding · WordPiece · Unigram LM · SentencePiece ·
fertility · vocabulary size · intrinsic vs extrinsic evaluation · BM25 · retrieval ·
morphology · segmentation agreement · HotpotQA · Sennrich · Kudo · Bostrom & Durrett
