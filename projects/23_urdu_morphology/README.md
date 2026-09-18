<h1 align="center">23 · Low-resource morphology</h1>
<p align="center"><i>Project 18 said subwords buy nothing, and said it only for English. On Urdu the algorithm is worth 6× more.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#project-18s-conclusion-does-not-travel">What does not travel</a> &middot;
  <a href="#the-tokenizer-that-quietly-indexes-3-of-the-corpus">A 3% index</a> &middot;
  <a href="#getting-the-corpus-at-20-kbs">Getting the corpus</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-15%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/corpus-5%2C029%20Urdu%20articles-orange" alt="corpus">
  <img src="https://img.shields.io/badge/fetched%20by-HTTP%20range-informational" alt="ranges">
</p>

---

[Project 18](../18_tokenisation) trained BPE, WordPiece and Unigram on English Wikipedia and
found that subword tokenisation **buys nothing** for a lexical retriever, and that the choice
of algorithm is worth less than the vocabulary-size knob beside it. Its limitations section
named the objection:

> **One corpus, English, Wikipedia prose.** The case for subwords is strongest in
> morphologically rich languages, which is exactly where this says nothing.

This is that corpus: **5,029 Urdu Wikipedia articles, 4.7 million tokens**, Perso-Arabic
script, heavy inflection and clitics. Same three algorithms, same vocabulary sizes, same
BM25, same measurement.

---

## The result

Title-to-body retrieval: the article title is the query, its body is the one gold document,
and the title is stripped from the body so nothing wins by matching a copy of itself.

| Tokenizer | vocab | fertility | intact | **recall@10** |
|---|---:|---:|---:|---:|
| BPE | 2,000 | 1.625 | 0.612 | 0.306 |
| WordPiece | 2,000 | **2.427** | **0.364** | **0.134** |
| Unigram | 2,000 | 2.014 | 0.448 | 0.251 |
| BPE | 4,000 | 1.325 | 0.776 | 0.405 |
| WordPiece | 4,000 | 1.456 | 0.710 | 0.398 |
| Unigram | 4,000 | 1.679 | 0.542 | 0.366 |
| BPE | 8,000 | 1.171 | 0.875 | 0.452 |
| WordPiece | 8,000 | 1.216 | 0.850 | 0.456 |
| Unigram | 8,000 | 1.529 | 0.591 | 0.423 |
| BPE | 16,000 | 1.093 | 0.931 | 0.471 |
| **WordPiece** | 16,000 | 1.107 | 0.922 | **0.472** |
| Unigram | *16,000* | 1.478 | 0.607 | 0.446 |
| **plain words** | 130,155 | 1.000 | 1.000 | **0.433** |

**Subwords are worth +0.039 here. Project 18 measured +0.002 on English.**

The prediction the earlier project made about itself was right: this is where subwords earn
their keep. A word-level Urdu index fragments across inflected forms, and a subword
vocabulary recovers the shared stems that BM25's exact matching otherwise misses.

---

## Project 18's conclusion does not travel

That is the smaller half of the story. The larger half is what happens to project 18's
*other* finding — that the algorithm barely matters:

| | English (project 18) | **Urdu (here)** |
|---|---:|---:|
| spread between algorithms at a fixed vocabulary | 0.030 | **0.172** |
| spread across vocabulary sizes | 0.075 | **0.337** |
| subwords vs plain words | +0.002 | **+0.039** |

**The algorithm is worth nearly six times more on Urdu than on English.** At a 2,000-piece
vocabulary, BPE scores 0.306 and WordPiece scores **0.134** — the same budget, the same
corpus, the same retriever, and less than half the result.

So project 18's headline — *"at a fixed vocabulary size the three algorithms differ by at
most 0.030"* — is **a fact about English**, not about tokenisation. Repeating its experiment
in the language it flagged does not merely strengthen the conclusion; it breaks the part that
sounded most general.

What does survive is the ranking of the two knobs. Vocabulary size still moves the result
about twice as much as the algorithm does (0.337 against 0.172), exactly as on English
(0.075 against 0.030). **The advice "tune the vocabulary size before arguing about the
algorithm" holds in both languages. The claim "the algorithm is nearly free" holds only in
one.**

### WordPiece is where it breaks

WordPiece is the algorithm that collapses. At 2,000 pieces its fertility is **2.427** against
BPE's 1.625 — it splits the average Urdu word into nearly two and a half pieces, and leaves
only 36% intact. By 16,000 pieces it has caught up and narrowly wins.

Its likelihood-based merge criterion needs a vocabulary large enough to cover Urdu's
morphology before it produces useful pieces; below that it shatters words into fragments too
short to discriminate. On English the same budget was already sufficient, so the failure
never appeared.

---

## The tokenizer that quietly indexes 3% of the corpus

Every other project in this lab shares `shared.benchmark.tokenize`. It matches `[a-z0-9]+`.

On Urdu it keeps **14,363 of 463,753 tokens — 3.1%** — and returns something non-empty for
**192 of 200** articles. What survives is stray Latin and digits: `kh`, `mi`, `1922`.

An index built with it would look populated, score non-zero, and rank documents by the
scattering of Latin fragments inside them. **That is worse than returning nothing**, which
would at least fail loudly. The first test in this project pins the behaviour, because a
project that reused the shared tokenizer out of habit would have produced a full results
table with no error anywhere in it.

---

## Getting the corpus at 20 KB/s

Urdu Wikipedia ships as one **167.6 MB** parquet. On this machine `hf_hub_download` stalls at
zero bytes on it and `HfFileSystem` streaming dies with `httpx.ReadTimeout` — but **small
HTTP range requests with many retries succeed at about 30 KB/s**.

So `src/fetch.py` takes the file in pieces:

1. the last 1 MB, which holds the parquet **footer** — the metadata describing every row
   group and its byte offsets;
2. the first 20 MB, in 5 MB chunks, each retried independently;
3. a local file of the original's exact length with those two pieces at their original
   offsets and a hole between them.

Offsets in a parquet footer are absolute, so the row groups lying entirely inside the
downloaded head read from that sparse file exactly as they would from the whole one. **Six of
201 row groups arrived complete: 6,000 articles for 21 MB of transfer instead of 168.**
Groups reaching into the hole are excluded by computing the boundary from the footer, not by
reading them and hoping — a group straddling the hole would come back as zeros, which is not
an error, just wrong data.

Each chunk is written as it arrives and a re-run skips what is on disk, so an interrupted
fetch resumes. That lesson is from project 01, where an embedding cache written only at the
end lost four hours of work.

---

## Limitations

- **The absolute numbers are not comparable to project 18's.** Its English task was
  HotpotQA question-to-passage with a 0.865 word baseline; this is title-to-body with a
  0.433 baseline. Different tasks of different difficulty. **Only the *gaps within each
  language* are comparable** — subword-minus-word, and algorithm spread — which is what
  every claim above is stated in.
- **The type/token ratios are not comparable either**, and the run says so: this corpus is
  4.7M tokens against project 18's 277K, and project 05 measured exactly how much the ratio
  falls with corpus size. Heaps' law makes the larger corpus look less diverse whatever the
  language, so no morphological conclusion is drawn from it.
- **6,000 articles is the first 3% of Urdu Wikipedia**, in whatever order the parquet stores
  them — not a random sample. If early row groups are correlated with article age or
  popularity, the corpus is skewed in a way this design cannot detect.
- **One low-resource language, and only nominally low-resource.** Urdu Wikipedia has 200,154
  articles. A language with a thousandth of that would stress subword vocabularies far
  harder, and nothing here speaks to it.
- **No morphological analysis.** The claim that subwords recover shared stems is an
  inference from the retrieval gap, not a measurement — no Urdu morphological analyser was
  used to check whether the learned pieces correspond to real morphemes.
- **Title-to-body is a proxy task.** It is standard, and its ground truth is exact, but it
  rewards lexical overlap between a title and its article rather than genuine information
  need.

## Run it

```bash
python src/fetch.py --megabytes 20   # once, ~10 minutes on a slow link; resumable
python src/run.py                    # the comparison, ~12 minutes
python src/run.py --quick            # 800 documents, two vocabulary sizes
pytest -q                            # 15 tests, no corpus, no network
```

## Layout

```
src/fetch.py   byte-range fetch and sparse-file assembly
src/urdu.py    the Urdu tokenizer and the title-to-body benchmark
src/run.py     the tables above; imports project 18's tokenizers rather than copying them
results/       urdu.json
```

The tokenizers are **imported from project 18**, not reimplemented. The whole point is to
compare against its numbers, and a second implementation would leave any difference ambiguous
between "Urdu" and "a different BPE".

## Keywords

low-resource NLP · morphology · Urdu · Perso-Arabic script · tokenisation · subword · BPE ·
WordPiece · Unigram LM · fertility · BM25 · retrieval · cross-lingual generalisation ·
Wikipedia · parquet row groups · HTTP range requests
