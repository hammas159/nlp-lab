<h1 align="center">12 · Gazetteer NER</h1>
<p align="center"><i>A gazetteer's precision ceiling is set by the ambiguity inside it, not by how many names it holds.</i></p>

<p align="center">
  <a href="#the-result">Result</a> &middot;
  <a href="#the-entries-that-are-also-ordinary-english">Ordinary English</a> &middot;
  <a href="#what-a-match-is-worth">What a match is worth</a> &middot;
  <a href="#method">Method</a> &middot;
  <a href="#limitations">Limitations</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/tests-42%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/downloads%20needed-none-success" alt="no downloads">
  <img src="https://img.shields.io/badge/Aho--Corasick-implemented%2C%20not%20imported-informational" alt="ahocorasick">
</p>

---

Every HotpotQA paragraph is a Wikipedia article, so its title is an entity name and the
**66,581 of them are a gazetteer nobody had to annotate**. The paragraph is also *about*
that entity, which gives a recall signal for free: the title should appear in its own text.

Dictionary matching is the oldest NER method and still the one reached for when there is a
list of names and no training data. The question is not whether it finds the names — it
does, exactly — but **what a match is worth**.

---

## The result

| | |
|---|---:|
| titles | 66,581 |
| unmatchable (tokens absent from the corpus) | 370 |
| **distinct surface forms** | **64,382** |
| surface forms shared by more than one entity | **1,372** |
| titles covered by those | **3,201 (4.8%)** |
| single-token entries | 6,309 |

Wikipedia disambiguates with a parenthetical — `Paris (film)`, `Mercury (element)` — and
that parenthetical never appears in running text. **Stripping it is what makes an entry
matchable, and it is also what makes two entities share one entry.** 1,372 surface forms
now stand for more than one thing, covering 4.8% of the gazetteer.

Those entries are **unresolvable by construction**. No context-free matcher can tell
`Paris (film)` from `Paris (city)`, because after stripping they are the same string. That
4.8% is a precision ceiling before a single document has been read.

---

## The entries that are also ordinary English

6,309 entries are a single token. Sorted by how much of the corpus that token appears in:

| entry | appears in | share of corpus |
|---|---:|---:|
| `A+` | 55,467 | **83.3%** |
| `To` | 36,844 | **55.3%** |
| `It` | 24,905 | 37.4% |
| `One` | 10,608 | 15.9% |
| `After...` | 8,086 | 12.1% |
| `Two` | 7,671 | 11.5% |
| `Time` | 5,197 | 7.8% |
| `They` | 4,765 | 7.2% |
| `She` | 4,327 | 6.5% |

**138 single-token entries appear in over 1% of paragraphs.** Each is a real Wikipedia
article — `It` is the Stephen King novel, `Time` is the magazine — and each fires thousands
of times where the entity is not meant.

**An important qualification, because it changes what this table means.** The tokenizer is
the lab's shared one: lowercased runs of word characters, punctuation dropped. So `A+`
becomes the pattern `a`, and `After...` becomes `after`. The 83.3% is the frequency of the
token `a`, not of the string `A+`.

That is not a flaw to apologise for — it is the same finding one level down. **The
normalisation that makes a gazetteer matchable is also what destroys the distinctions that
made some entries specific.** `A+` is a perfectly unambiguous string; `a` is the most common
token in English. A matcher that preserved the punctuation would lose every entry whose
surface form differs from its article title in any way, which is most of them. Both
directions lose, and the gazetteer cannot tell you which loss you are taking.

---

## What a match is worth

Tagging 3,000 paragraphs with the full 64,382-entry automaton, resolving overlaps by
longest match:

| | |
|---|---:|
| the paragraph's own title was found | **71.3%** |
| other gazetteer entries also matched — median | **9** |
| — mean | 9.5 |
| — maximum | 40 |
| overlapping matches discarded by resolution | 4,400 |

A paragraph about one entity matches, on average, **its own name plus 9.5 other entity
names**. A tagger with no disambiguation returns all of them, and has no way to rank them:
every match is exact, so there is no score to threshold.

That is the precision ceiling, and **it is a property of the gazetteer rather than of the
matcher.** Improving the matching algorithm cannot help — the matches are already exact.
Only context can, and a gazetteer has none.

Recall has the opposite shape: 71.3% is not the matcher failing but the article not always
restating its own title in the text, plus the 370 titles whose tokens the corpus never
contains.

---

## Method

**Aho-Corasick, implemented rather than imported.** Matching 64,382 names one at a time is
64,382 passes over the text. The automaton builds a trie of the patterns and adds failure
links, so one pass finds everything.

On 200 patterns it is **6.8× faster** than the naive per-pattern scan. The full gazetteer is
322× larger than that subset, and the naive scan grows with the pattern count while the
automaton does not — that ratio is a floor, not a result.

**Output links are the part that fails silently.** A node matching `york` must also report
`new york` ending there. Get the merge wrong and the automaton returns only the shortest of
any nested pair — no error, just fewer matches, on a gazetteer full of nested names.
`test_nested_patterns_are_all_reported` and `test_three_deep_nesting` assert it directly.
Outputs are merged along the failure chain **at build time** rather than walked per
position, which keeps the scan linear.

**Matching is over tokens, not characters.** A character automaton matches `cat` inside
`catalogue` and `Al` inside `Algorithm`; on 64,000 names that produces more spurious matches
than real ones. Tokens are interned to integers, so the automaton's alphabet is the
vocabulary rather than Unicode.

**Longest-match resolution is a convention, not a result.** `Paris` inside
`University of Paris` is a genuine match of a genuine entity. Keeping both is indefensible
and so is keeping the short one; this picks the longest and says so.
`test_resolution_is_a_convention_not_a_result` records that it is a choice.

---

## Limitations

- **No annotated entity spans.** Nothing here measures true precision, because that needs
  someone to have marked which mentions are real. What is measured is the *ceiling* —
  ambiguity within the gazetteer and the number of competing candidates per paragraph —
  which bounds precision without needing annotation.
- **The self-title recall signal is a proxy.** An article not restating its own title is
  counted as a miss even though nothing was missed, and a paragraph mentioning its title in
  a variant form (`the Seine` for `Seine River`) is also counted as a miss.
- **The tokenizer drives the noisiest rows.** See the qualification above: `A+` becoming `a`
  is a property of this preprocessing, not of Wikipedia. A different tokenizer gives a
  different table and a different set of problems.
- **Titles are not the same as mentions.** Wikipedia titles carry conventions — inverted
  names, appended qualifiers — that running text does not use, so this gazetteer is a
  harder starting point than a curated entity list would be.
- **One corpus.** HotpotQA's paragraphs are Wikipedia introductions, which mention their own
  subject unusually often. A gazetteer over news or clinical text would behave differently
  in both directions.
- **The speed comparison is small.** 200 patterns over 200 paragraphs, because the naive
  scan at full scale would not finish. It establishes the direction and the order of
  magnitude, not a benchmark.

## Run it

```bash
python src/run.py            # the full gazetteer, ~9 seconds
python src/run.py --quick    # a tenth of the corpus
pytest -q                    # 42 tests, no dataset, no network
```

## Layout

```
src/ahocorasick.py  the automaton, failure links, output merging, overlap resolution
src/gazetteer.py    titles into patterns, disambiguator stripping, collision counting
src/run.py          the tables above
results/            gazetteer.json
```

## Keywords

gazetteer · dictionary NER · named entity recognition · Aho-Corasick · multi-pattern
matching · trie · failure links · string matching · entity ambiguity · entity linking ·
surface form · disambiguation · longest match · precision ceiling · HotpotQA · Wikipedia
