"""An Urdu retrieval benchmark, and a tokenizer that actually returns something for it.

[Project 18](../18_tokenisation) trained BPE, WordPiece and Unigram on English Wikipedia
prose and found that subword tokenisation **buys nothing** for a lexical retriever: the best
subword configuration beat plain words by 0.002 after spending a 16,000-piece vocabulary to
stop splitting words at all. Its limitations section named the obvious objection:

> **One corpus, English, Wikipedia prose.** The case for subwords is strongest in
> morphologically rich languages, which is exactly where this says nothing.

This is that corpus. Urdu is written in the Perso-Arabic script, inflects heavily, and
concatenates clitics onto word forms - so a word-level index should fragment across
inflected forms in a way English's does not.

**The lab's shared tokenizer returns nothing here.** `shared.benchmark.tokenize` matches
`[a-z0-9]+`, which finds zero tokens in Urdu text - not an error, just an empty list, and
every downstream number would have been silently computed on empty documents. That is the
kind of failure this lab exists to catch, so it is tested rather than assumed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: Urdu letters, diacritics and digits - but NOT the punctuation that shares the Arabic
#: block with them. The Urdu comma (U+060C), full stop (U+06D4), question mark (U+061F) and
#: semicolon (U+061B) all sit inside 0600-06FF, so the obvious `[؀-ۿ]+` keeps them
#: as if they were letters and every "word" ending a sentence gets a full stop glued to it.
#: Latin and ASCII digits are kept as their own runs, because Urdu Wikipedia is full of
#: names, dates and units written in them.
TOKEN = re.compile(
    r"[ؠ-يً-ٟ٠-٩ٰ-ۓە"
    r"ۥ-ۦۮ-ۿݐ-ݿ]+"
    r"|[A-Za-z]+"
    r"|[0-9]+"
)

#: Written by src/fetch.py, which pulls byte ranges because the whole 167.6 MB file will
#: not come down on this connection. Local, so every later run and every test is offline.
ARTICLES = Path(__file__).resolve().parent.parent / "data" / "urdu_articles.parquet"


def tokenize(text: str) -> list[str]:
    """Urdu words, plus any Latin or numeric runs. Lowercased for the Latin part only -
    the Perso-Arabic script has no case, so lowercasing is a no-op there."""
    return [t.lower() for t in TOKEN.findall(text)]


def find_parquet() -> Path | None:
    """The extracted corpus, or an explicit override for tests."""
    if env := os.environ.get("URDU_ARTICLES"):
        candidate = Path(env)
        return candidate if candidate.exists() else None
    return ARTICLES if ARTICLES.exists() else None


def build(limit: int = 3_000, min_tokens: int = 60, seed: int = 0):
    """Return (titles, bodies, queries) for a title-to-body retrieval task.

    Urdu has no HotpotQA, so the benchmark is built the way title-to-passage retrieval
    benchmarks usually are: **the article title is the query and its body is the one gold
    document.** Titles are unique, so the ground truth is exact and needs no judgements -
    the same property that made HotpotQA usable for the rest of this lab.

    The title is removed from the body before indexing. Leaving it in would let every
    retriever win by matching the query against a copy of itself, which measures nothing.
    """
    import numpy as np
    import pandas as pd

    path = find_parquet()
    if path is None:
        raise LookupError(
            "Urdu Wikipedia not found. Fetch it with:\n  python src/fetch.py --megabytes 20"
        )

    frame = pd.read_parquet(path, columns=["title", "text"])
    titles, bodies = [], []
    for title, text in zip(frame["title"], frame["text"]):
        title = str(title).strip()
        body = strip_title(str(text), title)
        if len(tokenize(body)) >= min_tokens and len(tokenize(title)) >= 1:
            titles.append(title)
            bodies.append(body)

    if limit and limit < len(titles):
        rng = np.random.default_rng([seed, 0x0072_6475])
        pick = sorted(rng.choice(len(titles), size=limit, replace=False).tolist())
        titles = [titles[i] for i in pick]
        bodies = [bodies[i] for i in pick]
    return titles, bodies, list(titles)


def strip_title(text: str, title: str) -> str:
    """Remove the title from the body, so the query is not literally present in its answer.

    Urdu Wikipedia articles almost always open by repeating the title in bold. Leaving it
    in turns the task into exact string matching and every tokenizer scores near 1.0, which
    would hide exactly the differences the project is measuring.
    """
    if not title:
        return text
    return text.replace(title, " ")
