"""Two labelled sentiment corpora, read straight from their distributed files.

``sentence_polarity``  10,662 movie-review sentences, half positive and half negative
                       (Pang & Lee, ACL 2005). Sentence length is what lexicon methods are
                       actually used on, and short text is where rules about negation and
                       contrast have room to matter.
``twitter_samples``    10,000 tweets, half and half. A second *domain*, kept because VADER
                       was built for social media and the others were not - so if the
                       ranking between lexicons is really about the lexicon rather than
                       about the match between lexicon and domain, it should survive here.

Both are exactly balanced, so **the floor is 0.500** and every number is read against that.
"""

from __future__ import annotations

import json

from lexicons import find, require


def load_sentence_polarity() -> tuple[list[str], list[int]]:
    """One sentence per line, Latin-1, two files.

    The distribution has moved the files between `sentence_polarity/` and a nested
    `rt-polaritydata/` over the years, so both layouts are accepted rather than assuming
    whichever one happens to be on this machine.
    """
    texts, labels = [], []
    for name, label in (("rt-polarity.pos", 1), ("rt-polarity.neg", 0)):
        path = find("corpora", "sentence_polarity", name) or require(
            "corpora", "sentence_polarity", "rt-polaritydata", name
        )
        for line in path.read_text(encoding="latin-1").splitlines():
            line = line.strip()
            if line:
                texts.append(line)
                labels.append(label)
    return texts, labels


def load_twitter_samples() -> tuple[list[str], list[int]]:
    """JSON lines; the text is the `text` field."""
    root = require("corpora", "twitter_samples")
    texts, labels = [], []
    for name, label in (("positive_tweets.json", 1), ("negative_tweets.json", 0)):
        path = root / name
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                text = json.loads(line).get("text", "")
            except json.JSONDecodeError:
                continue
            if text:
                texts.append(text)
                labels.append(label)
    return texts, labels


CORPORA = {
    "movie sentences": load_sentence_polarity,
    "tweets": load_twitter_samples,
}


def available() -> list[str]:
    """Which corpora are actually on disk. Named rather than assumed, so a partial download
    produces a smaller table instead of a crash."""
    present = []
    for name, loader in CORPORA.items():
        try:
            loader()
            present.append(name)
        except LookupError:
            continue
    return present


def has(*relative: str) -> bool:
    return find(*relative) is not None
