"""Four sentence splitters of increasing sophistication, and a taxonomy of what breaks them.

Sentence segmentation is the first step of almost every pipeline and is treated as solved
preprocessing - a line of setup before the interesting part. It is not solved; it is
*conventional*, and the conventions differ.

``naive``        split after any of . ! ?
``abbreviation`` the same, but not after a known abbreviation
``trained``      the same, but the abbreviation list is *learned from the corpus* rather
                 than supplied - the core idea of Punkt (Kiss & Strunk, 2006): a token
                 that precedes a period far more often than chance is probably an
                 abbreviation, whatever language the text is in
``conservative`` require the following token to look like a sentence start (capital letter
                 or digit) as well

None of these is the right answer, because there is no right answer available here - see
the note in `run.py` about what the reference segmentation actually is.
"""

from __future__ import annotations

import re
from collections import Counter

#: Periods that end a sentence are followed by whitespace. This finds every candidate.
BOUNDARY = re.compile(r"[.!?]")

#: The list a regex splitter ships with. Deliberately short: a long list hides the point,
#: which is that the list is doing the work and is never complete.
COMMON_ABBREVIATIONS = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "jr",
        "sr",
        "vs",
        "etc",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "est",
        "fig",
        "no",
        "vol",
        "pp",
        "ed",
        "eds",
        "approx",
        "dept",
        "univ",
        "assn",
        "bros",
        "ave",
        "blvd",
        "rd",
        "apt",
        "ft",
        "cf",
        "ie",
        "eg",
        "al",
    ]
)


def naive(text: str) -> list[int]:
    """Every . ! ? followed by whitespace is a boundary.

    Returns the character offsets *after* which a sentence ends. This is the splitter that
    every tutorial writes and that nobody admits to shipping.
    """
    return [m.end() for m in BOUNDARY.finditer(text) if _followed_by_space(text, m.end())]


def _followed_by_space(text: str, position: int) -> bool:
    """Whitespace must actually follow.

    The end of the text is **not** a boundary here. A sentence obviously ends there, but
    the reference segmentation encodes only the *internal* splits - it is a list of pieces,
    and n pieces have n-1 joins. Emitting one anyway put a guaranteed false positive on
    every paragraph in the corpus and cost about fifteen points of precision before it was
    noticed, because the resulting errors were misfiled under a different category.
    """
    return position < len(text) and text[position].isspace()


def _preceding_token(text: str, position: int) -> str:
    """The word immediately before the punctuation at ``position - 1``."""
    end = position - 1
    start = end
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "'"):
        start -= 1
    return text[start:end].lower()


def _following_token(text: str, position: int) -> str:
    start = position
    while start < len(text) and text[start].isspace():
        start += 1
    end = start
    while end < len(text) and not text[end].isspace():
        end += 1
    return text[start:end]


def abbreviation(text: str, known: frozenset[str] = COMMON_ABBREVIATIONS) -> list[int]:
    """Naive, minus any period preceded by a known abbreviation."""
    out = []
    for match in BOUNDARY.finditer(text):
        position = match.end()
        if not _followed_by_space(text, position):
            continue
        if text[position - 1] == "." and _preceding_token(text, position) in known:
            continue
        out.append(position)
    return out


def learn_abbreviations(
    texts: list[str], threshold: float = 0.85, min_count: int = 4
) -> frozenset[str]:
    """Tokens that appear with a trailing period far more often than not.

    This is Punkt's central observation reduced to one rule: an abbreviation is a token
    that is *bound* to its period. `Mr` is followed by a period almost every time it
    appears; `cat` almost never is. No language-specific list is needed, which is the whole
    appeal - and the threshold is a parameter nobody reports.
    """
    with_period: Counter = Counter()
    total: Counter = Counter()
    for text in texts:
        for match in re.finditer(r"([A-Za-z][A-Za-z']*)(\.?)", text):
            token = match.group(1).lower()
            total[token] += 1
            if match.group(2):
                with_period[token] += 1
    return frozenset(
        token
        for token, count in total.items()
        if count >= min_count and with_period[token] / count >= threshold
    )


def trained(text: str, learned: frozenset[str]) -> list[int]:
    return abbreviation(text, known=learned)


def conservative(text: str, known: frozenset[str] = COMMON_ABBREVIATIONS) -> list[int]:
    """Also require the next token to look like a sentence start.

    Catches decimals and version numbers - `3.5` is not two sentences - at the cost of
    losing any sentence that genuinely begins with a lowercase word, which in this corpus
    means anything starting with a lowercase proper noun or a stylised title.
    """
    out = []
    for position in abbreviation(text, known=known):
        following = _following_token(text, position)
        if not following:
            out.append(position)
            continue
        if following[0].isupper() or following[0].isdigit() or following[0] in "\"'([":
            out.append(position)
    return out


def classify_disagreement(text: str, position: int) -> str:
    """Name the construction at a disputed boundary.

    The point of the taxonomy is that disagreements are not spread evenly over the text -
    they sit on a handful of constructions, and which of them a splitter handles is a
    design decision rather than a quality level.
    """
    before = _preceding_token(text, position)
    following = _following_token(text, position)
    character = text[position - 1] if position <= len(text) else "."

    if character in "!?":
        return "exclamation or question"
    if before in COMMON_ABBREVIATIONS:
        return "known abbreviation"
    if len(before) == 1 and before.isalpha():
        return "single initial"
    if before.isdigit() and following[:1].isdigit():
        return "decimal or version number"
    if before.isdigit():
        return "number before period"
    if not following:
        return "end of text"
    if following[0].islower():
        return "lowercase follows"
    # `following[0] in "\"'(["` and not `following[:1] in ...`: the empty string is a
    # substring of every string, so the slice form returns True at end of text and filed
    # thousands of end-of-paragraph errors under "quote follows".
    if following[0] in "\"'([":
        return "quote or bracket follows"
    if before and before[-1:].isupper():
        return "acronym"
    return "other"


SPLITTERS = ("naive", "abbreviation", "trained", "conservative")
