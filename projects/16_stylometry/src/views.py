"""Four views of the same function, each removing more of what is *about* the code.

Authorship attribution reports that a writer can be identified from style alone - function
words, punctuation habits, sentence rhythm - independent of subject. On source code the
equivalent claim is that a codebase has a recognisable house style.

The difficulty is that identifiers carry both. `av_frame_alloc` is a naming *convention*
(lower_snake, project prefix) and also a *topic* (video frames). A classifier given raw
tokens can use either, and reporting its accuracy as "style" assumes it used the first.

These four views strip topic progressively while holding the classifier fixed, so the drop
between them is the contribution of what was removed:

``lexical``     every token, identifiers included. Style *and* topic.
``masked``      identifiers, numbers and string literals replaced by placeholders. The
                *shape* of the code survives - which keywords, how often, in what order -
                and the subject matter does not.
``structural``  identifiers dropped entirely; only C keywords, operators and punctuation
                remain. Pure grammar and formatting habit.
``layout``      no tokens at all. Line lengths, indentation widths, blank-line and comment
                ratios, brace placement - the typographic fingerprint, binned into features.

Only ``layout`` and ``structural`` are defensible as "style". If accuracy on ``lexical``
is far above them, the headline number was topic.
"""

from __future__ import annotations

import re
from collections import Counter

#: C keywords plus the operators and punctuation that carry grammar. Anything not in here
#: and not punctuation is an identifier.
_KEYWORD_SOURCE = """
auto break case char const continue default do double else enum extern float for
goto if inline int long register restrict return short signed sizeof static struct
switch typedef union unsigned void volatile while
"""
C_KEYWORDS = frozenset(_KEYWORD_SOURCE.split())

TOKEN = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*|\"[^\"]*\"|'[^']*'|[^\s\w]")
IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
NUMBER = re.compile(r"^\d")
STRING = re.compile(r"^[\"']")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text)


def lexical(text: str) -> list[str]:
    return tokenize(text)


def masked(text: str) -> list[str]:
    """Identifiers, numbers and strings become placeholders.

    A function keeps its shape - `if ( ID ) { ID = ID ( ID , NUM ) ; }` - and loses what it
    is about. Crucially the *count* of identifiers per statement survives, so naming density
    is still measurable while naming content is not.
    """
    out = []
    for token in tokenize(text):
        if token in C_KEYWORDS:
            out.append(token)
        elif STRING.match(token):
            out.append("STR")
        elif NUMBER.match(token):
            out.append("NUM")
        elif IDENTIFIER.match(token):
            out.append("ID")
        else:
            out.append(token)
    return out


def structural(text: str) -> list[str]:
    """Identifiers removed outright. Only keywords, operators and punctuation survive."""
    return [
        token
        for token in tokenize(text)
        if token in C_KEYWORDS
        or not (IDENTIFIER.match(token) or NUMBER.match(token) or STRING.match(token))
    ]


#: Bin edges for the layout features. Continuous measurements are binned rather than used
#: raw so the same multinomial classifier can read them - a classifier change between views
#: would make the comparison a comparison of classifiers.
_LINE_BINS = (0, 20, 40, 60, 80, 120, 10_000)
_INDENT_BINS = (0, 1, 2, 4, 8, 1_000)


def _bin(value: float, edges: tuple[int, ...]) -> int:
    for i, edge in enumerate(edges[1:]):
        if value < edge:
            return i
    return len(edges) - 2


def layout(text: str) -> list[str]:
    """Typographic habits only: line lengths, indent widths, brace placement, comment ratio.

    No token content whatsoever. If a codebase is identifiable from this alone, that is
    style in the strictest sense available - it is how the code is *set on the page*.
    """
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        out.append(f"len{_bin(len(line), _LINE_BINS)}")
        indent = len(line) - len(line.lstrip())
        out.append(f"ind{_bin(indent, _INDENT_BINS)}")
        if not stripped:
            out.append("blank")
            continue
        if stripped.startswith(("/*", "*", "//")):
            out.append("comment")
        if stripped.startswith("{"):
            out.append("brace_own_line")
        if stripped.endswith("{"):
            out.append("brace_trailing")
        if stripped.startswith("#"):
            out.append("preprocessor")
        if stripped.endswith(";"):
            out.append("stmt_end")
        if "\t" in line:
            out.append("tab")
    return out


VIEWS = {
    "lexical": lexical,
    "masked": masked,
    "structural": structural,
    "layout": layout,
}


def view(name: str, text: str) -> list[str]:
    try:
        return VIEWS[name](text)
    except KeyError:
        raise ValueError(f"unknown view {name!r}; expected one of {sorted(VIEWS)}") from None


def vocabulary_of(texts: list[str], name: str, min_count: int = 3) -> dict[str, int]:
    counts: Counter = Counter()
    for text in texts:
        counts.update(view(name, text))
    return {tok: i for i, (tok, n) in enumerate(counts.most_common()) if n >= min_count}
