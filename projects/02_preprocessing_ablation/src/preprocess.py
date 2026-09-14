"""Preprocessing steps, each switchable, so their effect can be measured separately.

Every NLP tutorial teaches the same pipeline - lowercase, strip punctuation, remove
stopwords, stem - as though it were one indivisible step called "preprocessing". It is
not. It is four independent decisions, each of which can help or hurt, and almost nobody
measures them separately.

The stopword list and the stemmer are written out here rather than imported. Both are
small, both are specifications rather than algorithms with choices hidden inside them,
and a reader can check them against the source they came from.
"""

from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z0-9]+")
TOKEN_CASED = re.compile(r"[A-Za-z0-9]+")

# The classic NLTK English stopword list. Reproduced rather than imported so the exact
# set being removed is visible - "not", "no" and "own" in particular are words a
# retrieval system may badly want to keep.
STOPWORDS = frozenset(
    [
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "you're",
        "you've",
        "you'll",
        "you'd",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "she's",
        "her",
        "hers",
        "herself",
        "it",
        "it's",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "that'll",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "don't",
        "should",
        "should've",
        "now",
        "d",
        "ll",
        "m",
        "o",
        "re",
        "ve",
        "y",
        "ain",
        "aren",
        "aren't",
        "couldn",
        "couldn't",
        "didn",
        "didn't",
        "doesn",
        "doesn't",
        "hadn",
        "hadn't",
        "hasn",
        "hasn't",
        "haven",
        "haven't",
        "isn",
        "isn't",
        "ma",
        "mightn",
        "mightn't",
        "mustn",
        "mustn't",
        "needn",
        "needn't",
        "shan",
        "shan't",
        "shouldn",
        "shouldn't",
        "wasn",
        "wasn't",
        "weren",
        "weren't",
        "won",
        "won't",
        "wouldn",
        "wouldn't",
    ]
)


# --- Porter stemmer -------------------------------------------------------------------
# Porter (1980), "An algorithm for suffix stripping". Implemented rather than imported:
# it is a specification, and having it here means the exact behaviour being measured is
# readable rather than hidden behind a package version.

VOWELS = "aeiou"


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """The number of vowel-consonant sequences in the stem - Porter's `m`."""
    count = 0
    previous_vowel = False
    for i in range(len(stem)):
        vowel = not _is_consonant(stem, i)
        if previous_vowel and not vowel:
            count += 1
        previous_vowel = vowel
    return count


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _cvc(word: str) -> bool:
    """Consonant-vowel-consonant where the last is not w, x or y."""
    if len(word) < 3:
        return False
    if not (
        _is_consonant(word, len(word) - 1)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 3)
    ):
        return False
    return word[-1] not in "wxy"


STEP2 = [
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("abli", "able"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
]
STEP3 = [
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
]
STEP4 = [
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
]


def porter_stem(word: str) -> str:
    if len(word) <= 2:
        return word

    # Step 1a - plurals
    if word.endswith(("sses", "ies")):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b - past tense and progressive
    flag = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _contains_vowel(word[:-2]):
        word, flag = word[:-2], True
    elif word.endswith("ing") and _contains_vowel(word[:-3]):
        word, flag = word[:-3], True
    if flag:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
            word = word[:-1]
        elif _measure(word) == 1 and _cvc(word):
            word += "e"

    # Step 1c - terminal y
    if word.endswith("y") and _contains_vowel(word[:-1]):
        word = word[:-1] + "i"

    for suffix, replacement in STEP2:
        if word.endswith(suffix):
            if _measure(word[: -len(suffix)]) > 0:
                word = word[: -len(suffix)] + replacement
            break

    for suffix, replacement in STEP3:
        if word.endswith(suffix):
            if _measure(word[: -len(suffix)]) > 0:
                word = word[: -len(suffix)] + replacement
            break

    for suffix in STEP4:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1:
                if suffix in ("ion",) and not stem.endswith(("s", "t")):
                    break
                word = stem
            break
    if word.endswith("ion") and _measure(word[:-3]) > 1 and word[-4:-3] in "st":
        word = word[:-3]

    # Step 5 - terminal e and doubled l
    if word.endswith("e"):
        m = _measure(word[:-1])
        if m > 1 or (m == 1 and not _cvc(word[:-1])):
            word = word[:-1]
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]

    return word


# --- pipelines --------------------------------------------------------------------------


def make_tokenizer(
    lowercase: bool = True,
    remove_stopwords: bool = False,
    stem: bool = False,
    min_length: int = 1,
):
    """Build a tokenizer with a specific combination of steps switched on."""

    def tokenize(text: str) -> list[str]:
        pattern = TOKEN if lowercase else TOKEN_CASED
        tokens = pattern.findall(text.lower() if lowercase else text)
        if remove_stopwords:
            tokens = [t for t in tokens if t.lower() not in STOPWORDS]
        if stem:
            tokens = [porter_stem(t) for t in tokens]
        if min_length > 1:
            tokens = [t for t in tokens if len(t) >= min_length]
        return tokens

    return tokenize


# Each variant changes exactly one thing from the baseline, except the last, which is the
# full pipeline every tutorial recommends.
VARIANTS = {
    "baseline (lowercase only)": {},
    "+ remove stopwords": {"remove_stopwords": True},
    "+ Porter stemming": {"stem": True},
    "+ drop tokens < 3 chars": {"min_length": 3},
    "no lowercasing": {"lowercase": False},
    "stopwords + stemming": {"remove_stopwords": True, "stem": True},
    "the full tutorial pipeline": {"remove_stopwords": True, "stem": True, "min_length": 3},
}
