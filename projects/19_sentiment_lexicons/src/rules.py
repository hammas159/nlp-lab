"""Four rule sets of increasing sophistication, applied to whatever lexicon is handed in.

This is the axis the project exists to measure. Lexicon papers compare lexicons; this holds
the lexicon fixed and varies what is done with it.

``bag``          add up the polarity of every word found. No syntax, no context. What
                 "using a sentiment lexicon" means when nobody says otherwise.
``negation``     flip the polarity of words shortly after a negator. "not good" should not
                 score the same as "good".
``intensifier``  scale by boosters and dampeners. "extremely good" outweighs "good";
                 "slightly good" does not.
``contrastive``  downweight what comes before a contrastive conjunction. "The acting was
                 fine, but the film was unwatchable" is a negative sentence whose first
                 clause is positive.

Each adds to the previous, so the difference between two rows is exactly one rule.

The word lists are from Taboada et al. (2011) and the VADER paper. Where those two disagree
the disagreement is recorded rather than resolved silently - see NEGATORS below.
"""

from __future__ import annotations

import re

TOKEN = re.compile(r"[a-z']+|[!?]")

#: How far a negator reaches. VADER uses three tokens; Taboada et al. use a clause. Three is
#: the shorter, more conservative choice and is applied identically to every lexicon.
NEGATION_WINDOW = 3

#: `hardly`, `barely`, `scarcely`, `rarely` and `seldom` appear in VADER's negator list *and*
#: in its booster dictionary as strong dampeners - the published resource contradicts itself.
#: They are treated as negators here, because "hardly good" reads as negative rather than as
#: weakly positive. The choice is recorded because it is a choice.
NEGATORS = frozenset(
    [
        "no",
        "not",
        "none",
        "nobody",
        "nothing",
        "neither",
        "nor",
        "nowhere",
        "never",
        "cannot",
        "cant",
        "dont",
        "doesnt",
        "didnt",
        "isnt",
        "arent",
        "wasnt",
        "werent",
        "wont",
        "wouldnt",
        "shouldnt",
        "couldnt",
        "havent",
        "hasnt",
        "hadnt",
        "aint",
        "without",
        "lack",
        "lacks",
        "lacking",
        "hardly",
        "barely",
        "scarcely",
        "rarely",
        "seldom",
    ]
)

BOOSTERS = {
    w: 1.5
    for w in [
        "very",
        "extremely",
        "absolutely",
        "completely",
        "really",
        "totally",
        "utterly",
        "incredibly",
        "especially",
        "particularly",
        "remarkably",
        "substantially",
        "tremendously",
        "exceptionally",
        "deeply",
        "highly",
        "hugely",
        "enormously",
        "entirely",
        "purely",
        "thoroughly",
        "quite",
        "so",
    ]
}

DAMPENERS = {
    w: 0.5
    for w in [
        "slightly",
        "somewhat",
        "marginally",
        "partly",
        "fairly",
        "rather",
        "kinda",
        "sorta",
        "almost",
        "partially",
        "occasionally",
    ]
}

CONTRASTIVE = frozenset({"but", "however", "although", "though", "yet", "nevertheless"})

#: What the first clause is worth once a contrastive conjunction has overruled it. VADER
#: uses 0.5 before `but` and 1.5 after; the same shape is used here.
BEFORE_CONTRAST = 0.5
AFTER_CONTRAST = 1.5


def tokenize(text: str) -> list[str]:
    """Lowercase words, keeping `!` and `?`, and stripping the apostrophes inside
    contractions so `don't` matches the negator list as `dont`."""
    return [t.replace("'", "") or t for t in TOKEN.findall(text.lower())]


def score_bag(tokens: list[str], lexicon) -> float:
    return sum(lexicon.get(t) for t in tokens)


def score_negation(tokens: list[str], lexicon) -> float:
    total = 0.0
    for i, token in enumerate(tokens):
        value = lexicon.get(token)
        if value == 0.0:
            continue
        start = max(0, i - NEGATION_WINDOW)
        if any(t in NEGATORS for t in tokens[start:i]):
            value = -value
        total += value
    return total


def score_intensifier(tokens: list[str], lexicon) -> float:
    """Negation, plus boosters and dampeners applied to the word they modify.

    The multiplier is looked up on the token immediately before, which is where an English
    intensifier sits. A negated booster ("not very good") keeps the multiplier and flips the
    sign, which is the behaviour both source papers describe.
    """
    total = 0.0
    for i, token in enumerate(tokens):
        value = lexicon.get(token)
        if value == 0.0:
            continue
        if i > 0:
            previous = tokens[i - 1]
            value *= BOOSTERS.get(previous, DAMPENERS.get(previous, 1.0))
        start = max(0, i - NEGATION_WINDOW)
        if any(t in NEGATORS for t in tokens[start:i]):
            value = -value
        total += value
    return total


def score_contrastive(tokens: list[str], lexicon) -> float:
    """Everything above, plus: a contrastive conjunction overrules what preceded it."""
    pivot = next((i for i, t in enumerate(tokens) if t in CONTRASTIVE), None)
    if pivot is None:
        return score_intensifier(tokens, lexicon)
    before = score_intensifier(tokens[:pivot], lexicon)
    after = score_intensifier(tokens[pivot + 1 :], lexicon)
    return before * BEFORE_CONTRAST + after * AFTER_CONTRAST


RULES = {
    "bag of words": score_bag,
    "+ negation": score_negation,
    "+ intensifiers": score_intensifier,
    "+ contrastive": score_contrastive,
}


def score(rule: str, text: str, lexicon) -> float:
    try:
        return RULES[rule](tokenize(text), lexicon)
    except KeyError:
        raise ValueError(f"unknown rule set {rule!r}; expected one of {list(RULES)}") from None


def classify(value: float) -> int:
    """Positive if the total is above zero, negative otherwise.

    A zero total means the lexicon found nothing, or found an exact balance. Those are
    assigned to the negative class rather than dropped, so every corpus item is scored and
    the accuracy cannot be inflated by quietly abstaining on the hard ones. `run.py` reports
    the abstention rate separately.
    """
    return 1 if value > 0 else 0
