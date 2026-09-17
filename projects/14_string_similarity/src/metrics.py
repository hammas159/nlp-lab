"""Seven string similarity measures, implemented rather than imported.

They fall into three families, and the families disagree about what "similar" means:

**edit distance** counts operations. ``levenshtein`` allows insert, delete and substitute;
``damerau`` adds transposition, which matters because swapping two adjacent letters is one
of the commonest typing errors and Levenshtein charges two operations for it.
``keyboard`` is Levenshtein with a substitution cost that depends on how far apart the two
keys are - a model of *how* typing goes wrong rather than *how much*.

**phonetic** hashes a string to a code that sounds like it. ``soundex`` and ``metaphone``
both do this; two strings match or they do not, so these are not distances but equivalence
classes. They are the only family that can match `Smith` to `Smyth` at zero cost.

**character overlap** ignores order almost entirely. ``jaro_winkler`` weights a shared
prefix, which suits names; ``bigram`` is Jaccard over character bigrams.

Nothing here is novel. The point of implementing all seven is that they are routinely
compared on one corrupted dataset and ranked, when the ranking is a property of **how the
corruption was generated** - which `corrupt.py` makes explicit and `run.py` measures.
"""

from __future__ import annotations

import re

#: Adjacency on a QWERTY keyboard. Used only by `keyboard`, which is the measure that
#: encodes an error model rather than a notion of distance.
_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_NEIGHBOURS: dict[str, set[str]] = {}
for _r, _row in enumerate(_ROWS):
    for _c, _key in enumerate(_row):
        near = set()
        for _dr, _dc in ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1)):
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < len(_ROWS) and 0 <= _nc < len(_ROWS[_nr]):
                near.add(_ROWS[_nr][_nc])
        _NEIGHBOURS[_key] = near


def levenshtein(a: str, b: str) -> int:
    """Insert, delete, substitute — each costing one.

    Two rows rather than a full matrix: the recurrence only ever reads the previous row, so
    the quadratic memory is not needed and this is run millions of times.
    """
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def damerau(a: str, b: str) -> int:
    """Levenshtein plus transposition of two adjacent characters as a single operation.

    `form` to `from` is one transposition and two substitutions. Charging two for it is
    what makes plain Levenshtein rank a genuine typo below an unrelated word that happens
    to differ by one letter.
    """
    n, m = len(a), len(b)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = a[i - 1] != b[j - 1]
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                table[i][j] = min(table[i][j], table[i - 2][j - 2] + 1)
    return table[n][m]


def keyboard(a: str, b: str, near_cost: float = 0.4) -> float:
    """Levenshtein whose substitution cost depends on key distance.

    Replacing `a` with `s` costs ``near_cost``; replacing it with `p` costs a full unit.
    This is the only measure here that knows anything about *how* text gets corrupted.

    The expectation was that it would win when the corruption is typing. **It does not** -
    it comes fourth of seven on exactly the noise it models. Making near-key substitutions
    cheap forgives the corruption, and it equally forgives every *wrong* candidate that
    differs by a near-key substitution. A metric that encodes the error model buys
    tolerance and pays for it in discrimination, and on a candidate list of two thousand
    words the second cost is larger. Measured, not assumed.
    """
    if len(a) < len(b):
        a, b = b, a
    previous = [float(i) for i in range(len(b) + 1)]
    for i, ca in enumerate(a, start=1):
        current = [float(i)]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                substitution = 0.0
            elif cb in _NEIGHBOURS.get(ca, ()):
                substitution = near_cost
            else:
                substitution = 1.0
            current.append(
                min(previous[j] + 1.0, current[j - 1] + 1.0, previous[j - 1] + substitution)
            )
        previous = current
    return previous[-1]


def jaro(a: str, b: str) -> float:
    if not a or not b:
        return 1.0 if a == b else 0.0
    window = max(len(a), len(b)) // 2 - 1
    window = max(window, 0)

    a_flags = [False] * len(a)
    b_flags = [False] * len(b)
    matches = 0
    for i, ca in enumerate(a):
        for j in range(max(0, i - window), min(len(b), i + window + 1)):
            if not b_flags[j] and b[j] == ca:
                a_flags[i] = b_flags[j] = True
                matches += 1
                break
    if not matches:
        return 0.0

    transpositions = 0
    k = 0
    for i, flagged in enumerate(a_flags):
        if not flagged:
            continue
        while not b_flags[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1

    half = transpositions / 2
    return (matches / len(a) + matches / len(b) + (matches - half) / matches) / 3


def jaro_winkler(a: str, b: str, scale: float = 0.1, max_prefix: int = 4) -> float:
    """Jaro with a bonus for a shared prefix.

    The prefix bonus is why this measure is recommended for names: people mistype the end
    of a name far more often than the beginning. That is itself an error model, built into
    a measure usually presented as general purpose.
    """
    base = jaro(a, b)
    prefix = 0
    for ca, cb in zip(a[:max_prefix], b[:max_prefix]):
        if ca != cb:
            break
        prefix += 1
    return base + prefix * scale * (1 - base)


def soundex(text: str) -> str:
    """The 1918 census algorithm: first letter, then three consonant-class digits.

    Everything after the fourth code is discarded, so `Robert` and `Robertson` collide.
    That is not a defect - it was designed to group surnames on index cards - but it is
    routinely used as a general string matcher, where truncation at four is severe.
    """
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if not cleaned:
        return ""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        **dict.fromkeys("l", "4"),
        **dict.fromkeys("mn", "5"),
        **dict.fromkeys("r", "6"),
    }
    out = cleaned[0].upper()
    previous = codes.get(cleaned[0], "")
    for char in cleaned[1:]:
        code = codes.get(char, "")
        if code and code != previous:
            out += code
            if len(out) == 4:
                break
        if char not in "hw":
            previous = code
    return out.ljust(4, "0")


_METAPHONE_RULES = (
    ("ph", "f"),
    ("gh", "f"),
    ("ck", "k"),
    ("sch", "sk"),
    ("tch", "ch"),
    ("wr", "r"),
    ("kn", "n"),
    ("gn", "n"),
    ("mb", "m"),
    ("ie", "y"),
    ("ei", "y"),
    ("c", "k"),
    ("q", "k"),
    ("x", "ks"),
    ("z", "s"),
)


def metaphone(text: str) -> str:
    """A reduced Metaphone: apply the common digraph rules, then drop non-initial vowels.

    Not the full algorithm, and labelled as such. It keeps the part that matters for this
    comparison - that `ph` and `f`, or `ck` and `k`, become the same code - without the
    hundred context-sensitive cases the original carries.
    """
    cleaned = re.sub(r"[^a-z]", "", text.lower())
    if not cleaned:
        return ""
    for pattern, replacement in _METAPHONE_RULES:
        cleaned = cleaned.replace(pattern, replacement)
    head, tail = cleaned[0], cleaned[1:]
    tail = re.sub(r"[aeiou]", "", tail)
    out = head + tail
    return re.sub(r"(.)\1+", r"\1", out)


def bigram(a: str, b: str) -> float:
    """Jaccard over character bigrams. Order-insensitive beyond the pairs themselves."""
    left = {a[i : i + 2] for i in range(len(a) - 1)} or {a}
    right = {b[i : i + 2] for i in range(len(b) - 1)} or {b}
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _similarity_from_distance(distance: float, a: str, b: str) -> float:
    longest = max(len(a), len(b)) or 1
    return 1.0 - distance / longest


#: Every measure exposed as a similarity in [0, 1], so `run.py` can rank candidates with
#: one code path. The phonetic measures are 1 or 0 by nature - they are equivalence
#: classes, not distances - and that is part of what the comparison shows.
SIMILARITIES = {
    "levenshtein": lambda a, b: _similarity_from_distance(levenshtein(a, b), a, b),
    "damerau": lambda a, b: _similarity_from_distance(damerau(a, b), a, b),
    "keyboard": lambda a, b: _similarity_from_distance(keyboard(a, b), a, b),
    "jaro_winkler": jaro_winkler,
    "bigram": bigram,
    "soundex": lambda a, b: float(soundex(a) == soundex(b)),
    "metaphone": lambda a, b: float(metaphone(a) == metaphone(b)),
}


def similarity(name: str, a: str, b: str) -> float:
    try:
        return SIMILARITIES[name](a, b)
    except KeyError:
        raise ValueError(
            f"unknown measure {name!r}; expected one of {sorted(SIMILARITIES)}"
        ) from None
