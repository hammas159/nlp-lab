"""Two error models, stated explicitly, because the error model is what decides the answer.

A string metric is usually evaluated by corrupting a word list and seeing which measure
recovers the originals. The corruption is then described in a sentence and the ranking is
reported as a property of the measures. It is not - it is a property of the pair.

``typing``    keyboard-adjacent substitution, adjacent transposition, deletion and
              doubling. What a person does at speed on a keyboard.
``phonetic``  substitutions that preserve pronunciation: ph/f, c/k, ie/y, doubled
              consonants, silent-letter loss. What a person does writing down a name they
              have only heard.

Neither is a claim about real-world error rates. They are two clearly different
distributions, and the experiment is whether the ranking of measures survives swapping
one for the other.
"""

from __future__ import annotations

import numpy as np

_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_NEIGHBOURS: dict[str, list[str]] = {}
for _r, _row in enumerate(_ROWS):
    for _c, _key in enumerate(_row):
        near = []
        for _dr, _dc in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            _nr, _nc = _r + _dr, _c + _dc
            if 0 <= _nr < len(_ROWS) and 0 <= _nc < len(_ROWS[_nr]):
                near.append(_ROWS[_nr][_nc])
        _NEIGHBOURS[_key] = near

#: Pronunciation-preserving rewrites, applied in either direction.
_PHONETIC_PAIRS = (
    ("ph", "f"),
    ("c", "k"),
    ("s", "z"),
    ("ie", "y"),
    ("ei", "ai"),
    ("ck", "k"),
    ("qu", "kw"),
    ("x", "ks"),
    ("gh", ""),
    ("e", ""),
)


def typing(word: str, rng: np.random.Generator, edits: int = 1) -> str:
    """Apply ``edits`` keyboard-style corruptions."""
    out = word
    for _ in range(edits):
        if len(out) < 2:
            return out
        choice = rng.integers(4)
        position = int(rng.integers(len(out) - 1))
        if choice == 0:  # keyboard-adjacent substitution
            near = _NEIGHBOURS.get(out[position])
            if near:
                out = out[:position] + near[int(rng.integers(len(near)))] + out[position + 1 :]
        elif choice == 1:  # transpose adjacent
            out = out[:position] + out[position + 1] + out[position] + out[position + 2 :]
        elif choice == 2:  # delete
            out = out[:position] + out[position + 1 :]
        else:  # double a character
            out = out[: position + 1] + out[position] + out[position + 1 :]
    return out


def phonetic(word: str, rng: np.random.Generator, edits: int = 1) -> str:
    """Apply ``edits`` pronunciation-preserving rewrites.

    Only rewrites that actually apply to the word are considered, so a word with no
    applicable rule is returned unchanged rather than corrupted by a rule that does not fit.
    That is the honest behaviour and it means the two models corrupt different *numbers* of
    words, which `run.py` reports.
    """
    out = word
    for _ in range(edits):
        applicable = []
        for left, right in _PHONETIC_PAIRS:
            # A rewrite that would empty the string is not applicable. Without this an
            # "ee" corrupted three times becomes "", and an empty query scores equally
            # against every candidate - a silent way to make a measure look bad on words
            # it never saw.
            if left and left in out and out.replace(left, right, 1):
                applicable.append((left, right))
            if right and right in out and out.replace(right, left, 1):
                applicable.append((right, left))
        if not applicable:
            break
        source, target = applicable[int(rng.integers(len(applicable)))]
        index = out.find(source)
        out = out[:index] + target + out[index + len(source) :]
    return out or word


MODELS = {"typing": typing, "phonetic": phonetic}


def corrupt(model: str, word: str, rng: np.random.Generator, edits: int = 1) -> str:
    try:
        return MODELS[model](word, rng, edits)
    except KeyError:
        raise ValueError(
            f"unknown error model {model!r}; expected one of {sorted(MODELS)}"
        ) from None
