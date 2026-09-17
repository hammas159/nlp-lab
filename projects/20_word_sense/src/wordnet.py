"""Just enough WordNet to run Lesk: sense-ordered synsets, and their glosses.

Read from the distributed database files rather than through a library, for the same reason
as everywhere else in this lab - the format is simple, and the one decision that matters
becomes visible instead of inherited.

Two files do all the work:

``index.<pos>``   ``lemma pos synset_cnt p_cnt [ptrs...] sense_cnt tagsense_cnt off1 off2 ...``
                  The offsets are listed **in sense order**, so `offsets[n - 1]` is the
                  synset for sense *n*. That is exactly what SemCor's `wnsn` indexes into,
                  which is what makes this work without any sense-key plumbing.
``data.<pos>``    one synset per line, keyed by byte offset, with the gloss after ``|``.

SemCor carries Penn tags (`NN`, `VBD`, `JJ`) and WordNet uses four coarse ones, so the tags
are folded: anything starting `NN` is a noun, `VB` a verb, `JJ` an adjective, `RB` an adverb.
A tag that folds to nothing has no WordNet entry and its token is simply left to the caller.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

WORD = re.compile(r"[a-z]+")

#: WordNet's four open-class parts of speech. `a` also covers satellite adjectives (`s`).
POS_FILES = {"n": "noun", "v": "verb", "a": "adj", "r": "adv"}


def fold_pos(penn: str) -> str | None:
    """Penn tag to WordNet part of speech, or None if WordNet has no entry for it."""
    if penn.startswith("NN"):
        return "n"
    if penn.startswith("VB"):
        return "v"
    if penn.startswith("JJ"):
        return "a"
    if penn.startswith("RB"):
        return "r"
    return None


def data_roots() -> list[Path]:
    if env := os.environ.get("NLTK_DATA"):
        return [Path(env)]
    return [
        Path.home() / "nltk_data",
        Path.home() / "AppData" / "Roaming" / "nltk_data",
    ]


def find_wordnet() -> Path | None:
    for root in data_roots():
        candidate = root / "corpora" / "wordnet"
        if (candidate / "index.noun").exists():
            return candidate
    return None


def available() -> bool:
    return find_wordnet() is not None


def _require() -> Path:
    root = find_wordnet()
    if root is None:
        raise LookupError(
            "WordNet not found. Fetch it with:\n"
            "  python -c \"import nltk; nltk.download('wordnet')\""
        )
    return root


@lru_cache(maxsize=1)
def load() -> tuple[dict[tuple[str, str], list[int]], dict[tuple[str, int], str]]:
    """Return (sense-ordered offsets per lemma, gloss per synset).

    Cached, because it reads about 30MB and every disambiguator wants the same tables.
    """
    root = _require()
    senses: dict[tuple[str, str], list[int]] = {}
    glosses: dict[tuple[str, int], str] = {}

    for pos, name in POS_FILES.items():
        index_path = root / f"index.{name}"
        for line in index_path.read_text(encoding="latin-1").splitlines():
            if line.startswith("  ") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            lemma = parts[0]
            try:
                pointer_count = int(parts[3])
                # lemma pos synset_cnt p_cnt <p_cnt symbols> sense_cnt tagsense_cnt offsets...
                offsets = [int(x) for x in parts[4 + pointer_count + 2 :]]
            except (ValueError, IndexError):
                continue
            if offsets:
                senses[(lemma.lower(), pos)] = offsets

        data_path = root / f"data.{name}"
        for line in data_path.read_text(encoding="latin-1").splitlines():
            if line.startswith("  ") or "|" not in line:
                continue
            offset_text = line.split(" ", 1)[0]
            if not offset_text.isdigit():
                continue
            glosses[(pos, int(offset_text))] = line.split("|", 1)[1].strip()

    return senses, glosses


def gloss(lemma: str, penn_pos: str, sense: int) -> str:
    """The gloss of a lemma's nth sense, or "" when WordNet has no such entry.

    Returns empty rather than raising: SemCor tags multiword expressions and proper nouns
    that WordNet does not index under the same string, and a disambiguator should fall back
    on those rather than crash on them.
    """
    pos = fold_pos(penn_pos)
    if pos is None:
        return ""
    senses, glosses = load()
    offsets = senses.get((lemma.lower(), pos))
    if not offsets or sense < 1 or sense > len(offsets):
        return ""
    return glosses.get((pos, offsets[sense - 1]), "")


def bag(text: str) -> set[str]:
    """Lowercase word set - what Lesk overlaps."""
    return set(WORD.findall(text.lower()))
