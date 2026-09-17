"""Four sentiment lexicons, loaded into one shape so that only the lexicon differs.

They are not four versions of the same object. They disagree about what a lexicon *is*:

``VADER``        7,500 entries with human-rated valence from -4 to +4, built for social media
                 (Hutto & Gilbert, ICWSM 2014). The only one whose authors also shipped a
                 rule set, which is the whole reason this project exists.
``Opinion``      6,800 words in two flat lists, positive and negative, with no magnitude at
                 all (Hu & Liu, KDD 2004). Every word counts the same.
``SentiWordNet`` scores every WordNet *synset*, not every word (Baccianella et al., LREC
                 2010). Getting a word-level score means choosing how to collapse senses -
                 a decision the other three never have to make.
``AFINN``        2,477 words rated -5 to +5 by one person (Nielsen, 2011). The smallest and
                 the bluntest.

**These are read straight from their distributed text files** rather than through a library.
They are plain TSV and word lists; parsing them here removes a dependency, makes the
sense-collapsing decision visible instead of inherited, and keeps the tests runnable with no
corpus present at all.

**Everything is rescaled to [-1, 1].** Classification is by the sign of a sum, and dividing a
lexicon by a positive constant cannot change that sign, so the rescaling is free. It matters
for the intensifier rules, where a multiplier has to mean the same thing whether the
underlying scale ran to 4, to 1, or to 5.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def data_roots() -> list[Path]:
    """Where to look for corpus files.

    `NLTK_DATA` is an **override**, not an addition: if it is set, only it is searched. An
    explicit location that silently falls back to the home directory is worse than useless -
    it makes a run that was meant to be isolated quietly pick up whatever happened to be
    installed, which is exactly how a test suite stops testing what it claims to.
    """
    if env := os.environ.get("NLTK_DATA"):
        return [Path(env)]
    return [
        Path.home() / "nltk_data",
        Path.home() / "AppData" / "Roaming" / "nltk_data",
    ]


def find(*relative: str) -> Path | None:
    for root in data_roots():
        candidate = root.joinpath(*relative)
        if candidate.exists():
            return candidate
    return None


class MissingResource(LookupError):
    """Raised when a lexicon's file is not on disk. Carries the command that fetches it."""


def require(*relative: str) -> Path:
    path = find(*relative)
    if path is None:
        name = relative[-1]
        raise MissingResource(
            f"{name} not found. Fetch it with:\n"
            f"  python -c \"import nltk; nltk.download('{relative[1]}')\""
        )
    return path


@dataclass
class Lexicon:
    name: str
    polarity: dict[str, float] = field(repr=False)
    note: str = ""

    def __contains__(self, word: str) -> bool:
        return word in self.polarity

    def get(self, word: str, default: float = 0.0) -> float:
        return self.polarity.get(word, default)

    def __len__(self) -> int:
        return len(self.polarity)


def rescale(polarity: dict[str, float]) -> dict[str, float]:
    """Divide by the largest magnitude so every lexicon lives on [-1, 1]."""
    if not polarity:
        return {}
    largest = max(abs(v) for v in polarity.values()) or 1.0
    return {w: v / largest for w, v in polarity.items()}


def load_vader() -> Lexicon:
    """`word \\t mean-valence \\t stdev \\t [raw ratings]`, tab separated."""
    path = require("sentiment", "vader_lexicon", "vader_lexicon.txt")
    raw = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            raw[parts[0].lower()] = float(parts[1])
        except ValueError:
            continue
    return Lexicon("VADER", rescale(raw), "human-rated valence, -4 to +4")


def load_opinion() -> Lexicon:
    """Two word lists. Comment lines start with `;`, and the files are Latin-1."""
    positive = require("corpora", "opinion_lexicon", "positive-words.txt")
    negative = require("corpora", "opinion_lexicon", "negative-words.txt")
    raw: dict[str, float] = {}
    for path, value in ((positive, 1.0), (negative, -1.0)):
        for line in path.read_text(encoding="latin-1").splitlines():
            word = line.strip().lower()
            if word and not word.startswith(";"):
                raw[word] = value
    return Lexicon("Opinion", rescale(raw), "two flat word lists, no magnitude")


def load_sentiwordnet(max_senses: int = 3) -> Lexicon:
    """Collapse per-synset scores into per-word scores.

    SentiWordNet scores senses, so a word-level lexicon has to pick a rule. The usual choice
    is the first sense alone; averaging the first few is steadier and is what is used here.
    Either way it is a **decision**, not a reading of the resource - and it is why two papers
    that both say "we used SentiWordNet" can report different numbers.

    `SynsetTerms` are written `word#rank`, where rank 1 is the most frequent sense of that
    word, so the ranks give the sense order directly.
    """
    path = require("corpora", "sentiwordnet", "SentiWordNet_3.0.0.txt")
    buckets: dict[str, list[tuple[int, float]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        try:
            score = float(parts[2]) - float(parts[3])
        except ValueError:
            continue
        if score == 0.0:
            continue
        for term in parts[4].split():
            word, _, rank = term.rpartition("#")
            if not word or not rank.isdigit():
                continue
            buckets.setdefault(word.replace("_", " ").lower(), []).append((int(rank), score))
    raw = {}
    for word, senses in buckets.items():
        chosen = [s for _, s in sorted(senses)[:max_senses]]
        if chosen:
            raw[word] = sum(chosen) / len(chosen)
    return Lexicon("SentiWordNet", rescale(raw), f"mean of first {max_senses} senses")


def load_afinn() -> Lexicon:
    """AFINN ships as `word \\t score`. If the file is absent the lexicon is skipped rather
    than approximated - inventing a lexicon to compare against other lexicons would measure
    the invention."""
    path = find("corpora", "afinn", "AFINN-111.txt") or find("AFINN-111.txt")
    if path is None:
        raise MissingResource("AFINN-111.txt not found; see the README for where to put it")
    raw = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        word, _, value = line.rpartition("\t")
        try:
            raw[word.strip().lower()] = float(value)
        except ValueError:
            continue
    return Lexicon("AFINN", rescale(raw), "one rater, -5 to +5")


LOADERS = {
    "VADER": load_vader,
    "Opinion": load_opinion,
    "SentiWordNet": load_sentiwordnet,
    "AFINN": load_afinn,
}


def load_all(verbose: bool = True) -> list[Lexicon]:
    """Load every lexicon that is present, and say plainly which are not."""
    out = []
    for name, loader in LOADERS.items():
        try:
            out.append(loader())
        except (ImportError, LookupError) as exc:
            if verbose:
                print(f"  SKIPPED {name}: {str(exc).splitlines()[0][:80]}")
    return out


def agreement(a: Lexicon, b: Lexicon) -> tuple[float, int]:
    """Share of shared words on which two lexicons agree in sign, and how many are shared.

    Reported because "which lexicon" is only an interesting question if they disagree.
    """
    shared = set(a.polarity) & set(b.polarity)
    if not shared:
        return 0.0, 0
    same = sum(1 for w in shared if (a.get(w) > 0) == (b.get(w) > 0))
    return same / len(shared), len(shared)
