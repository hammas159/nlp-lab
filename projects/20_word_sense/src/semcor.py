"""Read SemCor, the sense-tagged corpus, straight from its XML.

SemCor tags each content word with the WordNet sense a human assigned it. The tags look
like this:

    <wf cmd="done" pos="NN" lemma="investigation" wnsn="1" lexsn="1:09:00::">investigation</wf>

The attribute that matters is **`wnsn`** - the WordNet *sense number*, which is a rank.
Sense 1 is WordNet's first-listed sense for that lemma, sense 2 the second, and so on.

That makes the most-frequent-sense baseline computable from this file alone: **always answer
1**. No sense inventory, no glosses, no WordNet installation. It is worth noticing why that
works, because it is slightly circular - WordNet's sense ordering was itself derived from
frequency counts over a tagged corpus, and SemCor is that corpus. `run.py` measures the
circularity rather than ignoring it, by also estimating the most frequent sense from a
training split and comparing the two.

The files are SGML rather than well-formed XML - unescaped `&` and stray attributes appear -
so they are parsed with a regular expression over `<wf .../>` tags rather than with an XML
parser that would reject them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

WF = re.compile(r"<wf\b([^>]*)>(.*?)</wf>", re.DOTALL)
ATTR = re.compile(r'(\w+)="([^"]*)"')
SENTENCE = re.compile(r"<s\b[^>]*>(.*?)</s>", re.DOTALL)


@dataclass(frozen=True)
class Token:
    """One sense-tagged word occurrence."""

    lemma: str
    pos: str
    sense: int
    surface: str
    document: str

    @property
    def key(self) -> tuple[str, str]:
        """A lemma's senses are numbered within its part of speech, so both are the key.

        Keying on the lemma alone would merge the noun and verb senses of `run` into one
        inventory and make every sense number mean two different things.
        """
        return (self.lemma, self.pos)


def data_roots() -> list[Path]:
    if env := os.environ.get("NLTK_DATA"):
        return [Path(env)]
    return [
        Path.home() / "nltk_data",
        Path.home() / "AppData" / "Roaming" / "nltk_data",
    ]


def find_semcor() -> Path | None:
    for root in data_roots():
        candidate = root / "corpora" / "semcor"
        if candidate.is_dir():
            return candidate
    return None


def parse_sentence(block: str, document: str) -> list[Token]:
    out = []
    for attributes, surface in WF.findall(block):
        fields = dict(ATTR.findall(attributes))
        if fields.get("cmd") != "done":
            continue
        lemma = fields.get("lemma")
        wnsn = fields.get("wnsn")
        if not lemma or not wnsn:
            continue
        # A few tokens carry several senses ("1;2") where the annotator allowed more than
        # one reading. The first is taken, and the count is reported by `load`.
        first = wnsn.split(";")[0].strip()
        if not first.isdigit() or int(first) < 1:
            continue
        out.append(
            Token(
                lemma=lemma.lower(),
                pos=fields.get("pos", ""),
                sense=int(first),
                surface=surface.strip(),
                document=document,
            )
        )
    return out


def load(limit_files: int | None = None) -> tuple[list[list[Token]], dict[str, int]]:
    """Return sentences (as lists of tagged tokens) and a few corpus statistics."""
    root = find_semcor()
    if root is None:
        raise LookupError(
            "SemCor not found. Fetch it with:\n  python -c \"import nltk; nltk.download('semcor')\""
        )
    files = sorted(root.glob("*/tagfiles/*"))
    if limit_files:
        files = files[:limit_files]
    if not files:
        raise LookupError(f"SemCor directory {root} contains no tagfiles")

    sentences: list[list[Token]] = []
    multi = 0
    for path in files:
        text = path.read_text(encoding="latin-1", errors="replace")
        document = path.stem
        for block in SENTENCE.findall(text):
            multi += sum(
                1 for a, _ in WF.findall(block) if ";" in dict(ATTR.findall(a)).get("wnsn", "")
            )
            tokens = parse_sentence(block, document)
            if tokens:
                sentences.append(tokens)
    stats = {
        "files": len(files),
        "sentences": len(sentences),
        "tagged_tokens": sum(len(s) for s in sentences),
        "multi_sense_tags": multi,
    }
    return sentences, stats


def inventory(sentences: list[list[Token]]) -> dict[tuple[str, str], set[int]]:
    """Which senses each (lemma, pos) is ever tagged with in this corpus.

    This is the *observed* inventory, not WordNet's. It is a lower bound on polysemy - a
    sense that never occurs in SemCor is invisible here - and that matters, because a random
    baseline over the observed inventory is stronger than a random baseline over the real
    one. Reported as a limitation rather than smoothed over.
    """
    out: dict[tuple[str, str], set[int]] = {}
    for sentence in sentences:
        for token in sentence:
            out.setdefault(token.key, set()).add(token.sense)
    return out


def polysemous(
    sentences: list[list[Token]], senses: dict[tuple[str, str], set[int]]
) -> list[Token]:
    """Only the tokens whose lemma has more than one observed sense.

    A word with one sense is not a disambiguation problem: every method scores it correctly,
    and including such tokens inflates every system equally while compressing the differences
    that the evaluation exists to show.
    """
    return [t for s in sentences for t in s if len(senses.get(t.key, ())) > 1]
