"""Three registers, one tokenizer, from what is already in the Hugging Face cache.

The registers are English encyclopaedic prose (HotpotQA paragraphs), C source
(Devign functions) and Python source (MBPP + HumanEval). They are here because a
vocabulary-growth exponent measured on prose is routinely applied to code, and the two
are not obviously the same process.

**One tokenizer for all three.** A code-aware tokenizer that keeps braces and an English
tokenizer that drops punctuation would make every cross-register difference a difference
in preprocessing. The shared tokenizer is deliberately plain - maximal runs of word
characters, case-folded - and `code_tokenize` exists only to measure how much that choice
moved the answer, reported as a sensitivity column rather than as the result.

Corpus sizes differ by two orders of magnitude, and the Heaps exponent is known to depend
on corpus size, so every cross-register comparison is made at a **matched token count**.
That is the whole reason `truncate` exists.
"""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

HUB = Path.home() / ".cache" / "huggingface" / "hub"

#: Case-folded runs of word characters. Applied identically to prose and to source code.
WORD = re.compile(r"[a-z0-9_]+")

#: Identifiers, numbers and individual punctuation marks, case preserved. Used only for
#: the sensitivity column - in C, `{` and `;` carry structure that `WORD` discards.
CODE = re.compile(r"[A-Za-z_]\w*|\d+|[^\s\w]")


def tokenize(text: str) -> list[str]:
    return WORD.findall(text.lower())


def code_tokenize(text: str) -> list[str]:
    return CODE.findall(text)


@dataclass(frozen=True)
class Corpus:
    name: str
    register: str
    tokens: list[str]

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def types(self) -> int:
        return len(set(self.tokens))

    def truncate(self, n: int) -> Corpus:
        return Corpus(self.name, self.register, self.tokens[:n])


def _snapshot(dataset: str, pattern: str) -> Path | None:
    hits = sorted(
        glob.glob(str(HUB / dataset / "snapshots" / "*" / "**" / pattern), recursive=True)
    )
    return Path(hits[0]) if hits else None


def _require(path: Path | None, dataset: str, how: str) -> Path:
    if path is None:
        raise FileNotFoundError(
            f"{dataset} not found in the Hugging Face cache. Fetch it with:\n  {how}\n"
            f"Looked under: {HUB}"
        )
    return path


def _prose_texts() -> list[str]:
    """HotpotQA distractor paragraphs, deduplicated by title.

    Paragraph order is the dataset's own. It is not shuffled: Heaps' law is measured over
    a *prefix* of the stream, and shuffling would impose an artificial homogeneity that
    real text does not have.
    """
    import pandas as pd

    path = _require(
        _snapshot("datasets--hotpotqa--hotpot_qa", "validation-*.parquet"),
        "HotpotQA (distractor)",
        'python -c "from datasets import load_dataset; '
        "load_dataset('hotpotqa/hotpot_qa','distractor')\"",
    )
    frame = pd.read_parquet(path)
    seen: dict[str, str] = {}
    for context in frame["context"]:
        for title, sentences in zip(list(context["title"]), list(context["sentences"])):
            seen.setdefault(str(title), " ".join(str(s) for s in sentences).strip())
    return list(seen.values())


def _c_texts() -> list[str]:
    """Devign C functions, test and validation splits.

    The train split has never been obtainable on this machine - the same 17.85 MB download
    that blocked `devign-leakage` and project 04.
    """
    import pandas as pd

    texts: list[str] = []
    for split in ("test", "validation"):
        path = _snapshot("datasets--google--code_x_glue_cc_defect_detection", f"{split}-*.parquet")
        if path is not None:
            texts.extend(str(f) for f in pd.read_parquet(path)["func"])
    if not texts:
        raise FileNotFoundError(
            "Devign not found in the Hugging Face cache. Fetch it with:\n"
            '  python -c "from datasets import load_dataset; '
            "load_dataset('google/code_x_glue_cc_defect_detection')\""
        )
    return texts


def _python_texts() -> list[str]:
    """MBPP reference solutions plus HumanEval prompts and canonical solutions.

    Two datasets are concatenated because either alone is too small to fit a growth curve
    to. Even together this corpus is ~1.3% the size of the prose one, which is the point
    of the finite-size section rather than a defect to apologise for.
    """
    import pandas as pd

    texts: list[str] = []

    mbpp = _snapshot("datasets--Muennighoff--mbpp", "mbpp.jsonl")
    if mbpp is not None:
        for line in mbpp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                texts.append(str(json.loads(line).get("code", "")))

    human = _snapshot("datasets--openai--openai_humaneval", "test-*.parquet")
    if human is not None:
        frame = pd.read_parquet(human)
        texts.extend(f"{p}{s}" for p, s in zip(frame["prompt"], frame["canonical_solution"]))

    if not texts:
        raise FileNotFoundError(
            "Neither MBPP nor HumanEval found in the Hugging Face cache. Fetch with:\n"
            '  python -c "from datasets import load_dataset; '
            "load_dataset('Muennighoff/mbpp'); load_dataset('openai/openai_humaneval')\""
        )
    return texts


#: key -> (display name, register, reader). Readers return raw text, not tokens, so that
#: section 7 can re-tokenize the same bytes under a second tokenizer without paying for
#: the parquet read again.
SOURCES = {
    "prose": ("HotpotQA paragraphs", "English prose", _prose_texts),
    "c": ("Devign C functions", "C source", _c_texts),
    "python": ("MBPP + HumanEval", "Python source", _python_texts),
}

_TEXT_CACHE: dict[str, list[str]] = {}


def raw_texts(key: str) -> list[str]:
    if key not in _TEXT_CACHE:
        _TEXT_CACHE[key] = SOURCES[key][2]()
    return _TEXT_CACHE[key]


def load(key: str, tok=tokenize) -> Corpus:
    name, register, _ = SOURCES[key]
    tokens: list[str] = []
    for text in raw_texts(key):
        tokens.extend(tok(text))
    return Corpus(name, register, tokens)


def load_all(tok=tokenize) -> dict[str, Corpus]:
    return {key: load(key, tok) for key in SOURCES}


if __name__ == "__main__":
    for key, corpus in load_all().items():
        print(
            f"  {key:8} {corpus.name:24} {len(corpus):>10,} tokens  "
            f"{corpus.types:>8,} types  ttr={corpus.types / len(corpus):.4f}"
        )
