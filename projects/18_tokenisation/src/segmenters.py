"""BPE, WordPiece and Unigram trained on the same corpus at the same vocabulary size.

The three dominant subword algorithms differ in how they *choose* a vocabulary:

``BPE``        starts from characters and repeatedly merges the most frequent adjacent pair
               (Sennrich, Haddow & Birch, ACL 2016). Greedy and bottom-up: a merge made
               early can never be reconsidered.
``WordPiece``  the same bottom-up shape, but merges the pair that most increases the
               likelihood of the training data rather than the most frequent pair
               (Schuster & Nakajima, 2012).
``Unigram``    top-down. Starts from a large candidate vocabulary and prunes the pieces
               whose removal costs the least likelihood, keeping a probabilistic model over
               segmentations (Kudo, ACL 2018). Bostrom & Durrett (Findings of EMNLP 2020)
               found its pieces align more closely with morphology than BPE's, and argued
               BPE's greedy construction is the reason.

Everything except the algorithm is pinned: same corpus, same target vocabulary size, same
lowercase normaliser, same whitespace pre-tokenizer. A comparison where one tokenizer gets a
bigger vocabulary or different normalisation is measuring the configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

SPECIALS = ["[UNK]"]


@dataclass
class Segmenter:
    """A trained tokenizer plus the name of the algorithm that produced it."""

    name: str
    tokenizer: object

    def encode(self, text: str) -> list[str]:
        return self.tokenizer.encode(text).tokens

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def _base(model, normalizer_lowercase: bool = True):
    from tokenizers import Tokenizer, normalizers, pre_tokenizers

    tokenizer = Tokenizer(model)
    # Lowercase, then split on whitespace and punctuation. Identical for all three, so the
    # only thing that differs downstream is which subword vocabulary was learned.
    steps = [normalizers.NFD(), normalizers.StripAccents()]
    if normalizer_lowercase:
        steps.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(steps)
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Whitespace(), pre_tokenizers.Digits(individual_digits=False)]
    )
    return tokenizer


def train_bpe(texts: list[str], vocabulary_size: int) -> Segmenter:
    from tokenizers import models, trainers

    tokenizer = _base(models.BPE(unk_token="[UNK]"))
    tokenizer.train_from_iterator(
        texts,
        trainers.BpeTrainer(
            vocab_size=vocabulary_size, special_tokens=SPECIALS, show_progress=False
        ),
    )
    return Segmenter("BPE", tokenizer)


def train_wordpiece(texts: list[str], vocabulary_size: int) -> Segmenter:
    from tokenizers import decoders, models, trainers

    tokenizer = _base(models.WordPiece(unk_token="[UNK]", max_input_chars_per_word=200))
    tokenizer.train_from_iterator(
        texts,
        trainers.WordPieceTrainer(
            vocab_size=vocabulary_size, special_tokens=SPECIALS, show_progress=False
        ),
    )
    tokenizer.decoder = decoders.WordPiece()
    return Segmenter("WordPiece", tokenizer)


def train_unigram(texts: list[str], vocabulary_size: int) -> Segmenter:
    from tokenizers import models, trainers

    tokenizer = _base(models.Unigram())
    tokenizer.train_from_iterator(
        texts,
        trainers.UnigramTrainer(
            vocab_size=vocabulary_size,
            special_tokens=SPECIALS,
            unk_token="[UNK]",
            show_progress=False,
        ),
    )
    return Segmenter("Unigram", tokenizer)


TRAINERS = {"BPE": train_bpe, "WordPiece": train_wordpiece, "Unigram": train_unigram}


def train(name: str, texts: list[str], vocabulary_size: int) -> Segmenter:
    try:
        return TRAINERS[name](texts, vocabulary_size)
    except KeyError:
        raise ValueError(
            f"unknown tokenizer {name!r}; expected one of {sorted(TRAINERS)}"
        ) from None


# --- intrinsic measurements ----------------------------------------------------------------
#
# These are what tokenizer papers report. The project exists to check whether they predict
# anything about a task.


def strip_marker(piece: str) -> str:
    """WordPiece marks continuations with `##`; BPE and Unigram do not. Comparing raw
    strings across algorithms without stripping it would make every WordPiece continuation
    look like a different piece from the identical BPE one."""
    return piece.removeprefix("##")


def fertility(segmenter: Segmenter, words: list[str]) -> float:
    """Mean number of pieces per word. 1.0 means nothing was ever split."""
    if not words:
        return 0.0
    return sum(len(segmenter.encode(w)) for w in words) / len(words)


def intact_rate(segmenter: Segmenter, words: list[str]) -> float:
    """Share of words the tokenizer leaves as a single piece."""
    if not words:
        return 0.0
    return sum(len(segmenter.encode(w)) == 1 for w in words) / len(words)


def unknown_rate(segmenter: Segmenter, words: list[str]) -> float:
    """Share of words that come back as `[UNK]` rather than as pieces.

    The three algorithms do not fail the same way, and it is not a detail. BPE and Unigram
    fall back to characters, so a word they have never seen still yields pieces that carry
    some of its identity. **WordPiece is greedy longest-match-first and gives up on the
    whole word if any prefix cannot be matched**, emitting one `[UNK]`.

    For a lexical retriever that is the difference between a rare word being partially
    matchable and being erased, so it has to be measured before any accuracy gap between
    the algorithms can be attributed to the vocabulary they learned.
    """
    if not words:
        return 0.0
    return sum("[UNK]" in segmenter.encode(w) for w in words) / len(words)


def segmentation_agreement(a: Segmenter, b: Segmenter, words: list[str]) -> float:
    """Share of words two tokenizers segment identically, ignoring continuation markers."""
    if not words:
        return 0.0
    same = 0
    for word in words:
        pieces_a = [strip_marker(p) for p in a.encode(word)]
        pieces_b = [strip_marker(p) for p in b.encode(word)]
        same += pieces_a == pieces_b
    return same / len(words)
