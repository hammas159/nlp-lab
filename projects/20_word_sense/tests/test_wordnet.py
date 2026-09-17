"""Tests for the WordNet reader and Lesk.

No WordNet database and no network: a miniature `index.<pos>` / `data.<pos>` pair is written
here in the real format. The format is the whole risk - the offsets in an index line sit
after a variable-length run of pointer symbols, so an off-by-one in that arithmetic yields
plausible glosses for the wrong senses and a silently wrong result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import disambiguate as D
import semcor as S
import wordnet as W

# `lemma pos synset_cnt p_cnt <p_cnt symbols> sense_cnt tagsense_cnt offset...`
# `bank` has two senses with three pointer symbols between the counts and the offsets.
INDEX_NOUN = """  1 This header line is ignored
bank n 2 3 @ ~ + 2 1 00000001 00000002
dog n 1 1 @ 1 1 00000003
"""

DATA_NOUN = """  1 header
00000001 09 n 01 bank 0 001 @ 00000009 n 0000 | sloping land beside a river or lake
00000002 09 n 01 bank 0 001 @ 00000009 n 0000 | a financial institution that lends money
00000003 05 n 01 dog 0 001 @ 00000009 n 0000 | a domesticated carnivorous mammal
"""


@pytest.fixture
def wordnet(tmp_path, monkeypatch):
    monkeypatch.setenv("NLTK_DATA", str(tmp_path))
    root = tmp_path / "corpora" / "wordnet"
    root.mkdir(parents=True)
    (root / "index.noun").write_text(INDEX_NOUN, encoding="latin-1")
    (root / "data.noun").write_text(DATA_NOUN, encoding="latin-1")
    for name in ("verb", "adj", "adv"):
        (root / f"index.{name}").write_text("  header\n", encoding="latin-1")
        (root / f"data.{name}").write_text("  header\n", encoding="latin-1")
    W.load.cache_clear()
    yield tmp_path
    W.load.cache_clear()


# --- part of speech folding -------------------------------------------------------------


@pytest.mark.parametrize(
    ("penn", "expected"),
    [("NN", "n"), ("NNS", "n"), ("NNP", "n"), ("VB", "v"), ("VBD", "v"), ("JJ", "a"), ("RB", "r")],
)
def test_penn_tags_fold_to_wordnet_parts_of_speech(penn, expected):
    assert W.fold_pos(penn) == expected


def test_a_tag_with_no_wordnet_entry_folds_to_nothing():
    """Determiners and prepositions have no WordNet entry; returning a part of speech for
    them would send the lookup hunting through the noun index."""
    assert W.fold_pos("DT") is None
    assert W.fold_pos("IN") is None


# --- the index format -------------------------------------------------------------------


def test_offsets_are_read_from_after_the_pointer_symbols(wordnet):
    """The arithmetic that matters. `bank` lists three pointer symbols between its counts
    and its offsets; counting from the wrong place returns a pointer symbol as an offset,
    or the sense_cnt as one, and every gloss after it is for the wrong synset."""
    senses, _ = W.load()
    assert senses[("bank", "n")] == [1, 2]


def test_senses_are_in_wordnet_order(wordnet):
    """Sense 1 is the first offset. This is the property that lets SemCor's `wnsn` index
    straight into the list with no sense-key lookup at all."""
    assert W.gloss("bank", "NN", 1).startswith("sloping land")
    assert W.gloss("bank", "NN", 2).startswith("a financial institution")


def test_header_lines_are_skipped(wordnet):
    senses, _ = W.load()
    assert ("1", "n") not in senses
    assert len(senses) == 2


def test_a_lemma_with_one_pointer_symbol_also_parses(wordnet):
    assert W.gloss("dog", "NN", 1).startswith("a domesticated")


# --- missing and out-of-range lookups ----------------------------------------------------


def test_an_unknown_lemma_returns_no_gloss(wordnet):
    """SemCor tags proper nouns and multiword expressions WordNet does not index under the
    same string. A disambiguator must fall back on those, not crash."""
    assert W.gloss("zzzz", "NN", 1) == ""


def test_a_sense_beyond_the_inventory_returns_no_gloss(wordnet):
    assert W.gloss("bank", "NN", 99) == ""


def test_a_part_of_speech_wordnet_does_not_have_returns_no_gloss(wordnet):
    assert W.gloss("bank", "DT", 1) == ""


def test_a_missing_database_raises_with_the_fetch_command(tmp_path, monkeypatch):
    monkeypatch.setenv("NLTK_DATA", str(tmp_path))
    W.load.cache_clear()
    with pytest.raises(LookupError, match="nltk.download"):
        W.load()
    W.load.cache_clear()


def test_available_reports_whether_the_database_is_there(wordnet, tmp_path, monkeypatch):
    assert W.available()
    monkeypatch.setenv("NLTK_DATA", str(tmp_path / "nowhere"))
    assert not W.available()


# --- Lesk ---------------------------------------------------------------------------------


def token(lemma="bank", pos="NN", sense=1, surface="bank", document="d1"):
    return S.Token(lemma=lemma, pos=pos, sense=sense, surface=surface, document=document)


def test_lesk_picks_the_sense_whose_gloss_matches_the_context(wordnet):
    """The whole method: `bank` beside `river` should take the sloping-land sense, whose
    gloss contains `river`."""
    model = D.Lesk().fit([[token()]], {("bank", "NN"): {1, 2}})
    target = token()
    sentence = [target, token(lemma="river", surface="river")]
    assert model.predict(target, sentence) == 1


def test_lesk_picks_the_other_sense_for_the_other_context(wordnet):
    """`money` appears in the financial gloss and not in the geographic one. If this and the
    previous test do not disagree, the method is not reading the gloss at all."""
    model = D.Lesk().fit([[token()]], {("bank", "NN"): {1, 2}})
    target = token()
    sentence = [target, token(lemma="money", surface="money")]
    assert model.predict(target, sentence) == 2


def test_lesk_ignores_the_target_word_itself(wordnet):
    """`bank` is in both glosses, so leaving it in the context adds the same constant to
    every sense and cannot discriminate - but it can create a spurious tie."""
    model = D.Lesk().fit([[token()]], {("bank", "NN"): {1, 2}})
    target = token()
    assert model.predict(target, [target]) in {1, 2}


def test_lesk_falls_back_when_no_gloss_overlaps(wordnet):
    """With no overlap there is no evidence, and an arbitrary pick would make the score
    depend on dictionary ordering rather than on the method."""
    model = D.Lesk().fit([[token()]], {("bank", "NN"): {1, 2}})
    target = token()
    sentence = [target, token(lemma="qqqq", surface="qqqq")]
    assert model.predict(target, sentence) in {1, 2}


def test_lesk_falls_back_for_a_lemma_wordnet_does_not_have(wordnet):
    model = D.Lesk().fit([[token()]], {("zzzz", "NN"): {1, 2}})
    target = token(lemma="zzzz", surface="zzzz")
    assert model.predict(target, [target]) == 1


def test_lesk_is_listed_only_when_wordnet_is_present(wordnet, tmp_path, monkeypatch):
    """A method that silently degraded to its fallback would still be scored under its own
    name, which would report a number for Lesk that Lesk did not produce."""
    assert D.Lesk in D.methods()
    monkeypatch.setenv("NLTK_DATA", str(tmp_path / "nowhere"))
    assert D.Lesk not in D.methods()


def test_the_word_bag_is_lowercase_words_only(wordnet):
    assert W.bag("The Bank's money!") == {"the", "bank", "s", "money"}
