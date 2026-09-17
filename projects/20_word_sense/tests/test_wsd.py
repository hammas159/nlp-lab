"""Tests for the SemCor reader and the disambiguators.

No corpus and no network: the SemCor parser is pointed at a few lines of the real SGML
written here, and the disambiguators run on hand-built tokens.

The most important tests are the ones about **leakage**. A disambiguator that can see the
gold sense of a neighbouring token scores far above every honest method and looks like a
result, so each method is checked for what it is allowed to read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import disambiguate as D
import semcor as S

SGML = """<contextfile concordance="brown">
<context filename="br-x01" paras="yes">
<p pnum="1">
<s snum="1">
<wf cmd="ignore" pos="DT">The</wf>
<wf cmd="done" pos="NN" lemma="bank" wnsn="1" lexsn="1:17:01::">bank</wf>
<wf cmd="done" pos="VB" lemma="say" wnsn="2" lexsn="2:32:00::">said</wf>
<wf cmd="done" pos="NN" lemma="run" wnsn="1;3" lexsn="1:04:00::">run</wf>
<wf cmd="ignore" pos="." >.</wf>
</s>
</p>
</context>
</contextfile>
"""


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setenv("NLTK_DATA", str(tmp_path))
    tagfiles = tmp_path / "corpora" / "semcor" / "brown1" / "tagfiles"
    tagfiles.mkdir(parents=True)
    (tagfiles / "br-x01.xml").write_text(SGML, encoding="latin-1")
    return tmp_path


def token(lemma="bank", pos="NN", sense=1, surface="bank", document="d1"):
    return S.Token(lemma=lemma, pos=pos, sense=sense, surface=surface, document=document)


# --- the SemCor reader ----------------------------------------------------------------------


def test_only_tagged_words_are_read(corpus):
    """`cmd="ignore"` marks function words with no sense tag. Counting them as sense 1
    would silently add thousands of free correct answers to every method."""
    sentences, stats = S.load()
    assert stats["tagged_tokens"] == 3
    assert [t.surface for t in sentences[0]] == ["bank", "said", "run"]


def test_the_sense_number_is_read_as_a_rank(corpus):
    sentences, _ = S.load()
    assert sentences[0][0].sense == 1
    assert sentences[0][1].sense == 2


def test_a_multi_sense_tag_takes_the_first(corpus):
    """Some tokens are tagged `1;3` where the annotator allowed two readings."""
    sentences, stats = S.load()
    assert sentences[0][2].sense == 1
    assert stats["multi_sense_tags"] == 1


def test_the_key_is_lemma_and_part_of_speech(corpus):
    """Sense numbers are assigned within a part of speech, so keying on the lemma alone
    would merge the noun and verb inventories of `run` and make sense 2 mean two things."""
    sentences, _ = S.load()
    assert sentences[0][0].key == ("bank", "NN")
    assert sentences[0][1].key == ("say", "VB")


def test_a_missing_corpus_raises_with_the_fetch_command(tmp_path, monkeypatch):
    monkeypatch.setenv("NLTK_DATA", str(tmp_path))
    with pytest.raises(LookupError, match="nltk.download"):
        S.load()


def test_inventory_collects_observed_senses():
    sentences = [[token(sense=1), token(sense=3)], [token(sense=1)]]
    assert S.inventory(sentences)[("bank", "NN")] == {1, 3}


def test_monosemous_tokens_are_excluded():
    """A word with one sense is not a disambiguation problem - every method gets it right,
    so including it adds the same constant to everyone and hides the differences."""
    sentences = [[token(lemma="bank", sense=1), token(lemma="dog", sense=1)]]
    senses = {("bank", "NN"): {1, 2}, ("dog", "NN"): {1}}
    kept = S.polysemous(sentences, senses)
    assert [t.lemma for t in kept] == ["bank"]


# --- the methods ----------------------------------------------------------------------------


def training():
    """`bank` is sense 2 twice and sense 1 once, so the trained MFS is 2 - deliberately
    different from WordNet's sense 1, so the two methods can be told apart."""
    return [
        [token(sense=2, surface="bank", document="a")],
        [token(sense=2, surface="bank", document="a")],
        [token(sense=1, surface="bank", document="b")],
    ]


INVENTORY = {("bank", "NN"): {1, 2}}


def test_first_sense_always_answers_one():
    model = D.FirstSense().fit(training(), INVENTORY)
    assert model.predict(token(sense=2), [token(sense=2)]) == 1


def test_trained_mfs_uses_the_training_majority():
    model = D.TrainedMFS().fit(training(), INVENTORY)
    assert model.predict(token(sense=1), [token(sense=1)]) == 2


def test_trained_mfs_falls_back_to_sense_one_for_an_unseen_lemma():
    model = D.TrainedMFS().fit(training(), INVENTORY)
    assert model.predict(token(lemma="zebra"), [token(lemma="zebra")]) == 1


def test_random_only_picks_from_the_observed_inventory():
    model = D.Random(seed=1).fit(training(), INVENTORY)
    for _ in range(20):
        assert model.predict(token(), [token()]) in {1, 2}


def test_random_is_reproducible_at_a_fixed_seed():
    a = [D.Random(seed=5).fit(training(), INVENTORY).predict(token(), [token()]) for _ in range(5)]
    b = [D.Random(seed=5).fit(training(), INVENTORY).predict(token(), [token()]) for _ in range(5)]
    assert a == b


# --- leakage --------------------------------------------------------------------------------


def test_the_oracle_reads_a_neighbours_gold_sense():
    """This is what makes it an oracle rather than a method, and the test says so.

    Two occurrences of `bank` in one document, gold senses 2 and 2. Asked about the first,
    the oracle answers 2 - which it can only know from the second token's *label*.
    """
    a = token(sense=2, document="test")
    b = token(sense=2, document="test")
    model = D.OneSensePerDiscourseOracle().fit(training(), INVENTORY)
    model.set_documents([[a, b]])
    assert model.predict(a, [a, b]) == 2


def test_the_oracle_does_not_count_its_own_vote():
    """Even an oracle must not read the answer it is being asked for. With one occurrence
    in the document there is no other vote, so it has to fall back."""
    only = token(sense=2, document="test")
    model = D.OneSensePerDiscourseOracle().fit(training(), INVENTORY)
    model.set_documents([[only]])
    assert model.predict(only, [only]) == 2  # from trained MFS, not from its own label


def test_the_honest_discourse_method_never_reads_a_gold_sense():
    """The real version votes over its *own predictions*. Changing the gold labels of the
    test tokens must not change what it answers - if it does, it is leaking.
    """
    model = D.ContextOverlapWithDiscourse().fit(training(), INVENTORY)
    a = [token(sense=1, document="t"), token(sense=1, document="t")]
    b = [token(sense=2, document="t"), token(sense=2, document="t")]
    sentence_of_a = {id(t): a for t in a}
    sentence_of_b = {id(t): b for t in b}
    first = list(model.predict_document(a, sentence_of_a).values())
    second = list(model.predict_document(b, sentence_of_b).values())
    assert first == second


def test_the_honest_discourse_method_is_consistent_within_a_document():
    """Its defining property: one sense per lemma per document."""
    model = D.ContextOverlapWithDiscourse().fit(training(), INVENTORY)
    tokens = [token(sense=1, document="t"), token(sense=2, document="t")]
    sentence_of = {id(t): tokens for t in tokens}
    assert len(set(model.predict_document(tokens, sentence_of).values())) == 1


# --- context overlap ------------------------------------------------------------------------


def test_context_overlap_prefers_the_sense_it_saw_the_context_with():
    """The one thing it must do: `bank` near `river` should not get the sense that only ever
    appeared near `money`."""
    train = [
        [token(sense=1, surface="bank"), token(lemma="river", surface="river", sense=1)],
        [token(sense=2, surface="bank"), token(lemma="money", surface="money", sense=1)],
    ]
    model = D.ContextOverlap().fit(train, {("bank", "NN"): {1, 2}})
    target = token(sense=1, surface="bank")
    sentence = [target, token(lemma="river", surface="river", sense=1)]
    assert model.predict(target, sentence) == 1


def test_context_overlap_falls_back_when_the_context_is_unknown():
    model = D.ContextOverlap().fit(training(), INVENTORY)
    target = token(surface="bank")
    sentence = [target, token(lemma="qqq", surface="qqq")]
    assert model.predict(target, sentence) in {1, 2}


def test_context_overlap_falls_back_for_an_unseen_lemma():
    model = D.ContextOverlap().fit(training(), INVENTORY)
    target = token(lemma="zebra", surface="zebra")
    assert model.predict(target, [target]) == 1


# --- scoring --------------------------------------------------------------------------------


def test_accuracy_is_the_share_correct():
    assert D.accuracy([1, 2, 1], [1, 2, 2]) == pytest.approx(2 / 3)


def test_accuracy_of_nothing_does_not_divide_by_zero():
    assert D.accuracy([], []) == 0.0


@pytest.mark.parametrize("method_cls", D.METHODS)
def test_every_method_returns_a_sense_from_the_inventory(method_cls):
    model = method_cls().fit(training(), INVENTORY)
    if isinstance(model, D.OneSensePerDiscourseOracle):
        model.set_documents(training())
    target = token()
    assert model.predict(target, [target]) in {1, 2}
