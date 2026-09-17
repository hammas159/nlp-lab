"""Tests for the lexicon and corpus parsers.

No dataset and no network: each parser is pointed at a small file written here, in the exact
format the real resource ships in. That tests the parsing - which is where the bugs are -
without needing 40MB of corpora present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import corpora as C
import lexicons as L


@pytest.fixture
def data(tmp_path, monkeypatch):
    """A fake NLTK_DATA tree, so the loaders are exercised end to end."""
    monkeypatch.setenv("NLTK_DATA", str(tmp_path))
    return tmp_path


def write(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding=encoding)


# --- VADER ----------------------------------------------------------------------------------


def test_vader_reads_word_and_valence(data):
    write(
        data / "sentiment" / "vader_lexicon" / "vader_lexicon.txt",
        "good\t1.9\t0.7\t[1, 2]\nbad\t-2.5\t0.8\t[-2, -3]\n",
    )
    lex = L.load_vader()
    assert lex.get("good") > 0 > lex.get("bad")


def test_vader_skips_malformed_lines(data):
    """The shipped file contains emoticon entries and blank lines; one bad row must not
    take the whole lexicon down."""
    write(
        data / "sentiment" / "vader_lexicon" / "vader_lexicon.txt",
        "good\t1.9\t0.7\t[1]\nbroken\n\nbad\tNOT_A_NUMBER\t0\t[]\nawful\t-3.0\t0.5\t[-3]\n",
    )
    lex = L.load_vader()
    assert len(lex) == 2
    assert "awful" in lex


# --- Opinion lexicon ------------------------------------------------------------------------


def test_opinion_reads_both_lists_and_skips_comments(data):
    root = data / "corpora" / "opinion_lexicon"
    write(root / "positive-words.txt", "; a comment\n;;;;\ngood\ngreat\n", encoding="latin-1")
    write(root / "negative-words.txt", "; header\nbad\n", encoding="latin-1")
    lex = L.load_opinion()
    assert lex.get("good") > 0 and lex.get("bad") < 0
    assert len(lex) == 3


def test_opinion_survives_latin_1_bytes(data):
    """The distributed files are not UTF-8; reading them as UTF-8 raises."""
    root = data / "corpora" / "opinion_lexicon"
    write(root / "positive-words.txt", "na\xefve\ngood\n", encoding="latin-1")
    write(root / "negative-words.txt", "bad\n", encoding="latin-1")
    assert len(L.load_opinion()) == 3


def test_opinion_has_no_magnitude(data):
    """Its defining property: every word counts the same, which is exactly what the other
    lexicons do not do."""
    root = data / "corpora" / "opinion_lexicon"
    write(root / "positive-words.txt", "good\ngreat\nsuperb\n", encoding="latin-1")
    write(root / "negative-words.txt", "bad\n", encoding="latin-1")
    lex = L.load_opinion()
    assert lex.get("good") == lex.get("great") == lex.get("superb")


# --- SentiWordNet ---------------------------------------------------------------------------


SWN_HEADER = "# POS\tID\tPosScore\tNegScore\tSynsetTerms\tGloss\n"


def test_sentiwordnet_scores_are_positive_minus_negative(data):
    write(
        data / "corpora" / "sentiwordnet" / "SentiWordNet_3.0.0.txt",
        SWN_HEADER + "a\t00001\t0.75\t0.25\tgood#1\ta gloss\n",
    )
    assert L.load_sentiwordnet().get("good") > 0


def test_sentiwordnet_takes_the_first_senses_by_rank(data):
    """`word#rank` gives the sense order. Taking senses in file order instead would mix a
    word's rare senses into its score and quietly change every number."""
    write(
        data / "corpora" / "sentiwordnet" / "SentiWordNet_3.0.0.txt",
        SWN_HEADER + "a\t1\t0.0\t0.9\tfine#5\tg\n" + "a\t2\t0.9\t0.0\tfine#1\tg\n",
    )
    assert L.load_sentiwordnet(max_senses=1).get("fine") > 0


def test_sentiwordnet_skips_neutral_synsets(data):
    write(
        data / "corpora" / "sentiwordnet" / "SentiWordNet_3.0.0.txt",
        SWN_HEADER + "a\t1\t0.0\t0.0\tflat#1\tg\n",
    )
    assert "flat" not in L.load_sentiwordnet()


def test_sentiwordnet_normalises_multiword_underscores(data):
    write(
        data / "corpora" / "sentiwordnet" / "SentiWordNet_3.0.0.txt",
        SWN_HEADER + "a\t1\t0.8\t0.0\twell_off#1\tg\n",
    )
    assert "well off" in L.load_sentiwordnet()


# --- AFINN ----------------------------------------------------------------------------------


def test_afinn_reads_tab_separated_scores(data):
    write(data / "corpora" / "afinn" / "AFINN-111.txt", "abandon\t-2\nadore\t3\n")
    lex = L.load_afinn()
    assert lex.get("adore") > 0 > lex.get("abandon")


def test_afinn_keeps_multiword_entries_whole(data):
    """AFINN contains phrases like `cool stuff`; splitting on whitespace would lose them."""
    write(data / "corpora" / "afinn" / "AFINN-111.txt", "cool stuff\t3\n")
    assert "cool stuff" in L.load_afinn()


# --- missing resources ----------------------------------------------------------------------


def test_a_missing_lexicon_raises_with_the_fetch_command(data):
    with pytest.raises(LookupError, match="nltk.download"):
        L.load_vader()


def test_load_all_returns_what_is_present_rather_than_failing(data, capsys):
    """A partial download must give a smaller table, not a crash - which is how this project
    was built while the corpora were still arriving."""
    write(data / "corpora" / "afinn" / "AFINN-111.txt", "adore\t3\n")
    loaded = L.load_all()
    assert [x.name for x in loaded] == ["AFINN"]
    assert "SKIPPED VADER" in capsys.readouterr().out


# --- corpora --------------------------------------------------------------------------------


def test_sentence_polarity_labels_both_files(data):
    root = data / "corpora" / "sentence_polarity" / "rt-polaritydata"
    write(root / "rt-polarity.pos", "a fine film\nanother\n", encoding="latin-1")
    write(root / "rt-polarity.neg", "dreadful\n", encoding="latin-1")
    texts, labels = C.load_sentence_polarity()
    assert len(texts) == 3
    assert labels == [1, 1, 0]


def test_twitter_samples_reads_json_lines(data):
    root = data / "corpora" / "twitter_samples"
    write(root / "positive_tweets.json", json.dumps({"text": "lovely"}) + "\n")
    write(root / "negative_tweets.json", json.dumps({"text": "awful"}) + "\n")
    texts, labels = C.load_twitter_samples()
    assert texts == ["lovely", "awful"]
    assert labels == [1, 0]


def test_a_corrupt_json_line_is_skipped(data):
    root = data / "corpora" / "twitter_samples"
    write(root / "positive_tweets.json", "{not json}\n" + json.dumps({"text": "ok"}) + "\n")
    write(root / "negative_tweets.json", json.dumps({"text": "bad"}) + "\n")
    assert C.load_twitter_samples()[0] == ["ok", "bad"]


def test_available_lists_only_what_is_on_disk(data):
    root = data / "corpora" / "sentence_polarity" / "rt-polaritydata"
    write(root / "rt-polarity.pos", "good\n", encoding="latin-1")
    write(root / "rt-polarity.neg", "bad\n", encoding="latin-1")
    assert C.available() == ["movie sentences"]


def test_sentence_polarity_accepts_the_flat_layout(data):
    """The distribution has shipped these files both nested in `rt-polaritydata/` and flat
    in `sentence_polarity/`. Assuming one layout meant the corpus loaded as 'not found' on
    a machine that plainly had it."""
    root = data / "corpora" / "sentence_polarity"
    write(root / "rt-polarity.pos", "a fine film\n", encoding="latin-1")
    write(root / "rt-polarity.neg", "dreadful\n", encoding="latin-1")
    texts, labels = C.load_sentence_polarity()
    assert texts == ["a fine film", "dreadful"]
    assert labels == [1, 0]
