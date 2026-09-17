"""Tests for the four views.

No dataset and no network. Each view is defined by what it *removes*, so the tests check
removal directly: the thing that should be gone is gone, and the thing that should survive
survives. A view that leaked identifier content would make the whole project's conclusion
wrong while producing a perfectly plausible accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from views import C_KEYWORDS, VIEWS, layout, lexical, masked, structural, view, vocabulary_of

SAMPLE = 'int av_frame_alloc(int count) {\n    char *buf = "hello";\n    return count + 42;\n}'


# --- lexical ------------------------------------------------------------------------------


def test_lexical_keeps_identifiers():
    assert "av_frame_alloc" in lexical(SAMPLE)


def test_lexical_keeps_keywords_and_punctuation():
    tokens = lexical(SAMPLE)
    assert "int" in tokens
    assert "{" in tokens


def test_lexical_keeps_numbers():
    assert "42" in lexical(SAMPLE)


# --- masked -------------------------------------------------------------------------------


def test_masked_removes_identifier_content():
    """The central guarantee of the project. If a project-specific name survived here, the
    'topic removed' view would still carry topic and the conclusion would be wrong."""
    tokens = masked(SAMPLE)
    assert "av_frame_alloc" not in tokens
    assert "buf" not in tokens
    assert "ID" in tokens


def test_masked_keeps_keywords():
    """Keywords are grammar, not subject matter, so they must survive."""
    tokens = masked(SAMPLE)
    assert "int" in tokens
    assert "return" in tokens


def test_masked_replaces_numbers_and_strings():
    tokens = masked(SAMPLE)
    assert "NUM" in tokens
    assert "STR" in tokens
    assert "42" not in tokens
    assert '"hello"' not in tokens


def test_masked_keeps_punctuation():
    assert "{" in masked(SAMPLE)


def test_masked_preserves_identifier_density():
    """Naming *content* goes, naming *frequency* stays - a function with five identifiers
    still shows five IDs, so density is still measurable."""
    assert masked("int a = b + c;").count("ID") == 3


# --- structural ---------------------------------------------------------------------------


def test_structural_drops_identifiers_entirely():
    tokens = structural(SAMPLE)
    assert "ID" not in tokens
    assert "av_frame_alloc" not in tokens


def test_structural_keeps_keywords_and_operators():
    tokens = structural(SAMPLE)
    assert "int" in tokens
    assert "+" in tokens
    assert ";" in tokens


def test_structural_is_shorter_than_masked():
    """Masked replaces identifiers with a token; structural removes them. If structural were
    not shorter, it would not be removing anything."""
    assert len(structural(SAMPLE)) < len(masked(SAMPLE))


def test_every_structural_token_is_a_keyword_or_punctuation():
    for token in structural(SAMPLE):
        assert token in C_KEYWORDS or not token.isalnum()


# --- layout -------------------------------------------------------------------------------


def test_layout_has_no_token_content():
    """Not a single character of the source may survive. This is the strictest 'style only'
    view, and its value depends entirely on that."""
    tokens = layout(SAMPLE)
    joined = " ".join(tokens)
    for word in ("av_frame_alloc", "int", "return", "hello", "buf"):
        assert word not in joined


def test_layout_notices_indentation():
    flat = layout("int a;\nint b;")
    indented = layout("int a;\n        int b;")
    assert flat != indented


def test_layout_notices_brace_placement():
    """`brace_own_line` is the discriminating marker, not `brace_trailing`.

    A line consisting of a lone `{` both starts *and* ends with a brace, so it earns both
    tags. That is not wrong - it is literally true of the line - but it means
    `brace_trailing` alone cannot separate the two conventions, and only `brace_own_line`
    can.
    """
    trailing = layout("if (x) {\n}")
    own_line = layout("if (x)\n{\n}")
    assert "brace_own_line" not in trailing
    assert "brace_own_line" in own_line


def test_layout_notices_comments():
    assert "comment" in layout("// a comment\nint a;")


def test_layout_notices_tabs():
    assert "tab" in layout("int a;\n\tint b;")


def test_layout_notices_blank_lines():
    assert "blank" in layout("int a;\n\nint b;")


def test_layout_bins_line_length():
    short = layout("int a;")
    long = layout("int " + "a" * 200 + ";")
    assert short != long


# --- the dispatcher and vocabulary --------------------------------------------------------


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_every_view_returns_tokens(name):
    assert len(view(name, SAMPLE)) > 0


@pytest.mark.parametrize("name", sorted(VIEWS))
def test_every_view_handles_empty_input(name):
    assert isinstance(view(name, ""), list)


def test_an_unknown_view_is_rejected():
    with pytest.raises(ValueError, match="unknown view"):
        view("semantic", SAMPLE)


def test_vocabulary_respects_the_minimum_count():
    texts = [SAMPLE] * 5 + ["int rare_name_here;"]
    vocabulary = vocabulary_of(texts, "lexical", min_count=3)
    assert "rare_name_here" not in vocabulary
    assert "int" in vocabulary


def test_vocabulary_shrinks_as_topic_is_stripped():
    """The feature counts are themselves the evidence: thousands of identifiers collapse to
    a few dozen structural tokens."""
    texts = [SAMPLE] * 5
    assert len(vocabulary_of(texts, "lexical")) > len(vocabulary_of(texts, "structural"))
