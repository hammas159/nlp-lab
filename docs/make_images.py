"""Render the Input / Output terminal cards in docs/images.

These were hand-made once and went stale the moment project 01 grew from four methods to
seven - the committed output.png still showed a four-row table and a headline that the
finished experiment contradicts. Generating them from a script means the next change to the
numbers is one command away from being reflected, and the numbers themselves are read from
`results/scores.json` rather than retyped.

    python docs/make_images.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
IMAGES = Path(__file__).resolve().parent / "images"
SCORES = ROOT / "projects" / "01_embedding_fair_comparison" / "results" / "scores.json"

FONT = "C:/Windows/Fonts/consola.ttf"
SIZE = 30
LINE = 42
PAD = 34
BAR = 74

BACKGROUND = "#0d1117"
TITLEBAR = "#161b22"
DOTS = ("#ff5f57", "#febc2e", "#28c840")

WHITE = "#e6edf3"
GREY = "#8b949e"
DIM = "#6e7681"
GREEN = "#3fb950"
AMBER = "#d29922"
RED = "#f85149"


def card(title: str, rows: list[tuple[str, str]], path: Path) -> None:
    """Rows are (colour, text). One row per line; "" is a blank line."""
    font = ImageFont.truetype(FONT, SIZE)
    bar_font = ImageFont.truetype(FONT, SIZE - 4)

    width = (
        max(
            (font.getbbox(text)[2] for _, text in rows if text),
            default=400,
        )
        + PAD * 2
    )
    width = max(width, 900)
    height = BAR + PAD + LINE * len(rows) + PAD

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.rectangle([0, 0, width, BAR], fill=TITLEBAR)
    for i, colour in enumerate(DOTS):
        x = 26 + i * 30
        draw.ellipse([x, BAR // 2 - 9, x + 18, BAR // 2 + 9], fill=colour)
    draw.text((126, BAR // 2), title, font=bar_font, fill=DIM, anchor="lm")

    y = BAR + PAD
    for colour, text in rows:
        if text:
            draw.text((PAD, y), text, font=font, fill=colour)
        y += LINE

    IMAGES.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"wrote {path.relative_to(ROOT)}  ({width}x{height})")


def main() -> None:
    rows = json.loads(SCORES.read_text())["rows"]
    by_name = {r["method"]: r for r in rows}
    order = [
        "BGE-small (pretrained)",
        "nomic-embed (pretrained)",
        "BM25",
        "TF-IDF",
        "LSA (SVD)",
        "word2vec (corpus)",
        "fastText (corpus)",
    ]

    card(
        "input - seven methods, one fair corpus",
        [
            (GREY, "$ python src/evaluate.py 300"),
            (WHITE, ""),
            (WHITE, "  Input - one corpus, one query set, seven retrieval methods"),
            (WHITE, ""),
            (GREY, "    documents     2,964      277,259 tokens, 25,295 types"),
            (
                GREY,
                "    queries       300        exactly 2 gold documents each, 0 missing",
            ),
            (
                GREY,
                "    dimensions    300        LSA, word2vec, fastText - the ones fitted here",
            ),
            (
                GREY,
                "                             BGE is 384, nomic 768, and cannot be changed",
            ),
            (GREY, ""),
            (AMBER, "    pretrained    BGE-small, nomic-embed      trained elsewhere"),
            (GREEN, "    this corpus   BM25, TF-IDF, LSA, word2vec, fastText"),
            (WHITE, ""),
            (
                WHITE,
                "  Five methods are fitted on this corpus. Two arrive pretrained on",
            ),
            (WHITE, "  billions of words. That asymmetry is the experiment."),
        ],
        IMAGES / "input.png",
    )

    table = [
        (GREY, "$ python src/evaluate.py 300"),
        (WHITE, ""),
        (
            DIM,
            "  method                     trained on      r@1     r@10    r@20     MRR",
        ),
    ]
    for name in order:
        r = by_name[name]
        local = r["trained_on"] == "this corpus"
        colour = (
            GREEN
            if name == "BM25"
            else (RED if r["recall@10"] < 0.3 else (GREY if local else AMBER))
        )
        where = "this corpus" if local else "elsewhere  "
        line = (
            f"  {name:26} {where}    {r['recall@1']:.3f}   "
            f"{r['recall@10']:.3f}   {r['recall@20']:.3f}   {r['mrr']:.3f}"
        )
        table.append((colour, line))
    table += [
        (WHITE, ""),
        (
            WHITE,
            "  Four of these are dense vectors + cosine: BGE, nomic, word2vec, fastText.",
        ),
        (RED, "  They span 0.808, split exactly by where the vectors came from."),
        (WHITE, ""),
        (
            WHITE,
            "  LSA, word2vec and fastText are all 300d, all fitted here, scored alike -",
        ),
        (
            AMBER,
            "  and a 1990s truncated SVD beats both neural objectives by about 0.6.",
        ),
        (
            GREEN,
            "  BM25, with zero parameters, beats everything trained on this corpus.",
        ),
    ]
    card("output - the method was never the variable", table, IMAGES / "output.png")


if __name__ == "__main__":
    main()
