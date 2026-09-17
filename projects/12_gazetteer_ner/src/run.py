"""What bounds a gazetteer tagger: coverage, or the ambiguity inside the gazetteer?

python src/run.py            # all 66,581 paragraphs
python src/run.py --quick    # a tenth
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from shared.benchmark import build as build_corpus
from shared.benchmark import tokenize

import gazetteer as gaz
from ahocorasick import AhoCorasick, longest_non_overlapping

RESULTS = Path(__file__).resolve().parent.parent / "results"


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--sample", type=int, default=3_000, help="paragraphs to tag")
    args = parser.parse_args()

    started = time.time()
    rule("1. The gazetteer")
    doc_ids, docs, _ = build_corpus(700 if args.quick else 7_405)
    gazetteer = gaz.build(doc_ids, docs)
    for key, value in gazetteer.stats.items():
        print(f"  {key:28} {value:,}")

    collision_rate = (
        gazetteer.stats["titles_lost_to_collision"] / gazetteer.stats["titles"]
        if gazetteer.stats["titles"]
        else 0.0
    )
    print(
        f"\n  {gazetteer.stats['colliding_surfaces']:,} surface forms are shared by more than "
        f"one entity,\n  covering {gazetteer.stats['titles_lost_to_collision']:,} titles "
        f"({collision_rate * 100:.1f}%). Those entries are unresolvable\n  by any "
        f"context-free matcher - the name does not identify the entity."
    )

    rule("2. The entries that are also ordinary English")
    frequency = gaz.document_frequency(docs, gazetteer.vocabulary)
    n_docs = len(docs)

    single = [
        (i, gazetteer.surfaces[i], frequency[gazetteer.patterns[i][0]])
        for i in range(len(gazetteer))
        if len(gazetteer.patterns[i]) == 1
    ]
    single.sort(key=lambda row: -row[2])
    print(f"  {len(single):,} gazetteer entries are a single token.")
    print(f"\n  {'entry':24} {'appears in':>12}  share of corpus")
    for _, surface, df in single[:12]:
        print(f"  {surface[:24]:24} {df:>12,}  {df / n_docs * 100:>8.1f}%")

    noisy = [row for row in single if row[2] / n_docs > 0.01]
    print(
        f"\n  {len(noisy):,} single-token entries appear in over 1% of paragraphs. Each one "
        f"fires\n  thousands of times and is almost never the entity."
    )

    rule("3. Tagging, and what a match is worth")
    sample_size = min(args.sample, len(docs))
    automaton = AhoCorasick(gazetteer.patterns)
    print(f"  automaton: {len(gazetteer):,} patterns, {automaton.size:,} trie nodes")

    surface_of_title = {}
    for i, titles in enumerate(gazetteer.titles_per_surface):
        for title in titles:
            surface_of_title[title] = i

    step = time.time()
    self_found = 0
    other_counts = []
    dropped_by_resolution = []

    for index in range(sample_size):
        tokens = [
            gazetteer.vocabulary[t] for t in tokenize(docs[index]) if t in gazetteer.vocabulary
        ]
        matches = automaton.find(tokens)
        resolved = longest_non_overlapping(matches)
        dropped_by_resolution.append(len(matches) - len(resolved))

        own = surface_of_title.get(doc_ids[index])
        patterns_here = {m.pattern for m in resolved}
        if own is not None and own in patterns_here:
            self_found += 1
        other_counts.append(len(patterns_here - ({own} if own is not None else set())))

    elapsed = time.time() - step
    others = np.array(other_counts)
    print(f"  tagged {sample_size:,} paragraphs in {elapsed:.1f}s")
    print(
        f"\n  the paragraph's own title was found in {self_found / sample_size * 100:.1f}% of them"
    )
    print(
        f"  other gazetteer entries matched as well: median {int(np.median(others))}, "
        f"mean {others.mean():.1f}, max {others.max():,}"
    )
    print(
        f"  overlapping matches discarded by longest-match resolution: "
        f"{sum(dropped_by_resolution):,}"
    )
    print(
        f"\n  A tagger with no disambiguation returns all of those. If one is the entity the\n"
        f"  paragraph is about, the other {others.mean():.1f} on average are candidates it has\n"
        f"  no way to rank - which is the precision ceiling, and it is set by the gazetteer,\n"
        f"  not by the matcher."
    )

    rule("4. Against a naive scan")
    naive_patterns = min(200, len(gazetteer))
    step = time.time()
    subset = [gazetteer.patterns[i] for i in range(naive_patterns)]
    sample_tokens = [
        [gazetteer.vocabulary[t] for t in tokenize(docs[i]) if t in gazetteer.vocabulary]
        for i in range(min(200, sample_size))
    ]
    hits = 0
    for tokens in sample_tokens:
        for pattern in subset:
            n = len(pattern)
            for start in range(len(tokens) - n + 1):
                if tokens[start : start + n] == pattern:
                    hits += 1
    naive_time = time.time() - step

    step = time.time()
    small = AhoCorasick(subset)
    for tokens in sample_tokens:
        small.find(tokens)
    automaton_time = time.time() - step

    print(
        f"  {naive_patterns} patterns over {len(sample_tokens)} paragraphs:\n"
        f"    one pattern at a time   {naive_time:.2f}s\n"
        f"    Aho-Corasick            {automaton_time:.2f}s   "
        f"({naive_time / automaton_time:.1f}x faster)"
    )
    print(
        f"  The full gazetteer is {len(gazetteer) / naive_patterns:.0f}x larger, and the naive\n"
        f"  scan grows with it while the automaton does not."
    )

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "gazetteer.json").write_text(
        json.dumps(
            {
                "gazetteer": gazetteer.stats,
                "collision_rate": collision_rate,
                "single_token_entries": len(single),
                "single_token_entries_over_one_percent": len(noisy),
                "noisiest": [
                    {"surface": s, "documents": int(df), "share": df / n_docs}
                    for _, s, df in single[:25]
                ],
                "tagging": {
                    "paragraphs": sample_size,
                    "seconds": elapsed,
                    "self_title_found_rate": self_found / sample_size,
                    "other_entries_median": float(np.median(others)),
                    "other_entries_mean": float(others.mean()),
                    "other_entries_max": int(others.max()),
                    "overlaps_discarded": int(sum(dropped_by_resolution)),
                },
                "speed": {
                    "patterns": naive_patterns,
                    "paragraphs": len(sample_tokens),
                    "naive_seconds": naive_time,
                    "automaton_seconds": automaton_time,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'gazetteer.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
