"""How much does exact-match-on-a-normal-form miss?

devign-leakage found duplicates by hashing a normalised form, and said plainly that
functions differing by a single statement would be invisible to it. This measures how
many that is, and - because Devign is labelled - how many of the pairs it finds disagree
about whether the same code is vulnerable.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minhash import LSH, MinHash, jaccard, shingles, tokenize_code

RESULTS = Path(__file__).resolve().parent.parent / "results"
CACHE_DIR = "datasets--google--code_x_glue_cc_defect_detection"

BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT = re.compile(r"//[^\n]*")
STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
CHAR = re.compile(r"'(?:[^'\\]|\\.)*'")
NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
SPACE = re.compile(r"\s+")
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")

KEYWORDS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "NULL",
    "size_t",
    "bool",
    "true",
    "false",
}

NEAR_DUPLICATE = 0.8  # Jaccard at or above this counts as a near-duplicate


def _cache_roots() -> list[Path]:
    roots = []
    if env := os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(env))
    if env := os.environ.get("HF_HOME"):
        roots.append(Path(env) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return roots


def load(split: str = "test"):
    import pandas as pd

    for root in _cache_roots():
        hits = sorted(
            glob.glob(
                str(root / CACHE_DIR / "snapshots" / "*" / "**" / f"{split}-*.parquet"),
                recursive=True,
            )
        )
        if hits:
            frame = pd.read_parquet(hits[0])
            return list(frame["func"]), [bool(t) for t in frame["target"]]
    raise FileNotFoundError(
        f"Devign '{split}' split not in the Hugging Face cache. Fetch it with:\n"
        '  python -c "from huggingface_hub import hf_hub_download as d; '
        f"d('google/code_x_glue_cc_defect_detection','data/{split}-00000-of-00001.parquet',"
        "repo_type='dataset')\""
    )


def normalise_exact(code: str) -> str:
    return SPACE.sub(" ", code).strip()


def normalise_structural(code: str) -> str:
    code = BLOCK_COMMENT.sub(" ", code)
    code = LINE_COMMENT.sub(" ", code)
    code = STRING.sub('"S"', code)
    code = CHAR.sub("'C'", code)
    code = NUMBER.sub("N", code)
    code = IDENT.sub(lambda m: m.group(0) if m.group(0) in KEYWORDS else "V", code)
    return SPACE.sub(" ", code).strip()


def hash_pairs(docs: list[str], normalise) -> set[tuple[int, int]]:
    """Pairs that collide under exact hashing of a normal form."""
    groups: dict[str, list[int]] = {}
    for i, d in enumerate(docs):
        groups.setdefault(normalise(d), []).append(i)
    pairs = set()
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.add((members[i], members[j]))
    return pairs


def main(split: str = "test", n_perm: int = 128) -> None:
    docs, labels = load(split)
    print(
        f"{split}: {len(docs)} functions, "
        f"{sum(labels)} vulnerable / {len(labels) - sum(labels)} not\n"
    )

    # --- exact and structural hashing -------------------------------------------------
    started = time.time()
    exact = hash_pairs(docs, normalise_exact)
    structural = hash_pairs(docs, normalise_structural)
    hash_seconds = time.time() - started
    print(f"  exact hash          {len(exact):5} pairs")
    print(f"  structural hash     {len(structural):5} pairs   ({hash_seconds:.1f}s)")

    # --- MinHash + LSH ------------------------------------------------------------------
    started = time.time()
    shingle_sets = [shingles(tokenize_code(d)) for d in docs]
    minhash = MinHash(n_perm=n_perm)
    signatures = [minhash.signature(s) for s in shingle_sets]
    lsh = LSH(bands=32, rows=n_perm // 32)
    for i, sig in enumerate(signatures):
        lsh.add(i, sig)
    candidates = lsh.candidate_pairs()
    lsh_seconds = time.time() - started
    print(
        f"  LSH candidates      {len(candidates):5} pairs   ({lsh_seconds:.1f}s, "
        f"threshold ~{lsh.threshold:.2f})"
    )

    # --- verify candidates with exact Jaccard --------------------------------------------
    started = time.time()
    verified = []
    for i, j in candidates:
        sim = jaccard(shingle_sets[i], shingle_sets[j])
        if sim >= NEAR_DUPLICATE:
            verified.append((i, j, sim))
    verify_seconds = time.time() - started
    print(f"  verified >= {NEAR_DUPLICATE}     {len(verified):5} pairs   ({verify_seconds:.1f}s)")

    # --- what did hashing miss? -----------------------------------------------------------
    found_by_hash = exact | structural
    missed = [(i, j, s) for i, j, s in verified if (i, j) not in found_by_hash]
    conflicting = [(i, j, s) for i, j, s in verified if labels[i] != labels[j]]
    missed_conflicting = [(i, j, s) for i, j, s in missed if labels[i] != labels[j]]

    print(f"\n  near-duplicates hashing MISSED : {len(missed)}")
    print(f"  of those, conflicting labels   : {len(missed_conflicting)}")
    print(f"  all near-dupes w/ conflicting  : {len(conflicting)}")

    # --- is the MinHash estimate any good? -------------------------------------------------
    # Measured on two samples, because one is misleading alone. Random pairs are almost all
    # disjoint, and MinHash returns exactly 0 for disjoint sets - so an MAE over random
    # pairs is mostly counting easy zeros and looks far better than theory allows. The
    # number that matters is the error on pairs that are actually similar.
    rng = np.random.default_rng(0)
    random_pairs = [(int(i), int(j)) for i, j in rng.choice(len(docs), size=(400, 2)) if i != j]
    similar_pairs = [(i, j) for i, j, _ in verified]

    def mae_over(pairs):
        if not pairs:
            return None
        return float(
            np.mean(
                [
                    abs(
                        jaccard(shingle_sets[i], shingle_sets[j])
                        - MinHash.similarity(signatures[i], signatures[j])
                    )
                    for i, j in pairs
                ]
            )
        )

    mae_random = mae_over(random_pairs)
    mae_similar = mae_over(similar_pairs)
    theory = 1 / np.sqrt(n_perm)
    print(f"\n  MinHash MAE vs exact Jaccard ({n_perm} permutations)")
    print(
        f"    {len(random_pairs):4} random pairs   : {mae_random:.4f}   "
        "<- mostly disjoint, so mostly exact zeros"
    )
    if mae_similar is not None:
        print(
            f"    {len(similar_pairs):4} near-duplicates: {mae_similar:.4f}   "
            "<- the number that matters"
        )
    print(f"    theoretical 1/sqrt(n)  : {theory:.4f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"duplicates_{split}.json").write_text(
        json.dumps(
            {
                "split": split,
                "documents": len(docs),
                "n_perm": n_perm,
                "lsh_threshold": lsh.threshold,
                "near_duplicate_cutoff": NEAR_DUPLICATE,
                "exact_pairs": len(exact),
                "structural_pairs": len(structural),
                "lsh_candidates": len(candidates),
                "verified_near_duplicates": len(verified),
                "missed_by_hashing": len(missed),
                "missed_and_conflicting": len(missed_conflicting),
                "all_conflicting": len(conflicting),
                "minhash_mae_random_pairs": mae_random,
                "minhash_mae_near_duplicates": mae_similar,
                "minhash_theoretical_se": float(theory),
                "seconds": {
                    "hash": round(hash_seconds, 1),
                    "lsh": round(lsh_seconds, 1),
                    "verify": round(verify_seconds, 1),
                },
                "examples": [
                    {
                        "a": int(i),
                        "b": int(j),
                        "jaccard": round(float(s), 3),
                        "label_a": labels[i],
                        "label_b": labels[j],
                    }
                    for i, j, s in sorted(missed, key=lambda x: -x[2])[:25]
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
