"""Hold the encoder fixed, vary the pooling - then swap the encoder and do it again.

python src/run.py            # 2,964 documents, 300 queries, every encoder present
python src/run.py --quick    # 600 documents
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

from shared.benchmark import build
from shared.bm25 import BM25

import pooling as P

RESULTS = Path(__file__).resolve().parent.parent / "results"
MAX_LENGTH = 256
BATCH = 64

#: Two encoders with opposite trained poolings, both 384-dimensional - so a difference
#: between them cannot be a difference in vector width.
ENCODERS = [
    (
        "BGE-small",
        "BAAI/bge-small-en-v1.5",
        "cls",
        "Represent this sentence for searching relevant passages: ",
    ),
    ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2", "mean", ""),
]


def available(model_name: str) -> bool:
    """Is this encoder's weight file actually on disk, and non-empty?

    Checked rather than assumed. An encoder whose weights are missing is skipped by name,
    the way project 19 skips a missing lexicon - one that silently failed to load and fell
    back to something else would report numbers under the wrong label. The size check
    matters because an interrupted download leaves a zero-byte placeholder.
    """
    from huggingface_hub import try_to_load_from_cache

    for filename in ("model.safetensors", "pytorch_model.bin"):
        hit = try_to_load_from_cache(model_name, filename)
        if isinstance(hit, str) and Path(hit).exists() and Path(hit).stat().st_size > 0:
            return True
    return False


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * max(len(title), 78)}", flush=True)


def encode_all(model_name: str, texts: list[str], idf: np.ndarray | None, tokenizer):
    """Return {pooling name: matrix}. Every pooling comes from the same forward pass.

    Pooled per batch rather than by storing hidden states: 2,964 documents at 256 tokens and
    384 dimensions is about 4.6 GB in float64, and none of it is needed once pooled.
    """
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    out: dict[str, list[np.ndarray]] = {name: [] for name in P.POOLINGS}
    with torch.no_grad():
        for start in range(0, len(texts), BATCH):
            batch = texts[start : start + BATCH]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            ).to(device)
            hidden = model(**encoded).last_hidden_state.cpu().numpy().astype(np.float64)
            mask = encoded["attention_mask"].cpu().numpy()
            ids = encoded["input_ids"].cpu().numpy()
            weights = idf[ids] if idf is not None else None
            for name in P.POOLINGS:
                out[name].append(P.pool(name, hidden, mask, weights))
    return {name: np.vstack(chunks) for name, chunks in out.items()}


def recall_at_10(doc_vectors, query_vectors, gold) -> float:
    documents = P.unit(doc_vectors)
    queries = P.unit(query_vectors)
    total = 0.0
    counted = 0
    for i, golds in enumerate(gold):
        if not golds:
            continue
        scores = documents @ queries[i]
        top = np.argpartition(-scores, 10)[:10]
        total += len(golds & set(top.tolist())) / len(golds)
        counted += 1
    return total / max(1, counted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    started = time.time()

    rule("1. The setup")
    doc_ids, docs, queries = build(300)
    if args.quick:
        doc_ids, docs = doc_ids[:600], docs[:600]
        titles = set(doc_ids)
        queries = [q for q in queries if any(t in titles for t in q.gold_titles)][:100]
    title_to_idx = {t: i for i, t in enumerate(doc_ids)}
    gold = [{title_to_idx[t] for t in q.gold_titles if t in title_to_idx} for q in queries]
    questions = [q.question for q in queries]
    answerable = sum(1 for g in gold if g)
    print(f"  {len(docs):,} documents, {answerable} answerable queries")

    bm25 = BM25().fit(docs)
    total = 0.0
    for i, golds in enumerate(gold):
        if not golds:
            continue
        scores = bm25.score(questions[i])
        top = np.argpartition(-scores, 10)[:10]
        total += len(golds & set(top.tolist())) / len(golds)
    bm25_recall = total / answerable
    print(f"  BM25 on the same corpus: recall@10 {bm25_recall:.3f}")
    print("  Every number below is one transformer being read five different ways.")

    present = [e for e in ENCODERS if available(e[1])]
    for label, model_name, *_ in ENCODERS:
        if not any(e[1] == model_name for e in present):
            print(f"  SKIPPED {label}: weights are not in the local cache")

    from transformers import AutoTokenizer

    rows = []
    similarities = []
    for label, model_name, trained_pooling, prefix in present:
        rule(f"2. {label} - trained with {trained_pooling.upper()} pooling")
        began = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenised = [
            tokenizer(d, truncation=True, max_length=MAX_LENGTH)["input_ids"] for d in docs
        ]
        idf = P.token_idf(tokenised, tokenizer.vocab_size)

        doc_vectors = encode_all(model_name, docs, idf, tokenizer)
        query_vectors = encode_all(model_name, [prefix + q for q in questions], idf, tokenizer)

        scores = {
            name: recall_at_10(doc_vectors[name], query_vectors[name], gold) for name in P.POOLINGS
        }
        print(f"  {'pooling':12} {'recall@10':>11} {'vs trained':>12}")
        for name in P.POOLINGS:
            marker = "  <- as trained" if name == trained_pooling else ""
            rows.append(
                {
                    "encoder": label,
                    "trained_pooling": trained_pooling,
                    "pooling": name,
                    "recall@10": scores[name],
                }
            )
            print(
                f"  {name:12} {scores[name]:>11.3f} "
                f"{scores[name] - scores[trained_pooling]:>+12.3f}{marker}"
            )
        best = max(scores, key=lambda n: scores[n])
        verdict = "MATCH" if best == trained_pooling else "MISMATCH"
        print(f"  best: {best}   trained: {trained_pooling}   {verdict}")

        # Three poolings scoring identically is either a coincidence or a fact about the
        # vectors. Measuring how similar the pooled vectors actually are tells them apart,
        # and turns an equality in a results table into a mechanism.
        print("\n  Mean cosine between the document vectors each pooling produces:")
        print("  " + " " * 10 + "".join(f"{n:>10}" for n in P.POOLINGS))
        for a in P.POOLINGS:
            cells = []
            for b in P.POOLINGS:
                similarity = float(
                    (P.unit(doc_vectors[a]) * P.unit(doc_vectors[b])).sum(axis=1).mean()
                )
                cells.append(f"{similarity:>10.3f}")
                if a < b:
                    similarities.append({"encoder": label, "a": a, "b": b, "cosine": similarity})
            print(f"  {a:10}" + "".join(cells))
        print(f"  ({time.time() - began:.1f}s)")

    rule("3. Does 'which pooling is best' have an answer?")
    by_encoder: dict[str, dict[str, float]] = {}
    for r in rows:
        by_encoder.setdefault(r["encoder"], {})[r["pooling"]] = r["recall@10"]
    names = list(P.POOLINGS)
    labels = [label for label, *_ in present]

    print(f"  {'pooling':12}" + "".join(f"{label:>14}" for label in labels))
    for name in names:
        print(f"  {name:12}" + "".join(f"{by_encoder[label][name]:>14.3f}" for label in labels))

    rankings = {label: sorted(names, key=lambda n: -by_encoder[label][n]) for label in labels}
    print()
    for label, order in rankings.items():
        print(f"  {label:12} ranks poolings: {' > '.join(order)}")

    if len(rankings) > 1:
        orders = list(rankings.values())
        agree = all(o == orders[0] for o in orders)
        word = "identically" if agree else "DIFFERENTLY"
        print(f"\n  The encoders rank the five poolings {word}.")
    else:
        print("\n  Only one encoder is present, so the cross-encoder comparison is pending.")

    print()
    for label, _, trained, _ in present:
        best = max(names, key=lambda n: by_encoder[label][n])
        worst = min(names, key=lambda n: by_encoder[label][n])
        spread = by_encoder[label][best] - by_encoder[label][worst]
        print(f"  {label:12} best={best:9} trained={trained:5} worst={worst:9} spread {spread:.3f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "pooling.json").write_text(
        json.dumps(
            {
                "documents": len(docs),
                "queries": answerable,
                "bm25_recall@10": bm25_recall,
                "encoders_present": labels,
                "rows": rows,
                "rankings": rankings,
                "pooled_vector_cosine": similarities,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {RESULTS / 'pooling.json'}  [{time.time() - started:.1f}s total]")


if __name__ == "__main__":
    main()
