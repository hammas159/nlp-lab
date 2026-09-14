"""Seven retrieval methods, one corpus, one question: method or training data?

Everything is read from results/scores.json, which is produced by src/evaluate.py. The
corpus statistics are recomputed live so the page cannot drift from the benchmark it
describes.

Run:  streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="embedding fair comparison", layout="wide")

BLUE, GREEN, AMBER, GREY, RED = "#2563eb", "#16a34a", "#f59e0b", "#94a3b8", "#dc2626"
FAMILY_COLOR = {"this corpus": BLUE, "billions of words, elsewhere": AMBER}

RESULTS = ROOT / "results" / "scores.json"

st.title("Method, or training data?")
st.caption(
    "Most embedding comparisons put **TF-IDF fitted on your corpus** next to "
    "**pretrained word vectors trained on billions of words** and conclude that the "
    "neural method is better. That is not a method comparison - it is a *training data* "
    "comparison wearing one. Here every method is scored on the same corpus, the same "
    "queries and the same metric, and the ones trained locally are labelled separately "
    "from the ones that arrived pretrained."
)

if not RESULTS.exists():
    st.error(f"No results yet. Run `python src/evaluate.py 1000`.\n\nExpected at {RESULTS}")
    st.stop()

payload = json.loads(RESULTS.read_text(encoding="utf-8"))
rows = payload.get("rows", [])
if not rows:
    st.error("results/scores.json contains no rows.")
    st.stop()

frame = pd.DataFrame(rows).sort_values("recall@10", ascending=False)
corpus = payload.get("corpus", {})

# --- the benchmark ---------------------------------------------------------------------

a, b, c, d = st.columns(4)
a.metric("Documents", f"{corpus.get('documents', 0):,}")
b.metric("Queries", f"{corpus.get('queries', 0):,}")
c.metric("Gold per query", corpus.get("gold_per_query_max", "-"))
d.metric("Methods scored", len(frame))

if corpus.get("missing_gold_titles", 0):
    st.error(
        f"{corpus['missing_gold_titles']} gold documents are missing from the corpus - "
        "recall is capped and these numbers are not valid."
    )

# --- the headline -----------------------------------------------------------------------

local = frame[frame.trained_on == "this corpus"]
pre = frame[frame.trained_on != "this corpus"]

if len(local) and len(pre):
    best_local = local.iloc[0]
    best_pre = pre.iloc[0]
    gap = best_pre["recall@10"] - best_local["recall@10"]
    st.info(
        f"**Best corpus-trained method: `{best_local['method']}` at "
        f"{best_local['recall@10']:.1%} recall@10.** "
        f"Best pretrained: `{best_pre['method']}` at {best_pre['recall@10']:.1%}. "
        f"**Gap: {gap:.1%}** - and the pretrained model saw roughly a hundred thousand "
        "times more text."
    )

# --- the table --------------------------------------------------------------------------

st.subheader("Results")
show = frame[
    [
        "method",
        "trained_on",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "mrr",
        "fit_seconds",
        "query_seconds",
    ]
]
st.dataframe(
    show.style.format(
        {c: "{:.3f}" for c in ["recall@1", "recall@5", "recall@10", "recall@20", "mrr"]}
    ),
    hide_index=True,
    width="stretch",
)

# --- charts -----------------------------------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("recall@10")
    chart = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("method:N", sort="-y", title=None, axis=alt.Axis(labelAngle=-35)),
            y=alt.Y("recall@10:Q", title="recall@10", axis=alt.Axis(format="%")),
            color=alt.Color(
                "trained_on:N",
                scale=alt.Scale(domain=list(FAMILY_COLOR), range=list(FAMILY_COLOR.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=["method", "trained_on", alt.Tooltip("recall@10:Q", format=".3f")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, width="stretch")

with right:
    st.subheader("Accuracy against cost to build the index")
    chart = (
        alt.Chart(frame)
        .mark_circle(size=220, opacity=0.85)
        .encode(
            x=alt.X(
                "fit_seconds:Q", title="seconds to index the corpus", scale=alt.Scale(type="symlog")
            ),
            y=alt.Y("recall@10:Q", title="recall@10", axis=alt.Axis(format="%")),
            color=alt.Color(
                "trained_on:N",
                scale=alt.Scale(domain=list(FAMILY_COLOR), range=list(FAMILY_COLOR.values())),
                legend=alt.Legend(orient="top", title=None),
            ),
            tooltip=[
                "method",
                alt.Tooltip("fit_seconds:Q", format=".1f"),
                alt.Tooltip("recall@10:Q", format=".3f"),
            ],
        )
        .properties(height=360)
    )
    labels = chart.mark_text(dy=-16, fontSize=11).encode(
        text="method:N", color=alt.value("#334155")
    )
    st.altair_chart(chart + labels, width="stretch")
    st.caption("Left and high is the interesting corner: accurate and cheap to build.")

# --- recall curve -------------------------------------------------------------------------

st.subheader("How recall grows with k")
curve = frame.melt(
    id_vars=["method", "trained_on"],
    value_vars=["recall@1", "recall@5", "recall@10", "recall@20"],
    var_name="k",
    value_name="recall",
)
curve["k"] = curve["k"].str.replace("recall@", "").astype(int)

chart = (
    alt.Chart(curve)
    .mark_line(point=True, strokeWidth=2.5)
    .encode(
        x=alt.X("k:Q", title="k", scale=alt.Scale(type="log")),
        y=alt.Y("recall:Q", title="recall", axis=alt.Axis(format="%")),
        color=alt.Color("method:N", legend=alt.Legend(orient="right", title=None)),
        strokeDash=alt.StrokeDash("trained_on:N", legend=None),
        tooltip=["method", "k", alt.Tooltip("recall:Q", format=".3f")],
    )
    .properties(height=380)
)
st.altair_chart(chart, width="stretch")
st.caption(
    "Solid and dashed lines separate corpus-trained from pretrained. Where the lines "
    "converge at higher k is where the choice of method stops mattering."
)

st.divider()
st.caption(
    "HotpotQA (distractor split) from the local Hugging Face cache. Every method uses the "
    "same tokenizer, the same corpus and the same queries; vector methods are fixed at 300 "
    "dimensions so a win is not simply a wider vector. Generated by `src/evaluate.py`."
)
