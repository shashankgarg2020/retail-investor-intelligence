"""Method and reliability: what the numbers rest on, measured, in plain language."""
from __future__ import annotations

import streamlit as st

from src.frontend.components import api, ui

scope = st.session_state["scope"]
ui.page_header("Method and reliability", scope)

st.markdown(
    "This dashboard reads **public posts** about each company from YouTube and Bluesky (real APIs) and "
    "Reddit (synthetic demonstration posts: Reddit API access was not granted). Each post goes through "
    "the same steps before it can count in a chart:")
steps = st.columns(4)
steps[0].markdown("**1. Is it about the company?**  \nKeyword rules, then an AI check for unclear cases. "
                  "'VEON' the telecom versus 'Le'Veon' the footballer.")
steps[1].markdown("**2. Is it usable?**  \nEnglish only; duplicates, spam and repeated bot templates "
                  "are set aside (still viewable in the explorer).")
steps[2].markdown("**3. Good or bad for investors?**  \nAn AI model reads each post as a shareholder would: "
                  "positive, neutral or negative, plus which business aspects it mentions.")
steps[3].markdown("**4. Counted by the database**  \nEvery number is a database count. The AI never "
                  "estimates figures; the assistant only explains them.")

try:
    q = api.get("/quality")
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("How accurate is it?")
if not q.get("available"):
    st.info("No evaluation has been run yet. Run `python -m scripts.evaluate all`.")
    st.stop()

sentiment, relevance = q.get("sentiment", {}), q.get("relevance", {})
labelled_by = sentiment.get("labelled_by") or relevance.get("labelled_by") or ""
if "claude" in labelled_by.lower():
    st.warning(
        "The reference labels behind these scores were drafted by an AI model, not by a person. Treat the "
        "numbers as indicative. The comparison between models is biased toward the AI stance model, and the "
        "relevance rules were adjusted after looking at errors on this same sample, so relevance scores are "
        "optimistic. A fresh, human-labelled sample is the next step.", icon="⚠️")

cols = st.columns(3)
if "llm" in sentiment:
    cols[0].metric("Sentiment accuracy (AI stance model)", f"{sentiment['llm']['accuracy']:.0%}",
                   help=f"Share of {sentiment['llm']['n']} labelled posts where the label matched.")
if "finbert" in sentiment:
    cols[1].metric("Sentiment accuracy (FinBERT baseline)", f"{sentiment['finbert']['accuracy']:.0%}",
                   help="The finance-news model used before; kept for comparison and as a fallback.")
if "strict" in relevance:
    cols[2].metric("Relevance filter F1", f"{relevance['strict']['f1']:.0%}",
                   help=f"Precision {relevance['strict']['precision']:.0%}, recall {relevance['strict']['recall']:.0%} "
                        f"on {relevance['n']} labelled posts.")

if "llm" in sentiment and "finbert" in sentiment:
    st.markdown("**Why the AI stance model replaced FinBERT.** FinBERT learned from news sentences, so it marks "
                "anything that reads like a report as neutral. On the same labelled posts it caught "
                f"{sentiment['finbert']['per_class']['positive']['recall']:.0%} of positive and "
                f"{sentiment['finbert']['per_class']['negative']['recall']:.0%} of negative posts; the AI stance "
                f"model caught {sentiment['llm']['per_class']['positive']['recall']:.0%} and "
                f"{sentiment['llm']['per_class']['negative']['recall']:.0%}.")
    with st.expander("Detailed sentiment scores"):
        rows = []
        for name, key in (("AI stance model", "llm"), ("FinBERT baseline", "finbert")):
            s = sentiment[key]
            rows.append({"Model": name, "Posts": s["n"], "Accuracy": f"{s['accuracy']:.2f}",
                         "Macro F1": f"{s['macro_f1']:.2f}",
                         **{f"{c.capitalize()} recall": f"{s['per_class'][c]['recall']:.2f}"
                            for c in ("positive", "neutral", "negative")}})
        if "majority_baseline" in sentiment:
            b = sentiment["majority_baseline"]
            rows.append({"Model": f"Always '{b['label']}'", "Accuracy": f"{b['accuracy']:.2f}"})
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("Macro F1 averages the three classes equally, so a model cannot score well by calling "
                   "everything neutral.")

if "per_company" in relevance:
    with st.expander("Relevance filter by company"):
        st.dataframe([{"Company": t, "Posts": v["n"], "Precision": f"{v['precision']:.2f}",
                       "Recall": f"{v['recall']:.2f}", "F1": f"{v['f1']:.2f}"}
                      for t, v in relevance["per_company"].items()], hide_index=True, width="stretch")
        st.caption("Precision: of the posts kept, how many are really about the company. "
                   "Recall: of the posts really about the company, how many were kept.")

retrieval = q.get("retrieval")
if retrieval:
    with st.expander("AI assistant: does it find the right posts?"):
        st.dataframe([{"Search method": m, f"Precision@{retrieval['k']}": f"{v['precision']:.2f}",
                       f"Recall@{retrieval['k']}": f"{v['recall']:.2f}", "MRR": f"{v['mrr']:.2f}"}
                      for m, v in retrieval["methods"].items()], hide_index=True, width="stretch")
        st.caption(f"{retrieval['questions']} test questions, each with the posts that answer it marked in advance. "
                   "Precision@K: share of the top results that are relevant. MRR: how high the first relevant "
                   "post ranks (1.0 = always first).")

latency = q.get("latency_ms")
if latency:
    with st.expander("Speed"):
        st.dataframe([{"Step": k, "Milliseconds": v} for k, v in latency.items()],
                     hide_index=True, width="stretch")
        st.caption("Dashboard queries take tens of milliseconds. An assistant answer takes about 5 to 9 seconds, "
                   "almost all of it the AI model writing the answer.")

st.subheader("What this cannot tell you")
st.markdown(
    "- People who post publicly are **not a representative sample** of shareholders.\n"
    "- Posts are **opinions and claims**, not verified facts.\n"
    "- Small samples swing easily: the dashboard warns below 30 posts.\n"
    "- Engagement (likes, shares) is not agreement and is not comparable across platforms.\n"
    "- Only English posts are analysed; other languages are collected but not scored.")
st.caption(f"Scores last updated {q.get('updated_at', 'unknown')}.")
