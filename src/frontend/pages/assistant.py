"""AI assistant: answers grounded in the collected posts, with citations and limitations."""
from __future__ import annotations

from typing import Any

import streamlit as st

from src.frontend.components import api, ui

scope = st.session_state["scope"]
ticker = scope["ticker"]
ui.page_header("Ask the data", scope)
platform_text = ui.PLATFORM_LABELS.get(scope["platform"], "all platforms") if scope["platform"] else "all platforms"
st.caption(f"Answers use only the posts collected for **{scope['company']['name']} ({ticker})**, "
           f"{scope['start']} to {scope['end']}, {platform_text}. Counts and percentages come from database "
           "queries; the model explains them and summarises the retrieved posts. A question that names "
           "another company or period overrides the sidebar.")

SUGGESTIONS = [
    f"What are the most discussed topics about {ticker}?",
    "What are investors most concerned about?",
    "Compare negative sentiment across all three platforms.",
    "What changed over the last seven days?",
    f"What are the three biggest concerns investors have about {ticker}, and what evidence supports them?",
]
STANCE = {"positive": "▲ Positive", "negative": "▼ Negative", "mixed": "◆ Mixed", "neutral": "● Neutral"}

history: list[dict[str, Any]] = st.session_state.setdefault("chat_history", [])


def render_answer(r: dict[str, Any]) -> None:
    if r["insufficient_evidence"]:
        st.warning(r["answer"], icon="⚠️")
    else:
        st.markdown(r["answer"])
    by_id = {p["id"]: p for p in r["evidence"]}
    for theme in r["themes"]:
        st.markdown(f"**{theme['title']}** · {STANCE[theme['stance']]}")
        st.markdown(theme["summary"])
        with st.expander(f"Evidence: {len(theme['evidence_ids'])} post(s)"):
            for post_id in theme["evidence_ids"]:
                if post_id in by_id:
                    ui.post_card(by_id[post_id], show_relevance=False)
    s = r["scope"]
    st.caption(f"Scope: {s['company']}, {s['from']} to {s['to']}, platform {s['platform']}, "
               f"sentiment filter {s['sentiment_filter']} · {r['posts_analysed']} relevant posts in scope · "
               f"{r['retrieved_count']} retrieved · question type {r['intent']} · {r['generated_by']}")
    with st.expander("Limitations and how this was produced"):
        for item in r["limitations"]:
            st.markdown(f"- {item}")
        for note in r["planner_notes"]:
            st.markdown(f"- {note}")
        if r["dropped_citations"]:
            st.markdown(f"- {r['dropped_citations']} citation(s) pointed at posts that were not retrieved "
                        "and were removed.")
        if r["platform_breakdown"]:
            st.dataframe(
                [{"Platform": ui.PLATFORM_LABELS.get(p["platform"], p["platform"]), "Posts": p["posts"],
                  "Positive": p["positive"], "Neutral": p["neutral"], "Negative": p["negative"]}
                 for p in r["platform_breakdown"]], hide_index=True, width="stretch")
        timing = ", ".join(f"{k.replace('_', ' ')} {v:.1f}s" for k, v in r["timings_seconds"].items())
        usage = r.get("usage") or {}
        cost = f", about ${usage['cost_usd']:.4f}" if usage.get("cost_usd") is not None else ""
        st.caption(f"Timing: {timing}{cost}")


for turn in history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if "error" in turn:
            st.error(turn["error"])
        else:
            render_answer(turn["response"])

question = st.chat_input("Ask about the collected discussion")
if not history and not question:
    st.markdown("**Try one of these**")
    for i, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, key=f"suggestion-{i}"):
            question = suggestion

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        turn: dict[str, Any] = {"question": question}
        try:
            with st.spinner("Querying the database, retrieving posts and writing the answer…"):
                turn["response"] = api.chat(question, ticker, scope["start"], scope["end"], scope["platform"])
            render_answer(turn["response"])
        except api.ApiError as exc:
            turn["error"] = str(exc)
            st.error(turn["error"])
    history.append(turn)

if history and st.sidebar.button("Clear conversation"):
    st.session_state["chat_history"] = []
    st.rerun()
