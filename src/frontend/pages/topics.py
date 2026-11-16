"""Topics and aspects: what people discuss, how they feel about each part of the business, and the posts behind it."""
from __future__ import annotations

import streamlit as st

from src.frontend.components import api, ui

scope = st.session_state["scope"]
ui.page_header("Topics and aspects", scope)

try:
    data = api.get(f"/analytics/{scope['ticker']}/topics", **ui.scope_params(scope))
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

ui.coverage_line(scope, data["overview"])
topics = [t for t in data["topics"] if t["posts"]]
if not topics:
    st.info("No relevant posts in this scope.")
    st.stop()

st.subheader("What people discuss")
ui.show_chart(ui.topic_volume_chart(topics))

st.dataframe(
    [{"Topic": t["label"], "Posts": t["posts"], "Positive": t["positive"], "Neutral": t["neutral"],
      "Negative": t["negative"], "Previous period": t["previous_posts"],
      "Change": t["change_display"] + ("" if t["change_significant"] else "  (too few posts to call a trend)")}
     for t in topics], hide_index=True, width="stretch")
st.caption("Topics are assigned from a fixed list per company, by keyword and by similarity to the topic "
           "description. 'Other' holds posts that fit none. Change compares with the previous period of equal length.")

st.subheader("How they feel about each part of the business")
try:
    aspects = api.get(f"/analytics/{scope['ticker']}/aspects", **ui.scope_params(scope))["aspects"]
except api.ApiError as exc:
    aspects = []
    st.caption(f"Aspect sentiment unavailable: {exc}")
if aspects:
    ui.show_chart(ui.sentiment_share_chart(aspects, "label"))
    st.caption("Aspect-based sentiment: the AI model notes which business aspects a post talks about and the "
               "stance on each. One post can be positive on growth and negative on debt, so aspect counts do "
               "not add up to the post total. Read small counts with care.")
else:
    st.caption("No aspect-level views in this scope.")

st.subheader("Read the posts behind a topic")
left, right = st.columns([2, 1])
chosen = left.selectbox("Topic", topics, format_func=lambda t: f"{t['label']} ({t['posts']})")
sentiment = right.selectbox("Sentiment", ["all", *ui.SENTIMENTS], format_func=str.capitalize)

try:
    posts = api.get(f"/analytics/{scope['ticker']}/topics/{chosen['topic']}/posts", limit=6,
                    sentiment=sentiment, **ui.scope_params(scope))
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

if not posts:
    st.caption("No posts match.")
for post in posts:
    ui.post_card(post, show_relevance=False)
if posts:
    st.caption("Most engaged posts first. Use the Content explorer to see every post on this topic.")
