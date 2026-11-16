"""Content explorer: every stored post, filterable, each with a link to its source."""
from __future__ import annotations

import streamlit as st

from src.frontend.components import api, ui

PAGE_SIZE = 20
scope = st.session_state["scope"]
ui.page_header("Post explorer", scope)
st.caption("Every collected post, including the ones left out of the metrics and why. "
           "Each card links to the original so any label can be checked.")

row = st.columns([3, 2, 2, 2])
query = row[0].text_input("Search text", placeholder="e.g. debt, guidance, GTA VI")
topic_options = {"all": "All topics", **scope["company"]["topics"], "other": "Other"}
topic = row[1].selectbox("Topic", list(topic_options), format_func=topic_options.get)
sentiment = row[2].selectbox("Sentiment", ["all", *ui.SENTIMENTS], format_func=str.capitalize)
sort = row[3].selectbox("Sort by", ["newest", "engagement", "relevance"], format_func=str.capitalize)

row = st.columns([3, 2, 4])
relevance_labels = {"relevant": "Relevant", "possibly_relevant": "Possibly relevant",
                    "irrelevant": "Irrelevant", "all": "All collected"}
relevance = row[0].selectbox("Relevance", list(relevance_labels), format_func=relevance_labels.get,
                             help="Only 'Relevant' posts count in the dashboard metrics. The other options show "
                                  "what was filtered out and why.")
min_engagement = row[1].number_input("Minimum engagement", min_value=0, value=0, step=5,
                                     help="Likes + comments + shares")
analytics_only = row[2].checkbox("Only posts counted in metrics", value=False)

filter_key = (scope["ticker"], scope["start"], scope["end"], scope["platform"], scope.get("opinions_only"), query, topic,
              sentiment, sort, relevance, min_engagement, analytics_only)
if st.session_state.get("explorer_key") != filter_key:
    st.session_state["explorer_key"], st.session_state["explorer_page"] = filter_key, 0
page = st.session_state["explorer_page"]

try:
    result = api.get(f"/posts/{scope['ticker']}", q=query, topic=topic, sentiment=sentiment, sort=sort,
                     relevance=relevance, min_engagement=min_engagement or None,
                     analytics_only=analytics_only or None, limit=PAGE_SIZE, offset=page * PAGE_SIZE,
                     **ui.scope_params(scope))
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

total = result["total"]
if total == 0:
    st.info("No posts match these filters.")
    st.stop()

first, last = page * PAGE_SIZE + 1, min((page + 1) * PAGE_SIZE, total)
st.caption(f"Showing {first} to {last} of {total} posts.")
for post in result["posts"]:
    ui.post_card(post)

previous, _, following = st.columns([1, 4, 1])
if previous.button("← Previous", disabled=page == 0, width="stretch"):
    st.session_state["explorer_page"] -= 1
    st.rerun()
if following.button("Next →", disabled=last >= total, width="stretch"):
    st.session_state["explorer_page"] += 1
    st.rerun()
