"""Sentiment and platforms: volume, sentiment, trend, platforms, content mix, emerging concerns, freshness."""
from __future__ import annotations

import streamlit as st

from src.frontend.components import api, ui

scope = st.session_state["scope"]
ui.page_header("Sentiment and platforms", scope)

try:
    data = api.get(f"/analytics/{scope['ticker']}/overview", **ui.scope_params(scope))
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

o, coverage = data["overview"], data["coverage"]
ui.coverage_line(scope, o)

if o["posts"] == 0:
    st.info(f"No relevant posts in this scope. {coverage['collected']} posts were collected for this period; "
            "see 'What was collected' below for why none count, or widen the date range.")
else:
    tiles = st.columns(5)
    tiles[0].metric("Relevant posts", f"{o['posts']:,}")
    tiles[1].metric("Positive", f"{o['positive_pct']:.0f}%", help=f"{o['positive']} posts")
    tiles[2].metric("Neutral", f"{o['neutral_pct']:.0f}%", help=f"{o['neutral']} posts")
    tiles[3].metric("Negative", f"{o['negative_pct']:.0f}%", help=f"{o['negative']} posts")
    tiles[4].metric(
        "Net sentiment", f"{o['post_weighted_sentiment']:+.2f}",
        help="Average of +1 (positive), 0 (neutral), -1 (negative). Every post counts once.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Sentiment over time")
        ui.show_chart(ui.sentiment_trend_chart(data["trend"]))
    with right:
        st.subheader("By platform")
        if data["by_platform"]:
            ui.show_chart(ui.sentiment_share_chart(data["by_platform"], "platform", ui.PLATFORM_LABELS))
        missing = [ui.PLATFORM_LABELS[p] for p in ui.PLATFORM_LABELS
                   if p not in {r["platform"] for r in data["by_platform"]}]
        if missing:
            st.caption("No relevant posts in this period from: " + ", ".join(missing) + ".")

    types = data.get("post_types") or []
    if types:
        st.subheader("What kind of posts these are")
        ui.show_chart(ui.sentiment_share_chart(types, "post_type", ui.POST_TYPE_LABELS))
        st.caption("Assigned by the AI model. News and automated market data describe events; opinions show what "
                   "individuals think. Use 'Investor opinions only' in the sidebar to count opinions alone.")

    with st.expander("Engagement-weighted sentiment (secondary measure)"):
        weighted = o["engagement_weighted_sentiment"]
        st.metric("Engagement-weighted net sentiment", "n/a" if weighted is None else f"{weighted:+.2f}")
        st.caption("Each post is weighted by log(1 + likes + comments + shares). Engagement is not agreement, "
                   "and it is not comparable across platforms, so the default measure above weights every "
                   "post equally. Posts with no engagement carry no weight here.")

    st.subheader("Emerging concerns")
    concerns = data["emerging_concerns"]
    if concerns:
        st.dataframe(
            [{"Topic": c["label"], "Posts": c["volume"], "Previous period": c["previous_volume"],
              "Change": c["growth"], "Negative posts": c["negative_posts"],
              "Negative share": f"{c['negative_pct']:.0f}%", "Platforms": c["platform_breadth"]}
             for c in concerns], hide_index=True, width="stretch")
        st.caption("Topics that grew against the previous equal period and carry at least 30% negative posts. "
                   "Components are shown separately; they are not combined into a single score because "
                   "no weighting has been validated. Open Topic intelligence to read the posts.")
    else:
        st.caption("No topic is both growing and substantially negative in this period "
                   "(minimum 5 posts; changes on smaller counts are treated as noise).")

st.subheader("Data freshness")
ui.freshness_table(data["freshness"])

with st.expander("What was collected, and what was left out"):
    labels = {
        "collected": "Collected in this period", "analysed": "Counted in metrics (relevant, English, original)",
        "irrelevant": "Not about the company", "possibly_relevant": "Possibly relevant, unconfirmed",
        "low_quality_or_promotional": "Low-quality, promotional or potentially automated",
        "duplicates": "Duplicate text", "unsupported_language": "Not in English",
        "not_yet_processed": "Waiting to be processed",
    }
    st.dataframe([{"": labels[k], "Posts": coverage[k]} for k in labels], hide_index=True, width="stretch")
    st.caption("Excluded posts stay in the database and can be read in the Content explorer.")
