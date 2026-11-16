"""Executive briefing: the one page a stakeholder reads. Findings first, evidence one click away."""
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from src.frontend.components import api, ui

scope = st.session_state["scope"]
ui.page_header("Executive briefing", scope)

try:
    brief = api.get(f"/analytics/{scope['ticker']}/briefing", **ui.scope_params(scope))
    aspects = api.get(f"/analytics/{scope['ticker']}/aspects", **ui.scope_params(scope))["aspects"]
    topics = api.get(f"/analytics/{scope['ticker']}/topics", **ui.scope_params(scope))
except api.ApiError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(f"<div class='headline'>{ui.escape(brief['headline'])}</div>", unsafe_allow_html=True)
st.write("")

k = brief["kpis"]
if k["posts"]:
    def delta(value, suffix=""):
        return None if value is None else f"{value:+.2f}{suffix}" if not suffix else f"{value:+.0f}{suffix}"

    tiles = st.columns(4)
    tiles[0].metric("Relevant posts", f"{k['posts']:,}",
                    delta=None if not k["posts_previous"] else f"{k['posts'] - k['posts_previous']:+d} vs previous",
                    delta_color="off", help="Posts judged to be about the company, in English, not duplicates.")
    tiles[1].metric("Net sentiment", f"{k['net']:+.2f}", delta=delta(k["net_delta"]),
                    help="Average of +1 (positive), 0 (neutral) and -1 (negative) across posts. "
                         "Change is against the previous period of equal length.")
    tiles[2].metric("Positive", f"{k['positive_pct']:.0f}%", delta=delta(k["positive_pct_delta"], " pts"))
    tiles[3].metric("Negative", f"{k['negative_pct']:.0f}%", delta=delta(k["negative_pct_delta"], " pts"),
                    delta_color="inverse")

points = brief["points"]
watch = [p for p in points if p["kind"] == "watch"]
good = [p for p in points if p["kind"] == "positive"]
context = [p for p in points if p["kind"] in ("info", "caveat")]

left, middle, right = st.columns(3)
with left:
    st.subheader("What to watch")
    for p in watch:
        ui.finding("watch", p["title"], p["detail"])
    if not watch:
        st.caption("No growing or clearly negative theme in this period.")
with middle:
    st.subheader("What is going well")
    for p in good:
        ui.finding("positive", p["title"], p["detail"])
    if not good:
        st.caption("No clearly positive theme with enough posts in this period.")
with right:
    st.subheader("Context")
    for p in context:
        ui.finding(p["kind"], p["title"], p["detail"])

st.divider()
chart_left, chart_right = st.columns(2)
with chart_left:
    st.subheader("Sentiment by business aspect")
    shown = [a for a in aspects if a["posts"] >= 2]
    if shown:
        ui.show_chart(ui.sentiment_share_chart(shown, "label"))
        st.caption("Each bar is one aspect of the business; a post that discusses two aspects counts in both. "
                   "Aspects with fewer than 2 mentions are hidden.")
    else:
        st.caption("Too few posts discuss specific aspects in this period.")
with chart_right:
    st.subheader("Most discussed topics")
    ranked = [t for t in topics["topics"] if t["posts"]][:6]
    if ranked:
        ui.show_chart(ui.topic_volume_chart(ranked))
    else:
        st.caption("No topics in this period.")

ui.link("pages/topics.py", "Read the posts behind each topic and aspect", "🧭")
ui.link("pages/assistant.py", "Ask a follow-up question", "💬")

# A copy of the briefing to paste into an email or a board pack.
lines = [f"# {scope['company']['name']} ({scope['ticker']}): retail investor briefing",
         f"_{ui.scope_text(scope)}. Generated {datetime.now(timezone.utc):%d %b %Y %H:%M} UTC._", "",
         brief["headline"], ""]
for heading, group in (("What to watch", watch), ("What is going well", good), ("Context", context)):
    if group:
        lines.append(f"## {heading}")
        lines += [f"- **{p['title']}.** {p['detail']}" for p in group]
        lines.append("")
lines.append("_Public social posts are not a representative sample of all shareholders. "
             "Sentiment labels are model output; see Method and reliability in the dashboard._")
st.download_button("Download this briefing (Markdown)", "\n".join(lines),
                   file_name=f"{scope['ticker']}_briefing_{scope['end']}.md", mime="text/markdown")
st.caption("Findings are written by fixed rules from the numbers above, not by an AI model, so the same data "
           "always gives the same briefing. Public posts are not a representative sample of all shareholders.")
