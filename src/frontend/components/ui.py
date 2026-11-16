"""Shared UI pieces: global filters, coverage line, charts, post cards."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from src.frontend.components import api

PLATFORM_LABELS = {"reddit": "Reddit", "youtube": "YouTube", "bluesky": "Bluesky"}
SENTIMENTS = ("positive", "neutral", "negative")
POST_TYPE_LABELS = {"opinion": "Opinion", "news": "News", "market_data": "Market data",
                    "question": "Question", "other": "Other", "unclassified": "Unclassified"}

# Sentiment is a polarity, so it gets a diverging scheme: two opposed hues and a
# neutral grey. The pair is validated for colour-vision deficiency in both themes.
# Colour is never the only cue: every chart has a legend and labelled hover values.
_PALETTE = {
    "light": {"positive": "#2a78d6", "neutral": "#c3c2b7", "negative": "#e34948",
              "single": "#2a78d6", "grid": "#e1e0d9", "ink": "#52514e"},
    "dark": {"positive": "#3987e5", "neutral": "#52514e", "negative": "#e66767",
             "single": "#3987e5", "grid": "#2c2c2a", "ink": "#c3c2b7"},
}


def palette() -> dict[str, str]:
    try:
        return _PALETTE["dark" if st.context.theme.type == "dark" else "light"]
    except Exception:
        return _PALETTE["light"]


# ------------------------------------------------------------------ filters

def sidebar_filters() -> dict[str, Any]:
    """Company, date range and platform. Shared by every page through session state."""
    companies = api.get("/companies")
    by_ticker = {c["ticker"]: c for c in companies}
    with st.sidebar:
        st.header("Scope")
        ticker = st.selectbox("Company", list(by_ticker), key="ticker",
                              format_func=lambda t: f"{t} · {by_ticker[t]['name']}")
        today = datetime.now(timezone.utc).date()
        preset = st.radio("Period", ["Last 7 days", "Last 30 days", "Last 90 days", "Last 365 days", "Custom"],
                          index=2, key="period", horizontal=False)
        if preset == "Custom":
            picked = st.date_input("Date range", (today - timedelta(days=29), today), max_value=today, key="dates")
            start, end = (picked if isinstance(picked, tuple) and len(picked) == 2 else (today - timedelta(days=29), today))
        else:
            days = int(preset.split()[1])
            start, end = today - timedelta(days=days - 1), today
        platform = st.selectbox("Platform", ["all", *PLATFORM_LABELS], key="platform",
                                format_func=lambda p: "All platforms" if p == "all" else PLATFORM_LABELS[p])
        opinions_only = st.toggle(
            "Investor opinions only", key="opinions_only",
            help="Leave out news headlines, automated market data and plain questions, and count only "
                 "posts where someone gives their own view. Post types are assigned by the AI model.")
    return {"ticker": ticker, "company": by_ticker[ticker], "start": start.isoformat(),
            "end": end.isoformat(), "platform": None if platform == "all" else platform,
            "opinions_only": opinions_only}


def scope_params(scope: dict[str, Any]) -> dict[str, Any]:
    return {"start": scope["start"], "end": scope["end"], "platform": scope["platform"],
            "opinions_only": True if scope.get("opinions_only") else None}


def scope_text(scope: dict[str, Any]) -> str:
    """'1 Sep to 29 Nov 2026 · all platforms · all posts' for page headers."""
    start, end = date.fromisoformat(scope["start"]), date.fromisoformat(scope["end"])
    platforms = PLATFORM_LABELS[scope["platform"]] if scope["platform"] else "All platforms"
    content = "Investor opinions only" if scope.get("opinions_only") else "All relevant posts"
    return f"{start:%d %b} – {end:%d %b %Y} · {platforms} · {content}"


# ------------------------------------------------------------------- styling

def apply_style() -> None:
    """A light layer of CSS: calmer type, card styling for findings, tidier metrics."""
    st.markdown("""
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1280px;}
      h1 {font-weight: 650; letter-spacing: -0.01em;}
      [data-testid="stMetricValue"] {font-size: 1.9rem; font-weight: 600;}
      [data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.8;}
      .scope-line {font-size: 0.9rem; opacity: 0.75; margin-top: -0.6rem; margin-bottom: 1rem;}
      .headline {font-size: 1.12rem; line-height: 1.55; padding: 1rem 1.2rem; border-radius: 10px;
                 border-left: 5px solid var(--accent, #2a78d6); background: rgba(42,120,214,0.07);}
      .finding {border-radius: 10px; padding: 0.75rem 0.95rem; margin-bottom: 0.6rem;
                border: 1px solid rgba(128,128,128,0.22); border-left-width: 5px;}
      .finding b {display: block; margin-bottom: 0.15rem;}
      .finding span {font-size: 0.92rem; opacity: 0.9;}
      .finding.watch {border-left-color: #e34948;}
      .finding.positive {border-left-color: #2a78d6;}
      .finding.info {border-left-color: #8a8980;}
      .finding.caveat {border-left-color: #d29b2a;}
      .pill {display: inline-block; padding: 0.1rem 0.55rem; border-radius: 999px; font-size: 0.8rem;
             border: 1px solid rgba(128,128,128,0.35); margin-right: 0.3rem;}
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, scope: dict[str, Any]) -> None:
    st.title(title)
    st.markdown(f"<div class='scope-line'>{escape(scope['company']['name'])} ({scope['ticker']}) · "
                f"{escape(scope_text(scope))}</div>", unsafe_allow_html=True)


def escape(value: str) -> str:
    return (value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def link(page: str, label: str, icon: str) -> None:
    """Link to another page; silently skipped when a page runs outside the app's navigation (tests)."""
    try:
        st.page_link(page, label=label, icon=icon)
    except st.errors.StreamlitAPIException:
        pass


def finding(kind: str, title: str, detail: str) -> None:
    """One briefing finding as a coloured card. Text comes from our own rules, still escaped."""
    st.markdown(f"<div class='finding {kind}'><b>{escape(title)}</b><span>{escape(detail)}</span></div>",
                unsafe_allow_html=True)


# ----------------------------------------------------------------- coverage

def _ago(value: str | None) -> str:
    if not value:
        return "never"
    then = datetime.fromisoformat(value)
    hours = (datetime.now(timezone.utc) - then).total_seconds() / 3600
    if hours < 1:
        return "under an hour ago"
    return f"{hours:.0f} h ago" if hours < 48 else f"{hours / 24:.0f} days ago"


def coverage_line(scope: dict[str, Any], overview: dict[str, Any]) -> None:
    """Always state what the numbers are based on (docs/blueprint.md §15.2)."""
    platforms = PLATFORM_LABELS.get(scope["platform"], "all platforms") if scope["platform"] else "all platforms"
    st.caption(f"Based on **{overview['posts']} relevant posts** for {scope['company']['name']} "
               f"({scope['ticker']}), {scope['start']} to {scope['end']}, {platforms}. "
               "Public posts are not a representative sample of all shareholders.")
    if overview["posts"] and overview["small_sample"]:
        st.warning(f"Small sample: {overview['posts']} posts. Percentages and trends here can swing on a handful "
                   "of posts, so read the sources before drawing conclusions.", icon="⚠️")
    if overview.get("synthetic_posts"):
        st.warning(f"{overview['synthetic_posts']} of these posts are SYNTHETIC demonstration data (Reddit), "
                   "not real public content.", icon="🧪")


# ------------------------------------------------------------------ refresh

def refresh_panel() -> None:
    """Sidebar 'Data' panel: when data was last updated, and a button to fetch new posts.

    While a refresh runs, the panel re-runs itself every few seconds (a fragment, so the
    rest of the page is left alone). When the run ends it clears the cached API responses
    and re-runs the whole app, so every chart shows the new data.
    """
    watching = st.session_state.get("refresh_watching", False)
    st.fragment(_refresh_body, run_every="4s" if watching else None)()


def _refresh_body() -> None:
    st.markdown("**Data**")
    try:
        s = api.refresh_status()
    except api.ApiError:
        st.caption("Data status unavailable.")
        return
    st.caption(f"Last updated {_ago(s['last_updated'])}.")

    if s["state"] == "running":
        st.session_state["refresh_watching"] = True
        st.info("Collecting and analysing new posts. This takes a few minutes; "
                "you can keep using the dashboard.", icon="⏳")
        return

    if st.session_state.get("refresh_watching"):
        # The run just ended: drop cached responses and redraw everything.
        st.session_state["refresh_watching"] = False
        st.cache_data.clear()
        if s["state"] == "succeeded":
            n = s.get("new_posts") or 0
            st.session_state["refresh_toast"] = (f"Data updated: {n} new post{'s' if n != 1 else ''} collected."
                                                 if n else "Data updated: no new posts since the last update.")
        else:
            st.session_state["refresh_toast"] = "Refresh did not finish: " + (s.get("message") or "see server log")
        st.rerun(scope="app")

    left = s.get("cooldown_seconds_left") or 0
    if st.button("Refresh data", icon="🔄", disabled=bool(left), width="stretch",
                 help="Collect new posts from YouTube and Bluesky and analyse them."):
        try:
            api.start_refresh()
        except api.ApiError as exc:
            st.warning(str(exc))
            return
        st.session_state["refresh_watching"] = True
        st.rerun(scope="app")
    if left:
        st.caption(f"Available again in {max(1, -(-left // 60))} min.")


def show_refresh_toast() -> None:
    message = st.session_state.pop("refresh_toast", None)
    if message:
        st.toast(message, icon="✅" if message.startswith("Data updated") else "⚠️")


def freshness_table(rows: list[dict[str, Any]]) -> None:
    status_text = {"success": "✅ Collected", "partial": "⚠️ Partly collected", "failed": "❌ Failed",
                   "skipped": "⏸️ Not configured", "running": "⏳ Running", None: "— Never run"}
    st.dataframe(
        [{"Platform": PLATFORM_LABELS[r["platform"]],
          "Last run": ("🧪 Synthetic data only (no API access)" if r.get("last_run_synthetic")
                       else status_text.get(r["last_status"], r["last_status"])),
          "Last successful collection": _ago(r["last_success"]),
          "Posts stored": r["posts"],
          "Synthetic": r["synthetic_posts"] or "",
          "Problem": r["last_error"] or ""} for r in rows],
        hide_index=True, width="stretch",
    )


# ------------------------------------------------------------------- charts

def _layout(fig: go.Figure, height: int, **kwargs: Any) -> go.Figure:
    colors = palette()
    fig.update_layout(
        height=height, margin=dict(l=0, r=8, t=8, b=0), barmode="stack", bargap=0.35,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, traceorder="normal"),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=13, color=colors["ink"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="closest", **kwargs,
    )
    fig.update_xaxes(gridcolor=colors["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False)
    return fig


def sentiment_trend_chart(trend: list[dict[str, Any]]) -> go.Figure:
    """Daily post counts, stacked by sentiment. One axis: counts."""
    colors = palette()
    days = [r["day"] for r in trend]
    fig = go.Figure()
    for s in SENTIMENTS:
        fig.add_bar(name=s.capitalize(), x=days, y=[r[s] for r in trend], marker_color=colors[s],
                    marker_line_width=0, hovertemplate=f"%{{x|%d %b}}<br>{s.capitalize()}: %{{y}} posts<extra></extra>")
    fig = _layout(fig, 300)
    fig.update_xaxes(showgrid=False, tickformat="%d %b")
    fig.update_yaxes(title="Posts per day", rangemode="tozero", tickformat="d")
    return fig


def sentiment_share_chart(rows: list[dict[str, Any]], label_key: str, label_map: dict[str, str] | None = None) -> go.Figure:
    """Share of positive/neutral/negative per row, with the post count in the label."""
    colors = palette()
    rows = [r for r in rows if r["posts"]]
    labels = [f"{(label_map or {}).get(r[label_key], r[label_key])}  (n={r['posts']})" for r in rows]
    fig = go.Figure()
    for s in SENTIMENTS:
        fig.add_bar(name=s.capitalize(), y=labels, x=[r[f"{s}_pct"] for r in rows], orientation="h",
                    marker_color=colors[s], marker_line_width=0, customdata=[r[s] for r in rows],
                    text=[f"{r[f'{s}_pct']:.0f}%" if r[f"{s}_pct"] >= 8 else "" for r in rows],
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(color="#0b0b0b" if s == "neutral" and colors is _PALETTE["light"] else "#ffffff"),
                    hovertemplate=f"%{{y}}<br>{s.capitalize()}: %{{x:.1f}}% (%{{customdata}} posts)<extra></extra>")
    fig = _layout(fig, 70 + 46 * max(len(rows), 1))
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title=None)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def topic_volume_chart(topics: list[dict[str, Any]]) -> go.Figure:
    """Posts per topic, stacked by sentiment, largest first."""
    colors = palette()
    topics = [t for t in topics if t["posts"]]
    fig = go.Figure()
    for s in SENTIMENTS:
        fig.add_bar(name=s.capitalize(), y=[t["label"] for t in topics], x=[t[s] for t in topics],
                    orientation="h", marker_color=colors[s], marker_line_width=0,
                    hovertemplate=f"%{{y}}<br>{s.capitalize()}: %{{x}} posts<extra></extra>")
    fig = _layout(fig, 70 + 34 * max(len(topics), 1))
    fig.update_xaxes(title="Posts", tickformat="d", rangemode="tozero")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def show_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# -------------------------------------------------------------------- posts

_SENTIMENT_BADGE = {"positive": ":blue-badge[▲ Positive]", "neutral": ":gray-badge[● Neutral]",
                    "negative": ":red-badge[▼ Negative]"}
_EXCLUSION = {"low_quality": "low-quality content", "promotional": "promotional content",
              "potentially_automated": "potentially automated (repeated template)"}


def _count(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def post_card(post: dict[str, Any], show_relevance: bool = True) -> None:
    """One post with its labels and a link to the original, so labels can be checked."""
    with st.container(border=True):
        badges = [f":gray-badge[{PLATFORM_LABELS.get(post['platform'], post['platform'])} {post['content_type']}]"]
        if post.get("sentiment"):
            confidence = f" {post['sentiment_confidence']:.2f}" if post.get("sentiment_confidence") else ""
            badges.append(_SENTIMENT_BADGE[post["sentiment"]].replace("]", f"{confidence}]"))
        if post.get("post_type"):
            badges.append(f":violet-badge[{POST_TYPE_LABELS.get(post['post_type'], post['post_type'])}]")
        if post.get("topic_label"):
            badges.append(f":gray-badge[{post['topic_label']}]")
        if post.get("is_synthetic"):
            badges.append(":orange-badge[🧪 SYNTHETIC]")
        st.markdown(" ".join(badges))
        if post.get("title") and post["title"] != post["text"]:
            st.markdown("**" + post["title"].replace("*", "\\*").replace("[", "\\[") + "**")
        if post.get("parent_title"):
            st.caption("In reply to: " + post["parent_title"])
        # st.text renders the post literally: no markdown, links or HTML from untrusted content.
        body = post["text"]
        st.text(body if len(body) <= 600 else body[:600] + " …")
        meta = [post["created_at"][:10], f"👍 {_count(post['likes'])}", f"💬 {_count(post['comments'])}",
                f"🔁 {_count(post['shares'])}"]
        if post.get("views") is not None:
            meta.append(f"👁 {_count(post['views'])}")
        if show_relevance and post.get("relevance_score") is not None:
            meta.append(f"relevance {post['relevance_score']:.2f} ({post['relevance_method']})")
        if post["url"].startswith("http"):
            meta.append(f"[Open original ↗]({post['url']})")
        st.caption(" · ".join(meta))
        if post.get("sentiment_reason"):
            why = f"Why {post['sentiment']}: {post['sentiment_reason']}"
            finbert = post.get("finbert_sentiment")
            if finbert and finbert != post["sentiment"]:
                why += f" (the FinBERT baseline said {finbert})"
            st.caption(why)
        if show_relevance and not post.get("in_analytics", True):
            reasons = []
            if post.get("relevance_label") != "relevant":
                reasons.append(f"{post['relevance_label'].replace('_', ' ')}: {post.get('relevance_reason') or ''}")
            if post.get("quality_flag") in _EXCLUSION:
                reasons.append(_EXCLUSION[post["quality_flag"]])
            if post.get("duplicate_of"):
                reasons.append(f"duplicate of post {post['duplicate_of']}")
            if post.get("language") != "en":
                reasons.append(f"language '{post.get('language')}' not supported")
            st.caption("Not counted in metrics: " + "; ".join(reasons))
