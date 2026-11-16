"""Executive briefing: the dashboard's findings written as plain sentences.

Built only from the SQL metrics, with fixed rules and thresholds; no LLM. Every
sentence quotes the numbers it rests on, so a stakeholder can check it against the
charts, and the same data always gives the same briefing.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import Connection

from config.settings import companies
from src.analytics import insights, metrics
from src.analytics.metrics import Filters

MIN_ASPECT_MENTIONS = 2     # an aspect needs this many mentions to be called a strength or worry
MIN_PLATFORM_POSTS = 5      # a platform needs this many posts to be compared
MIN_TONE_SHIFT = 0.10       # change in net sentiment worth reporting
PLATFORM_LABELS = {"reddit": "Reddit", "youtube": "YouTube", "bluesky": "Bluesky"}
TYPE_LABELS = {"opinion": "personal opinion", "news": "news headlines", "market_data": "automated market data",
               "question": "questions", "other": "other content", "unclassified": "unclassified"}


def tone(o: dict[str, Any]) -> str:
    """One word for the overall mood, from net sentiment and the positive/negative shares."""
    net = o.get("post_weighted_sentiment")
    if not o.get("posts") or net is None:
        return "no data"
    if net >= 0.15:
        return "mostly positive"
    if net <= -0.15:
        return "mostly negative"
    if (o.get("positive_pct") or 0) >= 20 and (o.get("negative_pct") or 0) >= 20:
        return "mixed"
    return "mostly neutral"


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}%"


def compose(name: str, current: dict[str, Any], previous: dict[str, Any], topics: list[dict[str, Any]],
            aspects: list[dict[str, Any]], concerns: list[dict[str, Any]], platforms: list[dict[str, Any]],
            types: list[dict[str, Any]], opinions_only: bool = False) -> dict[str, Any]:
    """Pure function: metrics in, briefing out. Kept separate from the database for testing."""
    points: list[dict[str, str]] = []
    n, prev_n = current.get("posts") or 0, previous.get("posts") or 0

    if n == 0:
        return {"headline": f"No relevant posts about {name} were found in this period.",
                "tone": "no data", "points": [], "kpis": _kpis(current, previous)}

    if prev_n:
        change = (n - prev_n) / prev_n * 100
        volume = f"{change:+.0f}% versus the previous period ({prev_n})"
    else:
        volume = "no posts in the previous period to compare with"
    headline = (f"{n} relevant posts about {name} ({volume}). The tone is {tone(current)}: "
                f"{_pct(current['positive_pct'])} positive, {_pct(current['negative_pct'])} negative, "
                f"net sentiment {current['post_weighted_sentiment']:+.2f} on a scale of -1 to +1.")

    if prev_n >= 5 and previous.get("post_weighted_sentiment") is not None:
        shift = current["post_weighted_sentiment"] - previous["post_weighted_sentiment"]
        if abs(shift) >= MIN_TONE_SHIFT:
            points.append({"kind": "positive" if shift > 0 else "watch",
                           "title": "Tone improved" if shift > 0 else "Tone worsened",
                           "detail": f"Net sentiment moved from {previous['post_weighted_sentiment']:+.2f} "
                                     f"to {current['post_weighted_sentiment']:+.2f} against the previous period."})

    for c in concerns[:2]:
        points.append({"kind": "watch", "title": f"Growing concern: {c['label']}",
                       "detail": f"{c['volume']} posts ({c['growth']}), {c['negative_pct']:.0f}% of them negative, "
                                 f"across {c['platform_breadth']} platform(s)."})

    mentioned = [a for a in aspects if a["posts"] >= MIN_ASPECT_MENTIONS]
    worries = sorted((a for a in mentioned if a["negative"] > a["positive"]), key=lambda a: (a["net"], -a["posts"]))
    strengths = sorted((a for a in mentioned if a["positive"] > a["negative"]), key=lambda a: (-a["net"], -a["posts"]))
    if worries:
        a = worries[0]
        points.append({"kind": "watch", "title": f"Main worry: {a['label']}",
                       "detail": f"{a['negative']} of {a['posts']} posts on this aspect are negative."})
    if strengths:
        a = strengths[0]
        points.append({"kind": "positive", "title": f"Main strength: {a['label']}",
                       "detail": f"{a['positive']} of {a['posts']} posts on this aspect are positive."})

    ranked = [t for t in topics if t["posts"] and t["topic"] != "other"]
    if ranked:
        t = ranked[0]
        points.append({"kind": "info", "title": f"Most discussed: {t['label']}",
                       "detail": f"{t['posts']} posts, {_pct(t.get('negative_pct'))} negative; "
                                 f"change {t['change_display']}."})

    comparable = [p for p in platforms if p["posts"] >= MIN_PLATFORM_POSTS]
    if len(comparable) >= 2:
        def name(p: dict[str, Any]) -> str:
            label = PLATFORM_LABELS.get(p["platform"], p["platform"])
            # A platform made of demonstration data must never read as a real finding.
            return f"{label} (synthetic demo data)" if p.get("synthetic_posts") == p["posts"] else label
        worst = max(comparable, key=lambda p: p["negative_pct"] or 0)
        best = max(comparable, key=lambda p: p["positive_pct"] or 0)
        if worst["platform"] != best["platform"]:
            detail = (f"{name(worst)} is the most negative ({_pct(worst['negative_pct'])} negative); "
                      f"{name(best)} the most positive ({_pct(best['positive_pct'])} positive).")
        else:
            detail = (f"{name(worst)} has both the most negative ({_pct(worst['negative_pct'])}) and the most "
                      f"positive ({_pct(worst['positive_pct'])}) posts: views there are more polarised.")
        points.append({"kind": "info", "title": "Platforms differ", "detail": detail})

    typed = {t["post_type"]: t["posts"] for t in types}
    total_typed = sum(typed.values())
    if total_typed and not opinions_only:
        opinion_share = 100 * typed.get("opinion", 0) / total_typed
        biggest = max(typed, key=typed.get)
        if opinion_share < 50:
            points.append({"kind": "caveat", "title": "Mostly news, not opinion",
                           "detail": f"Only {opinion_share:.0f}% of posts are personal opinions; the largest group "
                                     f"is {TYPE_LABELS.get(biggest, biggest)}. Switch on 'Investor opinions only' "
                                     "to see what individuals think."})
    if current.get("small_sample"):
        points.append({"kind": "caveat", "title": "Small sample",
                       "detail": f"{n} posts is too few for firm conclusions; read the posts behind each finding."})
    if current.get("synthetic_posts"):
        points.append({"kind": "caveat", "title": "Includes synthetic data",
                       "detail": f"{current['synthetic_posts']} posts are synthetic demonstration data, not real content."})

    return {"headline": headline, "tone": tone(current), "points": points, "kpis": _kpis(current, previous)}


def _kpis(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    def delta(key: str) -> float | None:
        a, b = current.get(key), previous.get(key)
        return None if a is None or b is None or not previous.get("posts") else round(a - b, 2)
    return {
        "posts": current.get("posts") or 0, "posts_previous": previous.get("posts") or 0,
        "net": current.get("post_weighted_sentiment"), "net_delta": delta("post_weighted_sentiment"),
        "positive_pct": current.get("positive_pct"), "positive_pct_delta": delta("positive_pct"),
        "negative_pct": current.get("negative_pct"), "negative_pct_delta": delta("negative_pct"),
        "neutral_pct": current.get("neutral_pct"),
    }


def build(conn: Connection, f: Filters) -> dict[str, Any]:
    current = metrics.overview(conn, f)
    previous = metrics.overview(conn, f.previous_period())
    return compose(
        companies()[f.ticker]["name"], current, previous, metrics.topics(conn, f), metrics.aspects(conn, f),
        insights.emerging_concerns(conn, f), metrics.by_platform(conn, f), metrics.post_types(conn, f),
        opinions_only=f.opinions_only,
    )
