"""Emerging concerns: what an IR team might want to look into first.

The blueprint describes a composite Investor Concern Score. Its weights have not
been validated, so the MVP shows the components side by side (volume, growth,
negative share, platform breadth) and does not collapse them into one number.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import Connection, text

from src.analytics import metrics
from src.analytics.metrics import Filters

MIN_NEGATIVE_POSTS = 2
MIN_NEGATIVE_SHARE = 30.0  # percent


def emerging_concerns(conn: Connection, f: Filters, limit: int = 5) -> list[dict[str, Any]]:
    """Topics that are growing and carry a meaningful share of negative posts."""
    clause, params = metrics.where(f)
    breadth = {r.topic: r.platforms for r in conn.execute(text(f"""
        SELECT topic, count(DISTINCT platform) AS platforms
        FROM analysis_posts WHERE {clause} GROUP BY topic
    """), params)}

    concerns = []
    for row in metrics.topics(conn, f):
        if row["topic"] == "other" or not row["change_significant"]:
            continue
        if row["change_direction"] not in ("up", "new"):
            continue
        if row["negative"] < MIN_NEGATIVE_POSTS or (row["negative_pct"] or 0) < MIN_NEGATIVE_SHARE:
            continue
        concerns.append({
            "topic": row["topic"], "label": row["label"],
            "volume": row["posts"], "previous_volume": row["previous_posts"],
            "growth": row["change_display"], "growth_absolute": row["change_absolute"],
            "negative_posts": row["negative"], "negative_pct": row["negative_pct"],
            "platform_breadth": breadth.get(row["topic"], 0),
        })
    concerns.sort(key=lambda c: (c["growth_absolute"], c["negative_posts"]), reverse=True)
    return concerns[:limit]
