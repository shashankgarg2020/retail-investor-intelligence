"""Deterministic analytics. Every number the product shows is computed here, in SQL.

All company metrics read the `analysis_posts` view, which already excludes
irrelevant, low-quality, duplicate and unsupported-language posts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from sqlalchemy import Connection, text

from config.settings import companies
from src.intelligence.topics import topic_label
from src.intelligence.trends import period_change

SMALL_SAMPLE = 30  # below this the UI and the assistant warn about sample size


@dataclass(frozen=True)
class Filters:
    ticker: str
    start: datetime          # inclusive
    end: datetime            # exclusive
    platform: str | None = None
    sentiment: str | None = None
    topic: str | None = None
    opinions_only: bool = False  # only posts the LLM typed as the author's own view

    def previous_period(self) -> "Filters":
        """The equally long window immediately before this one."""
        return replace(self, start=self.start - (self.end - self.start), end=self.start)


def where(f: Filters, alias: str = "") -> tuple[str, dict[str, Any]]:
    a = f"{alias}." if alias else ""
    clauses = [f"{a}ticker = :ticker", f"{a}created_at >= :start", f"{a}created_at < :end"]
    params: dict[str, Any] = {"ticker": f.ticker, "start": f.start, "end": f.end}
    for column in ("platform", "sentiment", "topic"):
        value = getattr(f, column)
        if value:
            clauses.append(f"{a}{column} = :{column}")
            params[column] = value
    if f.opinions_only:
        clauses.append(f"{a}post_type = 'opinion'")
    return " AND ".join(clauses), params


_AGGREGATES = """
    count(*)                                             AS posts,
    count(*) FILTER (WHERE sentiment = 'positive')       AS positive,
    count(*) FILTER (WHERE sentiment = 'neutral')        AS neutral,
    count(*) FILTER (WHERE sentiment = 'negative')       AS negative,
    round(avg(sentiment_value)::numeric, 3)              AS post_weighted_sentiment,
    round((sum(engagement_weight * sentiment_value)
           / nullif(sum(engagement_weight), 0))::numeric, 3) AS engagement_weighted_sentiment,
    count(*) FILTER (WHERE is_synthetic)                 AS synthetic_posts
"""


def _row(row) -> dict[str, Any]:
    out = dict(row._mapping)
    for key in ("post_weighted_sentiment", "engagement_weighted_sentiment"):
        if out.get(key) is not None:
            out[key] = float(out[key])
    posts = out.get("posts") or 0
    for label in ("positive", "neutral", "negative"):
        if label in out:
            out[f"{label}_pct"] = round(100 * out[label] / posts, 1) if posts else None
    return out


def overview(conn: Connection, f: Filters) -> dict[str, Any]:
    clause, params = where(f)
    row = conn.execute(text(f"""
        SELECT {_AGGREGATES},
               min(created_at) AS first_post, max(created_at) AS last_post,
               count(DISTINCT platform) AS platforms_with_data
        FROM analysis_posts WHERE {clause}
    """), params).one()
    out = _row(row)
    out["small_sample"] = out["posts"] < SMALL_SAMPLE
    return out


def by_platform(conn: Connection, f: Filters) -> list[dict[str, Any]]:
    clause, params = where(f)
    rows = conn.execute(text(f"""
        SELECT platform, {_AGGREGATES}
        FROM analysis_posts WHERE {clause}
        GROUP BY platform ORDER BY posts DESC
    """), params)
    return [_row(r) for r in rows]


def sentiment_trend(conn: Connection, f: Filters) -> list[dict[str, Any]]:
    clause, params = where(f)
    rows = conn.execute(text(f"""
        SELECT date_trunc('day', created_at)::date AS day, {_AGGREGATES}
        FROM analysis_posts WHERE {clause}
        GROUP BY 1 ORDER BY 1
    """), params)
    return [_row(r) for r in rows]


def topics(conn: Connection, f: Filters) -> list[dict[str, Any]]:
    """Topic ranking with sentiment mix and change against the previous equal period."""
    clause, params = where(f)
    current = {r.topic: _row(r) for r in conn.execute(text(f"""
        SELECT topic, {_AGGREGATES} FROM analysis_posts WHERE {clause} GROUP BY topic
    """), params)}
    prev_clause, prev_params = where(f.previous_period())
    previous = {r.topic: r.posts for r in conn.execute(text(f"""
        SELECT topic, count(*) AS posts FROM analysis_posts WHERE {prev_clause} GROUP BY topic
    """), prev_params)}

    out = []
    for topic in set(current) | set(previous):
        row = current.get(topic) or {"topic": topic, "posts": 0, "positive": 0, "neutral": 0,
                                     "negative": 0, "negative_pct": None}
        change = period_change(row["posts"], previous.get(topic, 0))
        out.append({
            **row, "label": topic_label(f.ticker, topic),
            "previous_posts": change.previous, "change_absolute": change.absolute,
            "change_percent": None if change.percent is None else round(change.percent, 1),
            "change_direction": change.direction, "change_significant": change.significant,
            "change_display": change.display,
        })
    return sorted(out, key=lambda r: r["posts"], reverse=True)


def aspects(conn: Connection, f: Filters) -> list[dict[str, Any]]:
    """Aspect-based sentiment (docs/blueprint.md §8.2): one row per business aspect with its stance mix.

    A post can address several aspects with different stances ("revenue is great, debt is
    worrying"), so these counts do not add up to the post total.
    """
    labels = companies()[f.ticker].get("aspects", {})
    clause, params = where(f, "ap")
    rows = conn.execute(text(f"""
        SELECT s.aspect, count(*) AS posts,
               count(*) FILTER (WHERE s.sentiment = 'positive') AS positive,
               count(*) FILTER (WHERE s.sentiment = 'neutral')  AS neutral,
               count(*) FILTER (WHERE s.sentiment = 'negative') AS negative,
               count(DISTINCT ap.platform) AS platforms
        FROM aspect_sentiment s JOIN analysis_posts ap ON ap.id = s.post_id
        WHERE {clause}
        GROUP BY s.aspect ORDER BY posts DESC
    """), params)
    out = []
    for r in rows:
        row = _row(r)
        row["label"] = labels.get(row["aspect"], row["aspect"].replace("_", " ").capitalize())
        row["net"] = round((row["positive"] - row["negative"]) / row["posts"], 2) if row["posts"] else None
        out.append(row)
    return out


def post_types(conn: Connection, f: Filters) -> list[dict[str, Any]]:
    """How much of the discussion is opinion versus news, market data or questions."""
    clause, params = where(replace(f, opinions_only=False))
    rows = conn.execute(text(f"""
        SELECT coalesce(post_type, 'unclassified') AS post_type, {_AGGREGATES}
        FROM analysis_posts WHERE {clause}
        GROUP BY 1 ORDER BY posts DESC
    """), params)
    return [_row(r) for r in rows]


def coverage(conn: Connection, ticker: str, start: datetime, end: datetime) -> dict[str, int]:
    """What was collected and why each post did or did not reach analytics."""
    row = conn.execute(text("""
        SELECT
            count(*) AS collected,
            count(*) FILTER (WHERE a.post_id IS NULL) AS not_yet_processed,
            count(*) FILTER (WHERE a.duplicate_of IS NOT NULL) AS duplicates,
            count(*) FILTER (WHERE a.duplicate_of IS NULL AND a.quality_flag <> 'ok') AS low_quality_or_promotional,
            count(*) FILTER (WHERE a.duplicate_of IS NULL AND a.quality_flag = 'ok'
                             AND p.language <> 'en') AS unsupported_language,
            count(*) FILTER (WHERE a.duplicate_of IS NULL AND a.quality_flag = 'ok' AND p.language = 'en'
                             AND a.relevance_label = 'irrelevant') AS irrelevant,
            count(*) FILTER (WHERE a.duplicate_of IS NULL AND a.quality_flag = 'ok' AND p.language = 'en'
                             AND a.relevance_label = 'possibly_relevant') AS possibly_relevant,
            count(*) FILTER (WHERE a.duplicate_of IS NULL AND a.quality_flag = 'ok' AND p.language = 'en'
                             AND a.relevance_label = 'relevant') AS analysed
        FROM posts p
        JOIN companies c ON c.id = p.company_id
        LEFT JOIN post_analysis a ON a.post_id = p.id
        WHERE c.ticker = :ticker AND p.created_at >= :start AND p.created_at < :end
    """), {"ticker": ticker, "start": start, "end": end}).one()
    return dict(row._mapping)


def freshness(conn: Connection, ticker: str) -> list[dict[str, Any]]:
    """Per platform: latest run, last successful collection, any error."""
    rows = conn.execute(text("""
        WITH latest AS (
            SELECT DISTINCT ON (r.platform) r.platform, r.status, r.error, r.started_at,
                   r.fetched, r.inserted, r.is_synthetic
            FROM ingestion_runs r JOIN companies c ON c.id = r.company_id
            WHERE c.ticker = :ticker
            ORDER BY r.platform, r.started_at DESC
        ),
        success AS (
            SELECT r.platform, max(r.completed_at) AS last_success
            FROM ingestion_runs r JOIN companies c ON c.id = r.company_id
            WHERE c.ticker = :ticker AND r.status IN ('success', 'partial') AND r.is_synthetic = FALSE
            GROUP BY r.platform
        ),
        stored AS (
            SELECT p.platform, count(*) AS posts, count(*) FILTER (WHERE p.is_synthetic) AS synthetic_posts,
                   max(p.created_at) AS newest_post
            FROM posts p JOIN companies c ON c.id = p.company_id
            WHERE c.ticker = :ticker GROUP BY p.platform
        )
        SELECT pl.platform, l.status AS last_status, l.is_synthetic AS last_run_synthetic,
               l.error AS last_error, l.started_at AS last_run,
               s.last_success, coalesce(st.posts, 0) AS posts,
               coalesce(st.synthetic_posts, 0) AS synthetic_posts, st.newest_post
        FROM (VALUES ('reddit'), ('youtube'), ('bluesky')) AS pl(platform)
        LEFT JOIN latest l ON l.platform = pl.platform
        LEFT JOIN success s ON s.platform = pl.platform
        LEFT JOIN stored st ON st.platform = pl.platform
        ORDER BY pl.platform
    """), {"ticker": ticker})
    return [dict(r._mapping) for r in rows]


_POST_COLUMNS = """
    p.id, c.ticker, p.platform, p.content_type, p.title, p.text, p.parent_title, p.author_id,
    p.created_at, p.collected_at, p.url, p.likes, p.comments, p.shares, p.views, p.language,
    p.is_synthetic, a.relevance_label, a.relevance_score, a.relevance_method, a.relevance_reason,
    a.quality_flag, a.duplicate_of, a.sentiment, a.sentiment_confidence, a.sentiment_model,
    a.sentiment_reason, a.finbert_sentiment, a.post_type, a.topic,
    (coalesce(p.likes, 0) + coalesce(p.comments, 0) + coalesce(p.shares, 0)) AS engagement
"""

_SORTS = {
    "newest": "p.created_at DESC",
    "engagement": "engagement DESC, p.created_at DESC",
    "relevance": "a.relevance_score DESC NULLS LAST, p.created_at DESC",
}


def _post(row, ticker: str) -> dict[str, Any]:
    out = dict(row._mapping)
    out["topic_label"] = topic_label(ticker, out["topic"]) if out.get("topic") else None
    out["in_analytics"] = bool(
        out.get("relevance_label") == "relevant" and out.get("quality_flag") == "ok"
        and out.get("duplicate_of") is None and out.get("language") == "en" and out.get("sentiment"))
    return out


def search_posts(conn: Connection, f: Filters, *, query: str | None = None,
                 relevance: str | None = "relevant", analytics_only: bool = False,
                 min_engagement: int = 0, sort: str = "newest",
                 limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """Content explorer. Unlike the metrics, this can show excluded posts and why."""
    clauses = ["c.ticker = :ticker", "p.created_at >= :start", "p.created_at < :end"]
    params: dict[str, Any] = {"ticker": f.ticker, "start": f.start, "end": f.end,
                              "limit": limit, "offset": offset}
    if f.platform:
        clauses.append("p.platform = :platform"); params["platform"] = f.platform
    if f.sentiment:
        clauses.append("a.sentiment = :sentiment"); params["sentiment"] = f.sentiment
    if f.topic:
        clauses.append("a.topic = :topic"); params["topic"] = f.topic
    if f.opinions_only:
        clauses.append("a.post_type = 'opinion'")
    if relevance:
        clauses.append("a.relevance_label = :relevance"); params["relevance"] = relevance
    if analytics_only:
        clauses.append("a.quality_flag = 'ok' AND a.duplicate_of IS NULL AND p.language = 'en' "
                       "AND a.relevance_label = 'relevant'")
    if min_engagement:
        clauses.append("(coalesce(p.likes, 0) + coalesce(p.comments, 0) + coalesce(p.shares, 0)) >= :min_eng")
        params["min_eng"] = min_engagement
    if query and query.strip():
        clauses.append("p.tsv @@ websearch_to_tsquery('english', :query)")
        params["query"] = query.strip()
    clause = " AND ".join(clauses)
    base = f"""
        FROM posts p
        JOIN companies c ON c.id = p.company_id
        JOIN post_analysis a ON a.post_id = p.id
        WHERE {clause}
    """
    total = conn.execute(text(f"SELECT count(*) {base}"), params).scalar_one()
    rows = conn.execute(text(f"""
        SELECT {_POST_COLUMNS} {base}
        ORDER BY {_SORTS.get(sort, _SORTS['newest'])} LIMIT :limit OFFSET :offset
    """), params)
    return {"total": total, "posts": [_post(r, f.ticker) for r in rows]}


def posts_by_ids(conn: Connection, ticker: str, ids: list[int]) -> list[dict[str, Any]]:
    if not ids:
        return []
    rows = conn.execute(text(f"""
        SELECT {_POST_COLUMNS}
        FROM posts p JOIN companies c ON c.id = p.company_id
        JOIN post_analysis a ON a.post_id = p.id
        WHERE c.ticker = :ticker AND p.id = ANY(:ids)
    """), {"ticker": ticker, "ids": ids})
    by_id = {r.id: _post(r, ticker) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def representative_posts(conn: Connection, f: Filters, limit: int = 3) -> list[dict[str, Any]]:
    """Posts that best illustrate a slice: most engaged first, then most confident."""
    clause, params = where(f)
    ids = conn.execute(text(f"""
        SELECT id FROM analysis_posts WHERE {clause}
        ORDER BY engagement_weight DESC, (post_type = 'opinion') DESC NULLS LAST,
                 sentiment_confidence DESC NULLS LAST, created_at DESC
        LIMIT :limit
    """), {**params, "limit": limit}).scalars().all()
    return posts_by_ids(conn, f.ticker, list(ids))
