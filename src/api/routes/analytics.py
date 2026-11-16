"""Analytics routes. Every value is a SQL aggregate over stored posts."""
from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Connection

from src.analytics import briefing, insights, metrics
from src.analytics.metrics import Filters
from src.api.deps import filters, get_conn

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/{ticker}/overview")
def overview(f: Filters = Depends(filters), conn: Connection = Depends(get_conn)):
    return {
        "scope": {"ticker": f.ticker, "start": f.start, "end": f.end, "platform": f.platform},
        "overview": metrics.overview(conn, f),
        "by_platform": metrics.by_platform(conn, replace(f, platform=None)),
        "trend": metrics.sentiment_trend(conn, f),
        "coverage": metrics.coverage(conn, f.ticker, f.start, f.end),
        "freshness": metrics.freshness(conn, f.ticker),
        "emerging_concerns": insights.emerging_concerns(conn, f),
        "post_types": metrics.post_types(conn, f),
    }


@router.get("/{ticker}/briefing")
def executive_briefing(f: Filters = Depends(filters), conn: Connection = Depends(get_conn)):
    """Plain-language findings built by fixed rules from the SQL metrics (no LLM)."""
    return briefing.build(conn, f)


@router.get("/{ticker}/aspects")
def aspect_sentiment(f: Filters = Depends(filters), conn: Connection = Depends(get_conn)):
    """Sentiment per business aspect (debt, growth, ...); one post can touch several aspects."""
    return {"aspects": metrics.aspects(conn, f), "overview": metrics.overview(conn, f)}


@router.get("/{ticker}/topics")
def topics(f: Filters = Depends(filters), conn: Connection = Depends(get_conn)):
    return {"topics": metrics.topics(conn, f), "overview": metrics.overview(conn, f)}


@router.get("/{ticker}/topics/{topic_id}/posts")
def topic_posts(topic_id: str, limit: int = Query(5, ge=1, le=20),
                f: Filters = Depends(filters), conn: Connection = Depends(get_conn)):
    """Representative posts for one topic (optionally one sentiment)."""
    return metrics.representative_posts(conn, replace(f, topic=topic_id), limit)
