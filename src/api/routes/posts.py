from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Connection

from src.analytics import metrics
from src.analytics.metrics import Filters
from src.api.deps import filters, get_conn

router = APIRouter(prefix="/posts", tags=["posts"])


@router.get("/{ticker}")
def search(
    f: Filters = Depends(filters),
    q: str | None = Query(None, max_length=200, description="Keyword search"),
    relevance: str | None = Query("relevant", pattern="^(relevant|possibly_relevant|irrelevant|all)$"),
    analytics_only: bool = Query(False, description="Only posts counted in the dashboard metrics"),
    min_engagement: int = Query(0, ge=0),
    sort: str = Query("newest", pattern="^(newest|engagement|relevance)$"),
    limit: int = Query(25, ge=1, le=100),
    offset: int = Query(0, ge=0),
    conn: Connection = Depends(get_conn),
):
    """Content explorer. Can show excluded posts, with the reason they were excluded."""
    return metrics.search_posts(
        conn, f, query=q, relevance=None if relevance == "all" else relevance,
        analytics_only=analytics_only, min_engagement=min_engagement,
        sort=sort, limit=limit, offset=offset,
    )
