from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import Connection, text

from config.settings import companies, topics_for
from src.analytics import metrics
from src.api.deps import get_conn, valid_ticker

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("")
def list_companies(conn: Connection = Depends(get_conn)):
    counts = {r.ticker: r for r in conn.execute(text("""
        SELECT c.ticker, count(p.id) AS collected, min(p.created_at) AS first_post,
               max(p.created_at) AS last_post
        FROM companies c LEFT JOIN posts p ON p.company_id = c.id GROUP BY c.ticker
    """))}
    return [
        {"ticker": ticker, "name": cfg["name"], "exchange": cfg["exchange"], "industry": cfg["industry"],
         "aliases": cfg["aliases"],
         "collected": counts[ticker].collected if ticker in counts else 0,
         "first_post": counts[ticker].first_post if ticker in counts else None,
         "last_post": counts[ticker].last_post if ticker in counts else None,
         "topics": {tid: t["label"] for tid, t in topics_for(ticker).items()}}
        for ticker, cfg in companies().items()
    ]


@router.get("/{ticker}/freshness")
def freshness(ticker: str, conn: Connection = Depends(get_conn)):
    """Per-platform collection status, so missing coverage is visible rather than implied."""
    return metrics.freshness(conn, valid_ticker(ticker))
