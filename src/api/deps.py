"""Shared request helpers: database connection, date range parsing, company validation."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterator

from fastapi import HTTPException, Query
from sqlalchemy import Connection

from config.settings import PLATFORMS, companies
from src.analytics.metrics import Filters
from src.database.connection import get_engine

DEFAULT_DAYS = 30


def get_conn() -> Iterator[Connection]:
    with get_engine().connect() as conn:
        yield conn


def valid_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker not in companies():
        raise HTTPException(404, f"Unknown company '{ticker}'. Known: {', '.join(companies())}")
    return ticker


def date_range(start: date | None, end: date | None) -> tuple[datetime, datetime]:
    """Inclusive calendar dates from the UI -> half-open UTC interval [start, end + 1 day)."""
    end = end or datetime.now(timezone.utc).date()
    start = start or end - timedelta(days=DEFAULT_DAYS - 1)
    if start > end:
        raise HTTPException(422, "start must not be after end")
    return (datetime.combine(start, time.min, timezone.utc),
            datetime.combine(end + timedelta(days=1), time.min, timezone.utc))


def filters(
    ticker: str,
    start: date | None = Query(None, description="First day, inclusive (YYYY-MM-DD)"),
    end: date | None = Query(None, description="Last day, inclusive (YYYY-MM-DD)"),
    platform: str | None = Query(None),
    sentiment: str | None = Query(None, pattern="^(positive|neutral|negative)$"),
    topic: str | None = Query(None),
    opinions_only: bool = Query(False, description="Only posts classified as the author's own opinion"),
) -> Filters:
    if platform and platform not in PLATFORMS:
        raise HTTPException(422, f"platform must be one of {', '.join(PLATFORMS)}")
    s, e = date_range(start, end)
    return Filters(valid_ticker(ticker), s, e, platform, sentiment, topic, opinions_only)
