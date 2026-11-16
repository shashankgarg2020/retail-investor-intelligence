"""Database engine. One place that knows how to connect."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from config.settings import get_settings


def _normalise_url(url: str) -> str:
    """Neon gives postgresql://...; SQLAlchemy needs the driver named."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


@lru_cache
def get_engine() -> Engine:
    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Add your Neon connection string to .env.")
    # pool_pre_ping: Neon suspends idle databases, so stale connections are expected.
    return create_engine(_normalise_url(url), pool_pre_ping=True, pool_size=5, max_overflow=5)


def vector_literal(values) -> str:
    """Render a vector for a parameterised CAST(:v AS vector)."""
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"
