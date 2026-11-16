"""All writes and pipeline reads. Plain parameterised SQL, no ORM."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy import Connection, text

from config.settings import companies
from src.database.connection import vector_literal
from src.database.models import CanonicalPost

# ---------------------------------------------------------------- companies

def seed_companies(conn: Connection) -> None:
    for ticker, cfg in companies().items():
        conn.execute(
            text("""
                INSERT INTO companies (ticker, name, exchange, industry, aliases)
                VALUES (:ticker, :name, :exchange, :industry, CAST(:aliases AS jsonb))
                ON CONFLICT (ticker) DO UPDATE
                SET name = EXCLUDED.name, exchange = EXCLUDED.exchange,
                    industry = EXCLUDED.industry, aliases = EXCLUDED.aliases
            """),
            {"ticker": ticker, "name": cfg["name"], "exchange": cfg["exchange"],
             "industry": cfg["industry"], "aliases": json.dumps(cfg["aliases"])},
        )


def company_id(conn: Connection, ticker: str) -> int:
    row = conn.execute(text("SELECT id FROM companies WHERE ticker = :t"), {"t": ticker}).first()
    if row is None:
        raise ValueError(f"Unknown company ticker: {ticker}")
    return row[0]


# ----------------------------------------------------------- ingestion runs

def start_run(conn: Connection, platform: str, cid: int, queries: list[str], is_synthetic: bool) -> int:
    return conn.execute(
        text("""
            INSERT INTO ingestion_runs (platform, company_id, status, queries, is_synthetic)
            VALUES (:p, :c, 'running', CAST(:q AS jsonb), :s) RETURNING id
        """),
        {"p": platform, "c": cid, "q": json.dumps(queries), "s": is_synthetic},
    ).scalar_one()


def finish_run(conn: Connection, run_id: int, *, status: str, fetched: int, inserted: int,
               duplicates: int, requests_ok: int, requests_failed: int,
               error: str | None, checkpoint: datetime | None) -> None:
    conn.execute(
        text("""
            UPDATE ingestion_runs
            SET status = :status, fetched = :fetched, inserted = :inserted, duplicates = :duplicates,
                requests_ok = :ok, requests_failed = :failed, error = :error,
                checkpoint = :checkpoint, completed_at = now()
            WHERE id = :id
        """),
        {"id": run_id, "status": status, "fetched": fetched, "inserted": inserted,
         "duplicates": duplicates, "ok": requests_ok, "failed": requests_failed,
         "error": error, "checkpoint": checkpoint},
    )


def last_checkpoint(conn: Connection, platform: str, cid: int) -> datetime | None:
    """Newest content timestamp from the last fully successful real run.

    A partial run is not used: some of its queries failed, so their window was
    never read. The next run re-reads the full window and the upsert absorbs repeats.
    """
    return conn.execute(
        text("""
            SELECT max(checkpoint) FROM ingestion_runs
            WHERE platform = :p AND company_id = :c AND is_synthetic = FALSE
              AND status = 'success'
        """),
        {"p": platform, "c": cid},
    ).scalar()


# -------------------------------------------------------------------- posts

def upsert_post(conn: Connection, post: CanonicalPost, cid: int, run_id: int | None) -> bool:
    """Insert, or refresh engagement if already stored. Returns True when new."""
    inserted = conn.execute(
        text("""
            INSERT INTO posts (company_id, platform, external_id, content_type, title, text, text_hash,
                               author_id, parent_external_id, parent_title, created_at, url,
                               likes, comments, shares, views, language, is_synthetic, extra,
                               ingestion_run_id)
            VALUES (:company_id, :platform, :external_id, :content_type, :title, :text, :text_hash,
                    :author_id, :parent_external_id, :parent_title, :created_at, :url,
                    :likes, :comments, :shares, :views, :language, :is_synthetic,
                    CAST(:extra AS jsonb), :run_id)
            ON CONFLICT (company_id, platform, external_id) DO UPDATE
            SET likes = EXCLUDED.likes, comments = EXCLUDED.comments,
                shares = EXCLUDED.shares, views = EXCLUDED.views
            RETURNING (xmax = 0) AS inserted
        """),
        {**post.model_dump(exclude={"company_ticker", "extra"}),
         "company_id": cid, "extra": json.dumps(post.extra), "run_id": run_id},
    ).scalar_one()
    return bool(inserted)


def unprocessed_posts(conn: Connection, ticker: str | None, reprocess: bool) -> list[dict[str, Any]]:
    where = [] if reprocess else ["a.post_id IS NULL"]
    params: dict[str, Any] = {}
    if ticker:
        where.append("c.ticker = :ticker")
        params["ticker"] = ticker
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    rows = conn.execute(
        text(f"""
            SELECT p.id, c.ticker, p.platform, p.content_type, p.title, p.text, p.text_hash,
                   p.author_id, p.parent_title, p.created_at, p.language, p.extra
            FROM posts p
            JOIN companies c ON c.id = p.company_id
            LEFT JOIN post_analysis a ON a.post_id = p.id
            {clause}
            ORDER BY p.created_at, p.id
        """),
        params,
    )
    return [dict(r._mapping) for r in rows]


def earlier_hashes(conn: Connection, ticker: str) -> dict[str, int]:
    """text_hash -> id of the earliest already-analysed, non-duplicate post."""
    rows = conn.execute(
        text("""
            SELECT DISTINCT ON (p.text_hash) p.text_hash, p.id
            FROM posts p
            JOIN companies c ON c.id = p.company_id
            JOIN post_analysis a ON a.post_id = p.id
            WHERE c.ticker = :t AND a.duplicate_of IS NULL
            ORDER BY p.text_hash, p.created_at, p.id
        """),
        {"t": ticker},
    )
    return {r[0]: r[1] for r in rows}


def set_language(conn: Connection, post_id: int, language: str) -> None:
    conn.execute(text("UPDATE posts SET language = :l WHERE id = :id"), {"l": language, "id": post_id})


ANALYSIS_DEFAULTS = {"finbert_sentiment": None, "finbert_confidence": None,
                     "post_type": None, "sentiment_reason": None}


def save_analysis(conn: Connection, rows: Iterable[dict[str, Any]]) -> None:
    stmt = text("""
        INSERT INTO post_analysis (post_id, relevance_label, relevance_score, relevance_method,
                                   relevance_reason, quality_flag, duplicate_of, sentiment,
                                   sentiment_confidence, sentiment_model, sentiment_reason,
                                   finbert_sentiment, finbert_confidence, post_type,
                                   topic, topic_score, topic_method, processed_at)
        VALUES (:post_id, :relevance_label, :relevance_score, :relevance_method,
                :relevance_reason, :quality_flag, :duplicate_of, :sentiment,
                :sentiment_confidence, :sentiment_model, :sentiment_reason,
                :finbert_sentiment, :finbert_confidence, :post_type,
                :topic, :topic_score, :topic_method, now())
        ON CONFLICT (post_id) DO UPDATE
        SET relevance_label = EXCLUDED.relevance_label, relevance_score = EXCLUDED.relevance_score,
            relevance_method = EXCLUDED.relevance_method, relevance_reason = EXCLUDED.relevance_reason,
            quality_flag = EXCLUDED.quality_flag, duplicate_of = EXCLUDED.duplicate_of,
            sentiment = EXCLUDED.sentiment, sentiment_confidence = EXCLUDED.sentiment_confidence,
            sentiment_model = EXCLUDED.sentiment_model, sentiment_reason = EXCLUDED.sentiment_reason,
            finbert_sentiment = EXCLUDED.finbert_sentiment, finbert_confidence = EXCLUDED.finbert_confidence,
            post_type = EXCLUDED.post_type, topic = EXCLUDED.topic,
            topic_score = EXCLUDED.topic_score, topic_method = EXCLUDED.topic_method,
            processed_at = now()
    """)
    for row in rows:
        conn.execute(stmt, {**ANALYSIS_DEFAULTS, **row})


def save_aspects(conn: Connection, post_id: int, aspects: list[tuple[str, str]], model: str) -> None:
    """Replace a post's aspect sentiment with the latest analysis."""
    conn.execute(text("DELETE FROM aspect_sentiment WHERE post_id = :id"), {"id": post_id})
    for aspect, sentiment in aspects:
        conn.execute(text("""
            INSERT INTO aspect_sentiment (post_id, aspect, sentiment, model)
            VALUES (:id, :aspect, :sentiment, :model)
        """), {"id": post_id, "aspect": aspect, "sentiment": sentiment, "model": model})


def save_embedding(conn: Connection, post_id: int, model: str, vector) -> None:
    conn.execute(
        text("""
            INSERT INTO embeddings (post_id, model, embedding)
            VALUES (:id, :m, CAST(:v AS vector))
            ON CONFLICT (post_id) DO UPDATE SET model = EXCLUDED.model, embedding = EXCLUDED.embedding
        """),
        {"id": post_id, "m": model, "v": vector_literal(vector)},
    )
