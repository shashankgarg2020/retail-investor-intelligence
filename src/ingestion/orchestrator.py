"""Runs collection for a company across platforms.

Each platform is isolated: a failure in one is recorded and the others carry on.
Every run is written to ingestion_runs so the dashboard can show freshness and failures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pydantic import ValidationError

from config.settings import PLATFORMS, companies, get_settings, platform_config
from src.database import repositories as repo
from src.database.connection import get_engine
from src.ingestion.base import BaseAdapter
from src.ingestion.bluesky import BlueskyAdapter
from src.ingestion.http import HttpClient, HttpError
from src.ingestion.reddit import RedditAdapter
from src.ingestion.youtube import YouTubeAdapter

log = logging.getLogger("ingestion")

ADAPTERS: dict[str, type[BaseAdapter]] = {
    "reddit": RedditAdapter, "youtube": YouTubeAdapter, "bluesky": BlueskyAdapter,
}


@dataclass
class RunResult:
    platform: str
    ticker: str
    status: str
    fetched: int = 0
    inserted: int = 0
    duplicates: int = 0
    invalid: int = 0
    requests_ok: int = 0
    requests_failed: int = 0
    error: str | None = None


def build_adapter(platform: str) -> BaseAdapter:
    cfg = platform_config()[platform]
    http = HttpClient(timeout=cfg["timeout_seconds"], max_retries=cfg["max_retries"],
                      backoff_base=cfg["backoff_base_seconds"], backoff_max=cfg["backoff_max_seconds"])
    return ADAPTERS[platform](get_settings(), cfg, http)


def collect_platform(adapter: BaseAdapter, ticker: str, limit: int | None = None) -> RunResult:
    engine = get_engine()
    cfg = adapter.config
    queries: list[str] = companies()[ticker]["search_queries"]
    result = RunResult(adapter.platform, ticker, "running")

    with engine.begin() as conn:
        cid = repo.company_id(conn, ticker)
        checkpoint = repo.last_checkpoint(conn, adapter.platform, cid)
        run_id = repo.start_run(conn, adapter.platform, cid, queries, is_synthetic=False)

    now = datetime.now(timezone.utc)
    # First run: fixed historical window. Later runs: from the checkpoint, minus an
    # overlap to catch late-arriving records. Re-read rows are absorbed by the upsert.
    since = (checkpoint - timedelta(hours=cfg["overlap_hours"]) if checkpoint
             else now - timedelta(days=companies()[ticker].get("lookback_days", cfg["lookback_days"])))
    newest = checkpoint
    errors: list[str] = []

    if not adapter.is_configured():
        errors.append("credentials not configured")
    else:
        for query in queries:
            cursor = None
            for _ in range(cfg["max_pages_per_query"]):
                if limit is not None and result.fetched >= limit:
                    break
                try:
                    page = adapter.fetch_posts(query, cursor, since)
                except (HttpError, KeyError, ValueError) as exc:
                    errors.append(f"{query!r}: {exc}")
                    log.error("%s %s query %r failed: %s", adapter.platform, ticker, query, exc)
                    break  # give up on this query, keep the others
                in_window = 0
                # Persist page by page so a later failure keeps what was already collected.
                with engine.begin() as conn:
                    for raw in page.items:
                        try:
                            post = adapter.normalise(raw, ticker)
                        except (ValidationError, KeyError, ValueError, TypeError) as exc:
                            result.invalid += 1
                            log.warning("%s: record failed normalisation: %s", adapter.platform, exc)
                            continue
                        if post is None or post.created_at < since:
                            continue
                        in_window += 1
                        result.fetched += 1
                        if repo.upsert_post(conn, post, cid, run_id):
                            result.inserted += 1
                        else:
                            result.duplicates += 1
                        if newest is None or post.created_at > newest:
                            newest = post.created_at
                cursor = adapter.get_next_page(page)
                # Results are newest-first: a page with nothing in the window means we are done.
                if cursor is None or in_window == 0:
                    break

    result.requests_ok = adapter.http.requests_ok
    result.requests_failed = adapter.http.requests_failed
    if not errors:
        result.status = "success"
    elif not adapter.is_configured():
        result.status = "skipped"
    else:
        result.status = "partial" if result.fetched else "failed"
    result.error = "; ".join(errors)[:1000] or None

    with engine.begin() as conn:
        repo.finish_run(conn, run_id, status=result.status, fetched=result.fetched,
                        inserted=result.inserted, duplicates=result.duplicates,
                        requests_ok=result.requests_ok, requests_failed=result.requests_failed,
                        error=result.error, checkpoint=newest)
    return result


def collect(ticker: str, platforms: tuple[str, ...] = PLATFORMS, limit: int | None = None) -> list[RunResult]:
    results = []
    for platform in platforms:
        adapter = build_adapter(platform)
        try:
            results.append(collect_platform(adapter, ticker, limit))
        except Exception as exc:  # last line of defence: never let one source stop the rest
            log.exception("%s %s crashed", platform, ticker)
            results.append(RunResult(platform, ticker, "failed", error=f"{type(exc).__name__}: {exc}"))
        finally:
            adapter.http.close()
    return results
