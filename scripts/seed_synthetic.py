"""Load clearly labelled SYNTHETIC posts for a platform that has no API access.

    python -m scripts.seed_synthetic --platform reddit
    python -m scripts.seed_synthetic --platform reddit --remove

Use only for a platform whose real adapter cannot be authorised in time. Every
record is stored with is_synthetic = true, gets a 'synthetic://' URL instead of a
source link, and is badged in every screen. Remove them once real data exists.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import text

from config.settings import PLATFORMS
from src.database import repositories as repo
from src.database.connection import get_engine
from src.database.models import CanonicalPost
from src.processing.normalizer import text_hash

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "sample" / "synthetic_posts.json"


def remove(platform: str) -> None:
    with get_engine().begin() as conn:
        deleted = conn.execute(text("DELETE FROM posts WHERE is_synthetic AND platform = :p"),
                               {"p": platform}).rowcount
        conn.execute(text("DELETE FROM ingestion_runs WHERE is_synthetic AND platform = :p"), {"p": platform})
    print(f"Removed {deleted} synthetic {platform} posts.")


def seed(platform: str) -> None:
    records = [r for r in json.loads(FIXTURE.read_text(encoding="utf-8"))["posts"] if r["platform"] == platform]
    now = datetime.now(timezone.utc)
    inserted = 0
    with get_engine().begin() as conn:
        real = conn.execute(text("SELECT count(*) FROM posts WHERE platform = :p AND NOT is_synthetic"),
                            {"p": platform}).scalar_one()
        if real:
            raise SystemExit(f"{platform} already has {real} real posts; not mixing synthetic data in.")
        for ticker in sorted({r["ticker"] for r in records}):
            cid = repo.company_id(conn, ticker)
            run_id = repo.start_run(conn, platform, cid, ["synthetic fixture"], is_synthetic=True)
            count = 0
            for index, r in enumerate(x for x in records if x["ticker"] == ticker):
                external_id = f"synthetic-{platform}-{ticker}-{index}"
                post = CanonicalPost(
                    platform=platform, external_id=external_id, company_ticker=ticker,
                    content_type=r["type"], title=r.get("title"), text=r["text"],
                    text_hash=text_hash(f"{r.get('title', '')} {r['text']}"),
                    author_id=f"synthetic-author-{index}", parent_title=r.get("parent"),
                    created_at=now - timedelta(days=r["days_ago"], hours=index),
                    url=f"synthetic://{external_id}", likes=r.get("likes"), comments=r.get("comments"),
                    views=r.get("views"), language="en", is_synthetic=True,
                )
                count += repo.upsert_post(conn, post, cid, run_id)
            repo.finish_run(conn, run_id, status="success", fetched=count, inserted=count, duplicates=0,
                            requests_ok=0, requests_failed=0,
                            error="SYNTHETIC fixture data, not collected from the platform", checkpoint=None)
            inserted += count
    print(f"Inserted {inserted} SYNTHETIC {platform} posts. Run: python -m scripts.run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", required=True, choices=PLATFORMS)
    parser.add_argument("--remove", action="store_true")
    args = parser.parse_args()
    remove(args.platform) if args.remove else seed(args.platform)


if __name__ == "__main__":
    main()
