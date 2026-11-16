"""Periodic collection and processing (docs/blueprint.md §4.1: "APScheduler initially").

    python -m scripts.scheduler                 # every 6 hours, first run immediately
    python -m scripts.scheduler --hours 12
    python -m scripts.scheduler --once          # one cycle, then exit (for cron or a CI job)

One cycle = collect new posts for every company and platform (incremental, from each
platform's checkpoint), then process anything not yet analysed. A failure in one company
or platform is logged and the rest of the cycle continues. Runs as its own process, so
the API that serves users never does ingestion or model work.

At larger scale this becomes Airflow/Prefect plus queue workers (see README, Scaling).
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from config.settings import PLATFORMS, companies

log = logging.getLogger("scheduler")


def cycle(use_llm: bool = True) -> dict[str, dict]:
    """Collect then process every company. Returns a per-company summary."""
    from scripts.run_pipeline import process_company
    from src.ingestion.orchestrator import collect

    started = time.perf_counter()
    summary: dict[str, dict] = {}
    for ticker in companies():
        entry: dict = {}
        try:
            runs = collect(ticker, tuple(PLATFORMS), None)
            entry["collected"] = {r.platform: f"{r.status} (+{r.inserted})" for r in runs}
        except Exception as exc:  # one company failing must not stop the others
            log.exception("collection failed for %s", ticker)
            entry["collected"] = f"failed: {type(exc).__name__}"
        try:
            stats = process_company(ticker, reprocess=False, use_llm=use_llm)
            entry["processed"] = stats.get("posts", 0)
        except Exception as exc:
            log.exception("processing failed for %s", ticker)
            entry["processed"] = f"failed: {type(exc).__name__}"
        summary[ticker] = entry
        log.info("%s: %s", ticker, entry)
    log.info("cycle finished in %.0fs", time.perf_counter() - started)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hours", type=float, default=6.0, help="Interval between cycles")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    parser.add_argument("--no-llm", action="store_true", help="Rules and FinBERT only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.once:
        cycle(use_llm=not args.no_llm)
        return

    from apscheduler.schedulers.blocking import BlockingScheduler
    scheduler = BlockingScheduler(timezone="UTC")
    # max_instances=1 and coalesce: a slow cycle is never run twice in parallel, and missed
    # runs (laptop asleep) collapse into one instead of firing in a burst.
    scheduler.add_job(cycle, "interval", hours=args.hours, kwargs={"use_llm": not args.no_llm},
                      next_run_time=datetime.now(timezone.utc), max_instances=1, coalesce=True,
                      id="collect_and_process")
    log.info("scheduler started: every %.1f hours (Ctrl+C to stop)", args.hours)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopped")


if __name__ == "__main__":
    main()
