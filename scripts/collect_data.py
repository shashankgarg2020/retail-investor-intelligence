"""Collect public posts for one or all companies.

    python -m scripts.collect_data --company VEON --limit 50
    python -m scripts.collect_data --company all --platforms bluesky youtube
    python -m scripts.collect_data --health
"""
from __future__ import annotations

import argparse
import logging

from config.settings import PLATFORMS, companies
from src.ingestion.orchestrator import build_adapter, collect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--company", default="all", help="Ticker or 'all'")
    parser.add_argument("--platforms", nargs="+", default=list(PLATFORMS), choices=PLATFORMS)
    parser.add_argument("--limit", type=int, default=None, help="Max records per platform per company")
    parser.add_argument("--health", action="store_true", help="Only check each platform's credentials")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.health:
        for platform in args.platforms:
            adapter = build_adapter(platform)
            if not adapter.is_configured():
                print(f"{platform:8} NOT CONFIGURED  (credentials missing in .env)")
            else:
                health = adapter.health_check()
                print(f"{platform:8} {'OK' if health.ok else 'FAILED':15} {health.detail}")
            adapter.http.close()
        return

    tickers = list(companies()) if args.company.lower() == "all" else [args.company.upper()]
    print(f"{'company':8} {'platform':8} {'status':8} {'fetched':>7} {'new':>5} {'dup':>5} "
          f"{'invalid':>7} {'req ok':>6} {'req fail':>8}  error")
    for ticker in tickers:
        for r in collect(ticker, tuple(args.platforms), args.limit):
            print(f"{r.ticker:8} {r.platform:8} {r.status:8} {r.fetched:>7} {r.inserted:>5} "
                  f"{r.duplicates:>5} {r.invalid:>7} {r.requests_ok:>6} {r.requests_failed:>8}  "
                  f"{r.error or ''}")


if __name__ == "__main__":
    main()
