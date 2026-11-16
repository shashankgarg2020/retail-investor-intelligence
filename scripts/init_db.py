"""Create the schema and seed the companies. Safe to re-run.

    python -m scripts.init_db
"""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import text

from src.database.connection import get_engine
from src.database.repositories import seed_companies

MIGRATIONS = Path(__file__).resolve().parent.parent / "src" / "database" / "migrations"


def main() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for path in sorted(MIGRATIONS.glob("*.sql")):
            print(f"Applying {path.name}")
            # Raw driver call with no parameters: psycopg then accepts a multi-statement script.
            conn.connection.driver_connection.execute(path.read_text(encoding="utf-8"))
        seed_companies(conn)
        tables = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )).scalars().all()
        tickers = conn.execute(text("SELECT ticker FROM companies ORDER BY ticker")).scalars().all()
    print("Tables:", ", ".join(tables))
    print("Companies:", ", ".join(tickers))


if __name__ == "__main__":
    main()
