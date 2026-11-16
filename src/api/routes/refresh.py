"""On-demand data refresh: collect new posts and analyse them, from the dashboard.

The work runs as a separate process (`python -m scripts.scheduler --once`), not inside
the API. The API keeps serving dashboards while it runs, never loads FinBERT, and the
models' memory is released when the run ends. One run at a time; a cooldown between
runs protects the platforms' API quotas and LLM spend.
"""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import Connection, text

from config.settings import ROOT, get_settings
from src.api.deps import get_conn

router = APIRouter(prefix="/refresh", tags=["refresh"])

LOG_FILE = Path(tempfile.gettempdir()) / "rii_refresh.log"
COMMAND = [sys.executable, "-m", "scripts.scheduler", "--once"]

_lock = threading.Lock()
_process: subprocess.Popen | None = None
_run: dict[str, Any] = {"state": "idle", "started_at": None, "finished_at": None}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _update() -> None:
    """Fold a finished process into the run record. Call with the lock held."""
    global _process
    if _process is not None and _process.poll() is not None:
        _run["state"] = "succeeded" if _process.returncode == 0 else "failed"
        _run["finished_at"] = _now()
        _process = None


def _cooldown_left() -> int:
    if _run["state"] == "running" or _run["started_at"] is None:
        return 0
    cooldown = get_settings().refresh_cooldown_min * 60
    return max(0, int(cooldown - (_now() - _run["started_at"]).total_seconds()))


def _last_log_line() -> str | None:
    try:
        lines = [ln.strip() for ln in LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()]
    except OSError:
        return None
    lines = [ln for ln in lines if ln]
    return lines[-1][-240:] if lines else None


def data_times(conn: Connection, since: datetime | None) -> dict[str, Any]:
    """When data last arrived and was analysed, and how many posts arrived since `since`."""
    row = conn.execute(text("""
        SELECT (SELECT max(collected_at) FROM posts) AS last_collected,
               (SELECT max(processed_at) FROM post_analysis) AS last_processed,
               (SELECT count(*) FROM posts WHERE CAST(:since AS timestamptz) IS NOT NULL
                                              AND collected_at >= :since) AS new_posts
    """), {"since": since}).one()
    return dict(row._mapping)


def _status(conn: Connection) -> dict[str, Any]:
    with _lock:
        _update()
        run = dict(_run)
        cooldown = _cooldown_left()
    times = data_times(conn, run["started_at"])
    return {**run, "cooldown_seconds_left": cooldown, "last_updated": times["last_processed"] or times["last_collected"],
            "new_posts": times["new_posts"] if run["started_at"] else None,
            "message": _last_log_line() if run["state"] in ("running", "failed") else None}


@router.get("/status")
def status(conn: Connection = Depends(get_conn)):
    return _status(conn)


@router.post("", status_code=202)
def start(conn: Connection = Depends(get_conn), x_refresh_token: str | None = Header(None)):
    """Start collecting and analysing new posts. Poll /refresh/status for progress."""
    global _process
    token = get_settings().refresh_token
    if token and not secrets.compare_digest(x_refresh_token or "", token):
        raise HTTPException(403, "Refreshing needs the refresh token.")
    with _lock:
        _update()
        if _run["state"] == "running":
            raise HTTPException(409, "A refresh is already running.")
        left = _cooldown_left()
        if left:
            raise HTTPException(429, f"Data was refreshed recently. Try again in {max(1, left // 60)} min.")
        log = open(LOG_FILE, "w", encoding="utf-8")
        try:
            _process = subprocess.Popen(COMMAND, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        finally:
            log.close()  # the child holds its own handle
        _run.update(state="running", started_at=_now(), finished_at=None)
    return _status(conn)
