"""FastAPI application.

    uvicorn src.api.main:app --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from config.settings import ROOT, get_settings
from src.api.routes import analytics, chat, companies, posts, refresh
from src.database.connection import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")


def _warm_embedding_model() -> None:
    # Loading the query-embedding model takes ~20 s on CPU. Doing it in the background at
    # startup means the first assistant question does not pay for it; the API serves
    # dashboard requests meanwhile because this runs on its own thread.
    try:
        from src.intelligence.embeddings import embed_query
        embed_query("warm up")
        log.info("embedding model loaded")
    except Exception:
        log.exception("embedding model warm-up failed; it will load on first use")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.environ.get("WARM_MODELS", "1") != "0":
        threading.Thread(target=_warm_embedding_model, daemon=True, name="warm-embeddings").start()
    yield


app = FastAPI(
    title="Retail Investor Intelligence API",
    description="Evidence-backed view of public retail-investor discussion for IR teams.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(companies.router)
app.include_router(analytics.router)
app.include_router(posts.router)
app.include_router(chat.router)
app.include_router(refresh.router)


@app.get("/health", tags=["meta"])
def health():
    settings = get_settings()
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        database = "ok"
    except Exception as exc:
        database = f"error: {type(exc).__name__}"
    return {"database": database, "llm_configured": bool(settings.anthropic_api_key),
            "llm_model": settings.llm_model}


EVAL_RESULTS = ROOT / "data" / "eval" / "results.json"


@app.get("/quality", tags=["meta"])
def quality():
    """Latest measured model quality, written by `python -m scripts.evaluate all`."""
    if not EVAL_RESULTS.exists():
        return {"available": False}
    return {"available": True, **json.loads(EVAL_RESULTS.read_text(encoding="utf-8"))}
