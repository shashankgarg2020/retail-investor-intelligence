from __future__ import annotations

import time
from collections import deque
from datetime import date
from threading import Lock

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import Connection

from config.settings import PLATFORMS
from src.api.deps import date_range, get_conn, valid_ticker
from src.rag import query_engine
from src.rag.response_schema import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])

# Simple per-client limit: each question can cost an LLM call.
RATE_LIMIT, RATE_WINDOW = 20, 60.0
_hits: dict[str, deque[float]] = {}
_lock = Lock()


def rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _lock:
        hits = _hits.setdefault(client, deque())
        while hits and now - hits[0] > RATE_WINDOW:
            hits.popleft()
        if len(hits) >= RATE_LIMIT:
            raise HTTPException(429, "Too many questions; wait a minute and try again.")
        hits.append(now)


def _embed(text: str):
    # Imported on first use so the API starts quickly and analytics never load the model.
    from src.intelligence.embeddings import embed_query
    return embed_query(text)


@router.post("", response_model=ChatResponse, dependencies=[Depends(rate_limit)])
def chat(body: ChatRequest, conn: Connection = Depends(get_conn)) -> ChatResponse:
    if body.platform and body.platform not in PLATFORMS:
        raise HTTPException(422, f"platform must be one of {', '.join(PLATFORMS)}")
    try:
        start, end = date_range(date.fromisoformat(body.start) if body.start else None,
                                date.fromisoformat(body.end) if body.end else None)
    except ValueError:
        raise HTTPException(422, "start and end must be ISO dates (YYYY-MM-DD)")
    return query_engine.answer(conn, body.question, valid_ticker(body.ticker), start, end,
                               body.platform, _embed)
