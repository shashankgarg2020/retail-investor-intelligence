"""Thin HTTP client for the FastAPI backend. The UI never touches the database."""
from __future__ import annotations

import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


class ApiError(Exception):
    pass


def _healthy(timeout: float = 2) -> bool:
    try:
        return httpx.get(f"{API_BASE_URL}/health", timeout=timeout).status_code == 200
    except httpx.HTTPError:
        return False


@st.cache_resource(show_spinner="Starting the analytics service…")
def ensure_backend() -> str:
    """Use a running API if there is one; otherwise start it inside this process.

    Separate processes are the normal setup (and what Docker Compose does). Single-process
    hosting such as Streamlit Community Cloud runs only `streamlit run`, so the API is started
    here on a background thread, once per server process, and still reached over HTTP: the
    dashboard code is the same in both setups.
    """
    if _healthy():
        return "external"
    url = urlparse(API_BASE_URL)
    if url.hostname not in ("127.0.0.1", "localhost"):
        return "unreachable"  # a remote API cannot be started from here

    import uvicorn

    from src.api.main import app
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=url.port or 8000, log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="embedded-api").start()
    for _ in range(120):
        if _healthy(timeout=1):
            return "embedded"
        time.sleep(0.5)
    return "unreachable"


def _request(method: str, path: str, timeout: float = 30, **kwargs: Any) -> Any:
    try:
        response = httpx.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **kwargs)
    except httpx.TransportError as exc:
        raise ApiError(f"Cannot reach the API at {API_BASE_URL}. Is it running? ({type(exc).__name__})") from exc
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"API error {response.status_code}: {detail}")
    return response.json()


def _clean(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v not in (None, "", "all")}


@st.cache_data(ttl=60, show_spinner=False)
def get(path: str, **params: Any) -> Any:
    return _request("GET", path, params=_clean(params))


def chat(question: str, ticker: str, start: str, end: str, platform: str | None) -> dict[str, Any]:
    body = _clean({"question": question, "ticker": ticker, "start": start, "end": end, "platform": platform})
    return _request("POST", "/chat", timeout=180, json=body)


def refresh_status() -> dict[str, Any]:
    """Not cached: the panel polls this while a refresh runs."""
    return _request("GET", "/refresh/status", timeout=15)


def start_refresh() -> dict[str, Any]:
    # The token (if any) lives in the dashboard's server environment, never in the browser.
    token = os.environ.get("REFRESH_TOKEN")
    return _request("POST", "/refresh", timeout=30, headers={"X-Refresh-Token": token} if token else None)
