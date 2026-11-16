"""One HTTP client for every adapter: timeouts, retries, backoff, rate-limit handling."""
from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

log = logging.getLogger("ingestion.http")
# httpx logs full request URLs at INFO, which would print API keys sent as query parameters.
logging.getLogger("httpx").setLevel(logging.WARNING)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class HttpError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class HttpClient:
    """Counts successful and failed requests so runs can report a success rate."""

    def __init__(self, *, timeout: float = 20, max_retries: int = 4,
                 backoff_base: float = 1.0, backoff_max: float = 30,
                 transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self._client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._sleep = sleep
        self.requests_ok = 0
        self.requests_failed = 0

    def _delay(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after and retry_after.isdigit():
                return min(float(retry_after), self.backoff_max)
        # Exponential backoff with jitter: 1s, 2s, 4s, ... capped.
        return min(self.backoff_base * 2 ** attempt, self.backoff_max) * (0.5 + random.random() / 2)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        last_error = "no attempt made"
        status: int | None = None
        for attempt in range(self.max_retries + 1):
            response: httpx.Response | None = None
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as exc:  # timeouts, connection errors
                last_error = f"{type(exc).__name__}: {exc}"
                status = None
            else:
                if response.status_code < 400:
                    self.requests_ok += 1
                    return response
                status = response.status_code
                last_error = f"HTTP {status}: {response.text[:200]}"
                if status not in RETRYABLE_STATUS:
                    break  # 401, 403, 404: retrying will not help
            if attempt < self.max_retries:
                delay = self._delay(attempt, response)
                log.warning("retrying %s %s in %.1fs (%s)", method, url.split("?")[0], delay, last_error)
                self._sleep(delay)
        self.requests_failed += 1
        raise HttpError(f"{method} {url.split('?')[0]} failed: {last_error}", status)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.request("GET", url, **kwargs).json()

    def post_json(self, url: str, **kwargs: Any) -> Any:
        return self.request("POST", url, **kwargs).json()

    def close(self) -> None:
        self._client.close()
