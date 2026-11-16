"""Adapter contract. The orchestrator only ever talks to this interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from config.settings import Settings
from src.database.models import CanonicalPost
from src.ingestion.http import HttpClient


@dataclass
class Page:
    """One page of raw platform records plus whatever is needed to get the next one."""
    items: list[dict[str, Any]] = field(default_factory=list)
    next_cursor: str | None = None


@dataclass
class Health:
    ok: bool
    detail: str


class BaseAdapter(ABC):
    platform: str

    def __init__(self, settings: Settings, config: dict[str, Any], http: HttpClient):
        self.settings = settings
        self.config = config
        self.http = http

    @abstractmethod
    def is_configured(self) -> bool:
        """True when the credentials this platform needs are present."""

    @abstractmethod
    def health_check(self) -> Health:
        """Cheap call proving the credentials and endpoint work."""

    @abstractmethod
    def fetch_posts(self, query: str, cursor: str | None, since: datetime | None) -> Page:
        """Fetch one page of raw records for a search query."""

    def get_next_page(self, page: Page) -> str | None:
        """Cursor for the following page, or None when there are no more."""
        return page.next_cursor

    @abstractmethod
    def normalise(self, raw: dict[str, Any], company_ticker: str) -> CanonicalPost | None:
        """Convert one raw record to the canonical model. None means skip (deleted, empty)."""
