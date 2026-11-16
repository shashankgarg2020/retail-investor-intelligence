"""Canonical data model shared by every platform adapter (docs/blueprint.md §6.1)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Platform = Literal["reddit", "youtube", "bluesky"]
ContentType = Literal["post", "comment", "video"]


class CanonicalPost(BaseModel):
    """One piece of public content, in the same shape regardless of source."""

    platform: Platform
    external_id: str = Field(min_length=1)
    company_ticker: str
    content_type: ContentType
    title: str | None = None
    text: str = Field(min_length=1)
    text_hash: str
    author_id: str | None = None
    parent_external_id: str | None = None
    parent_title: str | None = None
    created_at: datetime
    url: str = Field(min_length=1)
    # None means the platform did not provide the metric. Never default to 0.
    likes: int | None = None
    comments: int | None = None
    shares: int | None = None
    views: int | None = None
    language: str | None = None
    is_synthetic: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("created_at must be timezone-aware (UTC)")
        return v
