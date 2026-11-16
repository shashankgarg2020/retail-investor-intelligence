"""Schemas for the assistant: what the LLM must return, and what the API returns."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Theme(BaseModel):
    title: str = Field(description="Short name for the theme, under ten words.")
    summary: str = Field(description="What people are saying, in one to three sentences, as reported opinion.")
    stance: Literal["positive", "negative", "mixed", "neutral"]
    evidence_ids: list[int] = Field(description="Ids of supplied posts that express this theme.")


class AssistantDraft(BaseModel):
    """What the LLM produces. Validated by the SDK, then checked again by the engine."""
    answer: str = Field(description="Direct answer to the question in two to four sentences.")
    themes: list[Theme]
    limitations: list[str] = Field(description="Specific caveats about this answer's evidence.")
    insufficient_evidence: bool


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    ticker: str
    start: str | None = None      # ISO date; defaults applied by the API
    end: str | None = None
    platform: str | None = None


class ChatResponse(BaseModel):
    answer: str
    themes: list[Theme]
    evidence: list[dict[str, Any]]            # the cited posts, with source links
    insufficient_evidence: bool
    intent: str
    scope: dict[str, Any]                     # company, dates, platform, sentiment filter
    posts_analysed: int
    platform_breakdown: list[dict[str, Any]]
    metrics: dict[str, Any]                   # the SQL results the answer was given
    limitations: list[str]
    generated_by: str                         # model id, or the fallback name
    retrieved_count: int
    dropped_citations: int                    # ids the model cited that were not in the context
    planner_notes: list[str]
    timings_seconds: dict[str, float]
    usage: dict[str, Any] | None = None
