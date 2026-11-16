"""Investor-stance sentiment, post type and aspect sentiment from one LLM pass (docs/blueprint.md §8.1, 8.2).

Why this exists next to FinBERT: on the labelled evaluation set FinBERT called half of
the positive and more than half of the negative posts "neutral". It was trained on
financial-news sentences, where most lines are factual, so a headline like "Take-Two
forecasts $8 billion in bookings" reads as neutral to it. An investor reads it as good
news. The LLM is asked the investor's question directly: is this good or bad for a
shareholder?

FinBERT still runs on every post. Its label is stored as `finbert_sentiment`, so the
two can be compared against human labels (scripts/evaluate.py) and the system falls back
to it whenever the LLM is unavailable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from config.settings import companies
from src.rag.llm import LLMError, LLMUnavailable, get_llm
from src.rag.prompts import ANALYSIS_SYSTEM, analysis_user_prompt

log = logging.getLogger("pipeline.analysis")
BATCH_SIZE = 10
STANCES = ("positive", "neutral", "negative")
POST_TYPES = ("opinion", "news", "market_data", "question", "other")


class AspectView(BaseModel):
    aspect: str
    stance: Literal["positive", "neutral", "negative"]


class PostLabel(BaseModel):
    post_id: int
    stance: Literal["positive", "neutral", "negative"]
    post_type: Literal["opinion", "news", "market_data", "question", "other"]
    aspects: list[AspectView] = Field(default_factory=list)
    reason: str


class PostLabels(BaseModel):
    labels: list[PostLabel]


@dataclass
class Analysis:
    sentiment: str
    post_type: str
    reason: str
    model: str
    aspects: list[tuple[str, str]] = field(default_factory=list)   # (aspect id, stance)


def aspects_for(ticker: str) -> dict[str, str]:
    return companies()[ticker].get("aspects", {})


def accept(label: PostLabel, valid_ids: set[int], aspect_ids: set[str], model: str) -> Analysis | None:
    """Validate one model output. Unknown post ids are rejected; unknown aspects are dropped."""
    if label.post_id not in valid_ids:
        return None
    seen: dict[str, str] = {}
    for view in label.aspects:
        if view.aspect in aspect_ids and view.aspect not in seen:
            seen[view.aspect] = view.stance
    return Analysis(label.stance, label.post_type, label.reason[:200], f"llm:{model}", list(seen.items()))


def analyse(ticker: str, posts: list[dict[str, Any]], llm=None) -> tuple[dict[int, Analysis], dict[str, float]]:
    """Label posts in batches. Returns analyses by post id plus usage totals.

    Posts without a result (no API key, a failed batch, an id the model skipped) are
    simply absent; the pipeline then keeps FinBERT's label for them.
    """
    results: dict[int, Analysis] = {}
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "seconds": 0.0}
    if not posts:
        return results, usage
    if llm is None:
        try:
            llm = get_llm()
        except LLMUnavailable:
            log.warning("LLM sentiment skipped for %s: no ANTHROPIC_API_KEY; FinBERT labels are used", ticker)
            return results, usage

    aspects = aspects_for(ticker)
    company = companies()[ticker]
    for start in range(0, len(posts), BATCH_SIZE):
        batch = posts[start:start + BATCH_SIZE]
        valid_ids = {p["id"] for p in batch}
        try:
            result = llm.structured(system=ANALYSIS_SYSTEM, schema=PostLabels, fast=True, effort="low",
                                    user=analysis_user_prompt(company, ticker, aspects, batch),
                                    max_tokens=4000)
        except LLMUnavailable as exc:
            log.warning("LLM sentiment stopped: %s", exc)
            break
        except LLMError as exc:
            log.warning("LLM sentiment batch skipped: %s", exc)
            continue
        usage["calls"] += 1
        usage["input_tokens"] += result.input_tokens
        usage["output_tokens"] += result.output_tokens
        usage["cost_usd"] += result.cost_usd or 0.0
        usage["seconds"] += result.seconds
        for label in result.output.labels:
            analysis = accept(label, valid_ids, set(aspects), result.model)
            if analysis is not None and label.post_id not in results:
                results[label.post_id] = analysis
    return results, usage
