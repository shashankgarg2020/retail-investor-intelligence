"""Company relevance, stage 2: LLM check for posts the rules could not settle."""
from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel

from config.settings import companies
from src.processing.entity_resolution import IRRELEVANT, RELEVANT, Relevance
from src.rag.llm import LLMError, LLMUnavailable, get_llm
from src.rag.prompts import RELEVANCE_SYSTEM, relevance_user_prompt

log = logging.getLogger("pipeline.relevance")
BATCH_SIZE = 10


class Decision(BaseModel):
    post_id: int
    label: Literal["relevant", "irrelevant"]
    reason: str


class Decisions(BaseModel):
    decisions: list[Decision]


def classify_ambiguous(ticker: str, posts: list[dict[str, Any]]) -> tuple[dict[int, Relevance], dict[str, float]]:
    """Return decisions by post id, plus usage totals.

    Posts with no decision (no API key, a failed call, an id the model skipped)
    simply keep their 'possibly_relevant' label and stay out of analytics.
    """
    results: dict[int, Relevance] = {}
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "seconds": 0.0}
    if not posts:
        return results, usage
    try:
        llm = get_llm()
    except LLMUnavailable:
        log.warning("%d ambiguous %s posts left as possibly_relevant: no ANTHROPIC_API_KEY", len(posts), ticker)
        return results, usage

    for start in range(0, len(posts), BATCH_SIZE):
        batch = posts[start:start + BATCH_SIZE]
        valid_ids = {p["id"] for p in batch}
        try:
            result = llm.structured(system=RELEVANCE_SYSTEM, fast=True, schema=Decisions, effort="low",
                                    user=relevance_user_prompt(companies()[ticker], ticker, batch),
                                    max_tokens=2000)
        except LLMUnavailable as exc:
            log.warning("relevance check stopped: %s", exc)
            break
        except LLMError as exc:
            log.warning("relevance batch skipped: %s", exc)
            continue
        usage["calls"] += 1
        usage["input_tokens"] += result.input_tokens
        usage["output_tokens"] += result.output_tokens
        usage["cost_usd"] += result.cost_usd or 0.0
        usage["seconds"] += result.seconds
        for d in result.output.decisions:
            if d.post_id not in valid_ids:
                continue  # never accept an id we did not send
            label = RELEVANT if d.label == "relevant" else IRRELEVANT
            results[d.post_id] = Relevance(label, 0.75 if label == RELEVANT else 0.25,
                                           f"llm:{result.model}", d.reason[:300])
    return results, usage
