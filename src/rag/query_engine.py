"""The assistant: plan -> SQL metrics -> hybrid retrieval -> grounded generation -> validation.

Numbers come from SQL. The LLM only explains them and summarises the retrieved
posts. Every cited post id is checked against what was actually retrieved.
"""
from __future__ import annotations

import time
from dataclasses import replace
from datetime import datetime
from typing import Any, Callable

from sqlalchemy import Connection

from src.analytics import insights, metrics
from src.analytics.metrics import Filters
from src.rag import retriever
from src.rag.llm import LLMError, LLMUnavailable, get_llm
from src.rag.planner import Plan, plan as make_plan
from src.rag.prompts import ASSISTANT_SYSTEM, assistant_user_prompt
from src.rag.response_schema import AssistantDraft, ChatResponse, Theme

MAX_EVIDENCE = 14
MIN_EVIDENCE = 2   # fewer retrieved posts than this and the assistant abstains

STANDING_LIMITATIONS = [
    "Public social posts are not a representative sample of all shareholders.",
    "Sentiment and topic labels come from automated models and contain errors; check the source posts.",
]


def _context_post(post: dict[str, Any]) -> dict[str, Any]:
    """The fields the LLM sees for one post."""
    return {
        "id": post["id"], "platform": post["platform"], "created_at": post["created_at"].isoformat(),
        "sentiment": post["sentiment"], "topic": post.get("topic_label") or post.get("topic"),
        "likes": post["likes"], "title": post.get("title"), "parent_title": post.get("parent_title"),
        "text": post["text"],
    }


def gather(conn: Connection, p: Plan, embed: Callable[[str], Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect SQL metrics and evidence posts for a plan."""
    scope = Filters(p.ticker, p.start, p.end, p.platform)
    filtered = replace(scope, sentiment=p.sentiment)

    data: dict[str, Any] = {"overview": metrics.overview(conn, scope)}
    if p.intent in ("COUNT", "COMPARISON", "SENTIMENT"):
        data["by_platform"] = metrics.by_platform(conn, scope)
        data["coverage"] = metrics.coverage(conn, p.ticker, p.start, p.end)
    if p.intent in ("TOPIC", "TREND", "EXPLANATION", "EVIDENCE", "SUMMARY"):
        ranked = [t for t in metrics.topics(conn, filtered) if t["posts"] > 0]
        data["topics" if not p.sentiment else f"topics_{p.sentiment}_posts_only"] = [
            {k: t[k] for k in ("label", "posts", "positive", "neutral", "negative",
                               "previous_posts", "change_display", "change_significant")}
            for t in ranked[:8]
        ]
    else:
        ranked = []
    if p.intent == "TREND":
        data["emerging_concerns"] = insights.emerging_concerns(conn, scope)

    evidence: dict[int, dict[str, Any]] = {}
    if p.use_retrieval:
        # Representative posts for the leading topics anchor "top N" style questions.
        if p.intent in ("TOPIC", "TREND", "EXPLANATION", "EVIDENCE"):
            for topic in [t for t in ranked if t["topic"] != "other"][: p.limit]:
                for post in metrics.representative_posts(conn, replace(filtered, topic=topic["topic"]), limit=2):
                    evidence.setdefault(post["id"], post)
        for post in retriever.hybrid_search(conn, filtered, p.question, p.keywords, embed, top_k=10):
            evidence.setdefault(post["id"], post)
    return data, list(evidence.values())[:MAX_EVIDENCE]


def _extractive_answer(p: Plan, data: dict[str, Any], posts: list[dict[str, Any]]) -> AssistantDraft:
    """No-LLM fallback: restate the SQL metrics and group evidence by topic. Nothing is generated."""
    o = data["overview"]
    answer = (f"{o['posts']} relevant posts were analysed for {p.ticker} in this period: "
              f"{o['positive']} positive, {o['neutral']} neutral and {o['negative']} negative. "
              "The posts below are the closest matches to your question, grouped by topic.")
    groups: dict[str, list[dict[str, Any]]] = {}
    for post in posts:
        groups.setdefault(post.get("topic_label") or "Other", []).append(post)
    themes = []
    for label, items in list(groups.items())[:5]:
        counts = {s: sum(i["sentiment"] == s for i in items) for s in ("positive", "negative")}
        stance = ("mixed" if counts["positive"] and counts["negative"] else
                  "positive" if counts["positive"] else "negative" if counts["negative"] else "neutral")
        themes.append(Theme(title=label, stance=stance, evidence_ids=[i["id"] for i in items],
                            summary=f"{len(items)} retrieved post(s) on this topic. Read the sources for detail."))
    return AssistantDraft(answer=answer, themes=themes, insufficient_evidence=False, limitations=[
        "No LLM is configured, so this is a listing of metrics and matching posts, not a written summary."])


def validate_citations(draft: AssistantDraft, allowed_ids: set[int]) -> tuple[list[Theme], int]:
    """Keep only citations that point at retrieved posts. Drop themes left with none."""
    themes, dropped = [], 0
    for theme in draft.themes:
        valid = [i for i in dict.fromkeys(theme.evidence_ids) if i in allowed_ids]
        dropped += len(theme.evidence_ids) - len(valid)
        if valid:
            themes.append(theme.model_copy(update={"evidence_ids": valid}))
    return themes, dropped


def answer(conn: Connection, question: str, ticker: str, start: datetime, end: datetime,
           platform: str | None, embed: Callable[[str], Any]) -> ChatResponse:
    timings: dict[str, float] = {}
    t = time.perf_counter()
    p = make_plan(question, ticker, start, end, platform)
    data, posts = gather(conn, p, embed)
    timings["retrieval_and_sql"] = round(time.perf_counter() - t, 3)

    overview = data["overview"]
    platforms = data.get("by_platform") or metrics.by_platform(conn, Filters(p.ticker, p.start, p.end, p.platform))
    scope = {"company": p.ticker, "from": p.start.date().isoformat(), "to": p.end.date().isoformat(),
             "platform": p.platform or "all", "sentiment_filter": p.sentiment or "none"}
    limitations = list(STANDING_LIMITATIONS)
    if overview["small_sample"]:
        limitations.insert(0, f"Small sample: only {overview['posts']} relevant posts in this scope.")
    if overview["synthetic_posts"]:
        limitations.insert(0, f"{overview['synthetic_posts']} of the analysed posts are synthetic "
                              "demonstration data, not real public content.")

    def respond(draft: AssistantDraft, generated_by: str, usage: dict[str, Any] | None = None) -> ChatResponse:
        themes, dropped = validate_citations(draft, {post["id"] for post in posts})
        cited = list(dict.fromkeys(i for theme in themes for i in theme.evidence_ids))
        extra, insufficient = [], draft.insufficient_evidence
        if p.use_retrieval and not themes and not insufficient:
            # An answer about the discussion with no checkable source is not shown as grounded.
            insufficient = True
            extra.append("The answer could not be tied to any retrieved post, so treat it as unsupported.")
        by_id = {post["id"]: post for post in posts}
        return ChatResponse(
            answer=draft.answer, themes=themes, evidence=[by_id[i] for i in cited],
            insufficient_evidence=insufficient, intent=p.intent, scope=scope,
            posts_analysed=overview["posts"], platform_breakdown=platforms, metrics=data,
            limitations=extra + draft.limitations + limitations, generated_by=generated_by,
            retrieved_count=len(posts), dropped_citations=dropped, planner_notes=p.notes,
            timings_seconds=timings, usage=usage,
        )

    # Abstain without calling the LLM when there is nothing to ground an answer in.
    if overview["posts"] == 0 or (p.use_retrieval and len(posts) < MIN_EVIDENCE):
        reason = ("No relevant posts are stored for this company, period and platform."
                  if overview["posts"] == 0 else
                  "The collected posts do not contain enough on this to answer.")
        return respond(AssistantDraft(answer=reason, themes=[], limitations=[], insufficient_evidence=True),
                       "abstained (no LLM call)")

    t = time.perf_counter()
    try:
        result = get_llm().structured(
            system=ASSISTANT_SYSTEM, schema=AssistantDraft,
            # Low effort: the task is summarising pre-computed metrics and retrieved posts,
            # and thinking tokens were most of the latency (measured ~9.5s medium vs ~4.5-8s low).
            effort="low",
            user=assistant_user_prompt(p.question, scope, data, [_context_post(x) for x in posts]),
        )
    except LLMUnavailable:
        return respond(_extractive_answer(p, data, posts), "extractive fallback (no LLM configured)")
    except LLMError as exc:
        draft = _extractive_answer(p, data, posts)
        draft.limitations.insert(0, f"The LLM call failed ({exc}); showing the non-LLM fallback.")
        return respond(draft, "extractive fallback (LLM error)")
    timings["llm"] = round(time.perf_counter() - t, 3)
    usage = {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
             "cost_usd": result.cost_usd}
    return respond(result.output, result.model, usage)
