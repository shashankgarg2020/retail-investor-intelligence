"""Process collected posts: quality, relevance, sentiment, topic, embedding.

    python -m scripts.run_pipeline                  # everything not yet processed
    python -m scripts.run_pipeline --company VEON
    python -m scripts.run_pipeline --reprocess      # redo all (after a rule or model change)
    python -m scripts.run_pipeline --no-llm         # rules + FinBERT only (no LLM relevance or stance)

Runs as a batch job, separate from the API, so the ML models are never loaded
in the process that serves users.
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import Counter
from typing import Any

from config.settings import EMBEDDING_MODEL, SENTIMENT_MODEL, SUPPORTED_LANGUAGES, companies
from src.database import repositories as repo
from src.database.connection import get_engine
from src.processing import quality
from src.processing.deduplication import find_duplicates
from src.processing.entity_resolution import IRRELEVANT, POSSIBLY, resolve
from src.processing.relevance_llm import classify_ambiguous

log = logging.getLogger("pipeline")


def full_text(post: dict[str, Any]) -> str:
    """Title plus body, without repeating the title when it is the body."""
    title, text = post.get("title"), post["text"]
    return text if not title or title == text else f"{title}. {text}"


def process_company(ticker: str, reprocess: bool, use_llm: bool) -> dict[str, Any]:
    engine = get_engine()
    timings: dict[str, float] = {}
    with engine.connect() as conn:
        posts = repo.unprocessed_posts(conn, ticker, reprocess)
        known = {} if reprocess else repo.earlier_hashes(conn, ticker)
    stats: dict[str, Any] = {"ticker": ticker, "posts": len(posts)}
    if not posts:
        return stats

    # 1. Language, quality, duplicates, relevance: cheap, rule-based, every post.
    t = time.perf_counter()
    automated = quality.repeated_templates(posts)
    duplicates = find_duplicates(posts, known)
    rows: dict[int, dict[str, Any]] = {}
    for post in posts:
        text = full_text(post)
        post["language"] = quality.detect_language(text, post.get("language"))
        flag, _ = quality.assess_text(text)
        if flag == quality.OK and post["id"] in automated:
            flag = quality.POTENTIALLY_AUTOMATED
        relevance = resolve(ticker, text, post.get("parent_title"))
        rows[post["id"]] = {
            "post_id": post["id"], "relevance_label": relevance.label,
            "relevance_score": relevance.score, "relevance_method": relevance.method,
            "relevance_reason": relevance.reason, "quality_flag": flag,
            "duplicate_of": duplicates.get(post["id"]),
            "sentiment": None, "sentiment_confidence": None, "sentiment_model": None,
            "topic": None, "topic_score": None, "topic_method": None,
        }
    timings["rules"] = time.perf_counter() - t

    def worth_analysing(post: dict[str, Any]) -> bool:
        row = rows[post["id"]]
        return (row["quality_flag"] == quality.OK and row["duplicate_of"] is None
                and post["language"] in SUPPORTED_LANGUAGES)

    # 2. LLM check, only for ambiguous posts that would otherwise reach analytics.
    ambiguous = [p for p in posts if rows[p["id"]]["relevance_label"] == POSSIBLY and worth_analysing(p)]
    stats["ambiguous"] = len(ambiguous)
    if use_llm and ambiguous:
        t = time.perf_counter()
        decisions, usage = classify_ambiguous(ticker, ambiguous)
        for post_id, relevance in decisions.items():
            rows[post_id].update(relevance_label=relevance.label, relevance_score=relevance.score,
                                 relevance_method=relevance.method, relevance_reason=relevance.reason)
        timings["llm_relevance"] = time.perf_counter() - t
        stats["llm_resolved"] = len(decisions)
        stats["llm_usage"] = usage

    # 3. Models, only for posts that can count: skip irrelevant, low-quality and duplicates.
    targets = [p for p in posts if rows[p["id"]]["relevance_label"] != IRRELEVANT and worth_analysing(p)]
    stats["analysed"] = len(targets)
    vectors = None
    aspects: dict[int, tuple[list[tuple[str, str]], str]] = {}
    if targets:
        from src.intelligence import embeddings, sentiment, topics  # heavy imports, only when needed
        texts = [full_text(p) for p in targets]

        # FinBERT is the measured baseline and the fallback. It needs PyTorch, which the lightweight
        # cloud deployment leaves out; there the LLM stance below labels every post.
        if sentiment.available():
            t = time.perf_counter()
            sentiments = sentiment.classify(texts)
            timings["sentiment"] = time.perf_counter() - t
        else:
            log.info("FinBERT not installed (requirements-ml.txt); using the LLM stance only")
            sentiments = [(None, None)] * len(texts)

        t = time.perf_counter()
        vectors = embeddings.embed_documents(texts)
        timings["embedding"] = time.perf_counter() - t

        t = time.perf_counter()
        assigned = topics.assign(ticker, texts, vectors)
        timings["topics"] = time.perf_counter() - t

        for post, (label, score), (topic, topic_score, method) in zip(targets, sentiments, assigned):
            rows[post["id"]].update(sentiment=label, sentiment_confidence=score,
                                    sentiment_model=SENTIMENT_MODEL if label else None, finbert_sentiment=label,
                                    finbert_confidence=score, topic=topic,
                                    topic_score=topic_score, topic_method=method)

        # Investor stance, post type and aspects from the LLM. FinBERT stays the fallback.
        if use_llm:
            from src.intelligence import llm_analysis
            t = time.perf_counter()
            analysed, usage = llm_analysis.analyse(ticker, targets)
            timings["llm_sentiment"] = time.perf_counter() - t
            stats["llm_sentiment"] = {"labelled": len(analysed), **{k: round(v, 4) for k, v in usage.items()}}
            for post_id, a in analysed.items():
                # The LLM gives no score; a softmax-style number would imply a calibration it does not have.
                rows[post_id].update(sentiment=a.sentiment, sentiment_confidence=None,
                                     sentiment_model=a.model, sentiment_reason=a.reason,
                                     post_type=a.post_type)
                aspects[post_id] = (a.aspects, a.model)

    # 4. Persist.
    t = time.perf_counter()
    with engine.begin() as conn:
        for post in posts:
            repo.set_language(conn, post["id"], post["language"])
        repo.save_analysis(conn, rows.values())
        for post in posts:
            # Posts without a fresh analysis (now irrelevant, or the LLM was off) lose stale aspects.
            views, model = aspects.get(post["id"], ([], ""))
            repo.save_aspects(conn, post["id"], views, model)
        if vectors is not None:
            for post, vector in zip(targets, vectors):
                repo.save_embedding(conn, post["id"], EMBEDDING_MODEL, vector)
    timings["persist"] = time.perf_counter() - t

    stats["relevance"] = dict(Counter(r["relevance_label"] for r in rows.values()))
    stats["quality"] = dict(Counter(r["quality_flag"] for r in rows.values()))
    stats["duplicates"] = len(duplicates)
    stats["languages"] = dict(Counter(p["language"] for p in posts))
    stats["sentiment"] = dict(Counter(r["sentiment"] for r in rows.values() if r["sentiment"]))
    stats["timings_seconds"] = {k: round(v, 2) for k, v in timings.items()}
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--company", default="all")
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tickers = list(companies()) if args.company.lower() == "all" else [args.company.upper()]
    for ticker in tickers:
        stats = process_company(ticker, args.reprocess, use_llm=not args.no_llm)
        print(f"\n== {ticker}: {stats['posts']} posts processed")
        for key in ("relevance", "quality", "duplicates", "languages", "ambiguous", "llm_resolved",
                    "llm_usage", "analysed", "llm_sentiment", "sentiment", "timings_seconds"):
            if key in stats:
                print(f"   {key}: {stats[key]}")


if __name__ == "__main__":
    main()
