"""Hybrid retrieval: Postgres full-text + pgvector, fused with Reciprocal Rank Fusion.

Each post is one retrieval document (docs/blueprint.md §9.2): social posts are short, so
there is no chunking. Metadata filters are applied inside both searches, so a
question about one company can never retrieve another company's posts.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

from sqlalchemy import Connection, text

from src.analytics import metrics
from src.analytics.metrics import Filters
from src.database.connection import vector_literal

RRF_K = 60
CANDIDATES = 30          # per retriever, before fusion
MAX_PER_AUTHOR = 2       # stop one prolific account dominating the evidence
MIN_SIMILARITY = 0.45    # vector-only hits below this are treated as off-topic


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank)."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return dict(scores)


def keyword_search(conn: Connection, f: Filters, keywords: list[str], limit: int = CANDIDATES) -> list[int]:
    if not keywords:
        return []
    clause, params = metrics.where(f, alias="v")
    # OR the terms: a natural-language question rarely appears whole in a post.
    query = " or ".join(dict.fromkeys(k.lower() for k in keywords))
    rows = conn.execute(text(f"""
        SELECT v.id
        FROM analysis_posts v
        JOIN posts p ON p.id = v.id,
             websearch_to_tsquery('english', :q) AS query
        WHERE {clause} AND p.tsv @@ query
        ORDER BY ts_rank_cd(p.tsv, query) DESC, v.created_at DESC
        LIMIT :limit
    """), {**params, "q": query, "limit": limit})
    return [r.id for r in rows]


def vector_search(conn: Connection, f: Filters, query_vector, limit: int = CANDIDATES) -> list[tuple[int, float]]:
    clause, params = metrics.where(f, alias="v")
    rows = conn.execute(text(f"""
        SELECT v.id, 1 - (e.embedding <=> CAST(:qv AS vector)) AS similarity
        FROM analysis_posts v
        JOIN embeddings e ON e.post_id = v.id
        WHERE {clause}
        ORDER BY e.embedding <=> CAST(:qv AS vector)
        LIMIT :limit
    """), {**params, "qv": vector_literal(query_vector), "limit": limit})
    return [(r.id, float(r.similarity)) for r in rows]


def hybrid_search(conn: Connection, f: Filters, question: str, keywords: list[str],
                  embed: Callable[[str], Any], top_k: int = 10) -> list[dict[str, Any]]:
    """Top posts for a question, each annotated with how it was found."""
    kw_ids = keyword_search(conn, f, keywords)
    vec = vector_search(conn, f, embed(question))
    similarity = dict(vec)
    kw_set = set(kw_ids)
    # A vector hit that is both dissimilar and unsupported by keywords is noise.
    vec_ids = [i for i, s in vec if s >= MIN_SIMILARITY or i in kw_set]

    scores = rrf([kw_ids, vec_ids])
    ranked = sorted(scores, key=scores.get, reverse=True)
    posts = metrics.posts_by_ids(conn, f.ticker, ranked)

    selected, per_author = [], defaultdict(int)
    for post in posts:
        author = post.get("author_id") or f"anon-{post['id']}"
        if per_author[author] >= MAX_PER_AUTHOR:
            continue
        per_author[author] += 1
        post["retrieval"] = {
            "rrf_score": round(scores[post["id"]], 5),
            "keyword_rank": kw_ids.index(post["id"]) + 1 if post["id"] in kw_set else None,
            "vector_rank": vec_ids.index(post["id"]) + 1 if post["id"] in vec_ids else None,
            "similarity": round(similarity[post["id"]], 3) if post["id"] in similarity else None,
        }
        selected.append(post)
        if len(selected) >= top_k:
            break
    return selected
