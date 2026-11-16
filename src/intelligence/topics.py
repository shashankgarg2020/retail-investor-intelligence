"""Topic assignment against a predefined taxonomy (docs/blueprint.md §8.3).

Keyword hits decide when they exist, because they are easy to explain to a user.
Embedding similarity to the topic description breaks ties and covers posts that
talk about a topic without using its keywords. Anything else is 'other'.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from config.settings import topics_for
from src.intelligence.embeddings import embed_documents

OTHER = "other"
SIMILARITY_THRESHOLD = 0.65  # tuned by inspection on collected posts; see README limitations


def _keyword_pattern(keywords: list[str]) -> re.Pattern[str]:
    alternatives = sorted((re.escape(k) for k in keywords), key=len, reverse=True)
    return re.compile(r"(?<![\w$])(?:" + "|".join(alternatives) + r")(?!\w)", re.IGNORECASE)


@lru_cache
def _taxonomy(ticker: str):
    topics = topics_for(ticker)
    ids = list(topics)
    patterns = [_keyword_pattern(topics[t]["keywords"]) for t in ids]
    vectors = embed_documents([f"{topics[t]['label']}. {topics[t]['description']}" for t in ids])
    return ids, patterns, vectors


def topic_label(ticker: str, topic_id: str) -> str:
    if topic_id == OTHER:
        return "Other"
    return topics_for(ticker).get(topic_id, {}).get("label", topic_id)


def keyword_hits(ticker: str, text: str) -> dict[str, int]:
    """Distinct keywords matched per topic. Pure function, no model needed."""
    topics = topics_for(ticker)
    hits = {}
    for topic_id, cfg in topics.items():
        found = {m.group(0).lower() for m in _keyword_pattern(cfg["keywords"]).finditer(text)}
        if found:
            hits[topic_id] = len(found)
    return hits


def assign(ticker: str, texts: list[str], vectors: np.ndarray) -> list[tuple[str, float, str]]:
    """Return (topic_id, score, method) per text. `vectors` are the texts' embeddings."""
    if not texts:
        return []
    ids, patterns, topic_vectors = _taxonomy(ticker)
    similarity = vectors @ topic_vectors.T  # both sides are unit length
    results = []
    for row, text in enumerate(texts):
        hits = np.array([len({m.group(0).lower() for m in p.finditer(text)}) for p in patterns])
        if hits.max() > 0:
            # Most keyword hits wins; similarity breaks ties.
            best = int(np.lexsort((similarity[row], hits))[-1])
            results.append((ids[best], float(similarity[row, best]), "keyword+embedding"))
            continue
        best = int(similarity[row].argmax())
        score = float(similarity[row, best])
        if score >= SIMILARITY_THRESHOLD:
            results.append((ids[best], score, "embedding"))
        else:
            results.append((OTHER, score, "below-threshold"))
    return results
