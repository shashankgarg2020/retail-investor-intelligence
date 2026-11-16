"""Text-level duplicate detection.

Platform-ID duplicates never get this far: the unique key on
(company, platform, external_id) absorbs them at insert time.

This pass catches the same text posted again under a different ID. It matches
exact normalised text only. Two people independently raising the same concern in
their own words are two opinions and are both kept.
"""
from __future__ import annotations

from typing import Any

# Short phrases ("bullish", "to the moon") are legitimately repeated by different people.
MIN_DUPLICATE_LENGTH = 40


def find_duplicates(posts: list[dict[str, Any]], known: dict[str, int] | None = None) -> dict[int, int]:
    """Map duplicate post id -> id of the earliest post with the same text.

    `posts` must be ordered oldest first. `known` holds hashes already stored.
    """
    first_seen: dict[str, int] = dict(known or {})
    duplicates: dict[int, int] = {}
    for post in posts:
        if len(post["text"]) < MIN_DUPLICATE_LENGTH:
            continue
        original = first_seen.get(post["text_hash"])
        if original is None:
            first_seen[post["text_hash"]] = post["id"]
        elif original != post["id"]:
            duplicates[post["id"]] = original
    return duplicates
