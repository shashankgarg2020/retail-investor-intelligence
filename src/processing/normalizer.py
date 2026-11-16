"""Text and timestamp normalisation, shared by all adapters."""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from datetime import datetime, timezone

_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
_WHITESPACE = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def clean_text(raw: str | None) -> str:
    """Display-safe text: HTML entities decoded, whitespace collapsed. Meaning is untouched."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return _WHITESPACE.sub(" ", text).strip()


def text_hash(text: str) -> str:
    """Hash of the text with case, URLs and punctuation removed.

    Catches copy-pasted reposts that differ only by a tracking link or spacing.
    It does not merge posts that merely say similar things.
    """
    canonical = _URL.sub("", text.lower())
    canonical = _NON_ALNUM.sub(" ", canonical).strip()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_timestamp(value: str | int | float | datetime) -> datetime:
    """Any platform timestamp -> timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(value, tz=timezone.utc)
    else:
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_int(value) -> int | None:
    """Engagement counts: keep None when the platform gave nothing."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
