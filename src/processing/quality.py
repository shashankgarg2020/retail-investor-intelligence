"""Language detection and cautious quality flags.

Labels describe the content, not the account. Nothing here claims an author is
a bot: 'potentially_automated' means the same author posted the same template
several times in the collected data.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0  # langdetect is random by default; make it repeatable

OK = "ok"
LOW_QUALITY = "low_quality"
PROMOTIONAL = "promotional"
POTENTIALLY_AUTOMATED = "potentially_automated"

_URL = re.compile(r"https?://\S+")
_CASHTAG = re.compile(r"\$[A-Za-z]{1,6}\b")
_HASHTAG = re.compile(r"#\w+")
_WORD = re.compile(r"[A-Za-z]{2,}")
_DIGITS = re.compile(r"[\d.,%$+-]+")
_NON_ALNUM = re.compile(r"[^a-z]+")
_PROMO = re.compile(
    r"join (my|our) (telegram|discord|whatsapp)|free (signals|trial|alerts)|discord\.gg|t\.me/"
    r"|link in bio|sign up (now|today)|start (your )?(free )?trial|guaranteed (returns|profit)"
    r"|dm me|subscribe (now|today) for",
    re.IGNORECASE,
)

MIN_WORDS = 4
MAX_CASHTAGS = 5
MAX_LINKS = 2
TEMPLATE_REPEATS = 3


def detect_language(text: str, platform_hint: str | None = None) -> str:
    """Two-letter code, or 'und' when it cannot be determined.

    Short social posts confuse statistical detectors, so a non-English verdict is
    only accepted when the detector is confident and the text is long enough.
    """
    if platform_hint:
        return platform_hint
    words = _WORD.findall(_URL.sub("", text))
    if len(words) < 3:
        return "und"
    try:
        guesses = detect_langs(text)
    except LangDetectException:
        return "und"
    top = guesses[0]
    if top.lang == "en":
        return "en"
    if top.prob >= 0.90 and len(text) >= 40:
        return top.lang
    mostly_ascii = sum(ch.isascii() for ch in text) / len(text) > 0.95
    return "en" if mostly_ascii else "und"


def assess_text(text: str) -> tuple[str, str]:
    """Flag for one post judged on its own."""
    body = _URL.sub(" ", text)
    links = len(_URL.findall(text))
    cashtags = {c.upper() for c in _CASHTAG.findall(body)}
    hashtags = _HASHTAG.findall(body)
    words = _WORD.findall(_HASHTAG.sub(" ", _CASHTAG.sub(" ", body)))

    if _PROMO.search(text):
        return PROMOTIONAL, "promotional call to action"
    if links > MAX_LINKS:
        return PROMOTIONAL, f"{links} links"
    if len(cashtags) > MAX_CASHTAGS:
        return LOW_QUALITY, f"lists {len(cashtags)} tickers"
    if len(words) < MIN_WORDS:
        return LOW_QUALITY, "too little text to analyse"
    if len(hashtags) > 8 or len(hashtags) > len(words):
        return LOW_QUALITY, "mostly hashtags"
    return OK, ""


def template_key(text: str) -> str:
    """Text with links, numbers and punctuation removed: equal for templated posts."""
    stripped = _DIGITS.sub(" ", _URL.sub(" ", text.lower()))
    return _NON_ALNUM.sub(" ", stripped).strip()


def repeated_templates(posts: list[dict[str, Any]]) -> set[int]:
    """Post ids where one author repeated the same template TEMPLATE_REPEATS+ times."""
    counts: Counter[tuple[str, str]] = Counter()
    keys: dict[int, tuple[str, str]] = {}
    for post in posts:
        if not post.get("author_id"):
            continue
        key = (post["author_id"], template_key(post["text"]))
        if len(key[1]) < 30:
            continue  # short phrases repeat naturally
        keys[post["id"]] = key
        counts[key] += 1
    return {pid for pid, key in keys.items() if counts[key] >= TEMPLATE_REPEATS}
