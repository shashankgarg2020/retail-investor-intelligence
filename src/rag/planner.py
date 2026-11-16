"""Deterministic query planner (docs/blueprint.md §9.6). No model call, fully auditable.

Turns a question into: an intent, the filters it implies, and which data
sources to use (SQL metrics, retrieval, or both).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from config.settings import PLATFORMS, companies

# Checked in order; the first match wins.
_INTENTS: list[tuple[str, str]] = [
    ("COMPARISON", r"\bcompar\w*|\bversus\b|\bvs\.?\b|which platform|across (all |the )?(three |3 )?platforms|differ\w* (across|between)"),
    ("TREND", r"\bchang\w*|\btrend\w*|\bemerging\b|\bincreas\w*|\bdeclin\w*|\brising\b|\bgrowing\b|\bnew (concern|topic|theme)s?"),
    ("COUNT", r"how many|number of|\bcount\b|\bvolume\b|percentage|what percent|\bshare of\b|how much discussion"),
    ("TOPIC", r"\btopics?\b|\bthemes?\b|most discussed|talking about (the )?most"),
    ("EVIDENCE", r"\bevidence\b|show (me )?(the )?(posts|examples)|examples of|\bquote"),
    ("EXPLANATION", r"\bwhy\b|driving|\breasons?\b|\bconcern\w*|\bworr\w*|\brisks?\b|\bcritici\w*"),
    ("SENTIMENT", r"\bsentiment\b|positive or negative|\bfeel\w*\b|\bmood\b|\bbullish\b|\bbearish\b"),
]
_NEGATIVE = re.compile(r"\bnegativ\w*|\bconcern\w*|\bworr\w*|\bbearish\b|\bcritici\w*|\brisks?\b|\bcomplain\w*|\bfear\w*", re.I)
_POSITIVE = re.compile(r"\bpositiv\w*|\bbullish\b|\boptimis\w*|\bexcit\w*|\bpraise\w*", re.I)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "fourteen": 14, "thirty": 30}
_NUM = r"(\d+|" + "|".join(_NUMBER_WORDS) + r")"
_LAST_N = re.compile(rf"(?:last|past|previous)\s+{_NUM}\s+(day|week|month)s?", re.I)
_TOP_N = re.compile(rf"(?:top|biggest|main|largest)\s+{_NUM}\b|\b{_NUM}\s+(?:biggest|main|top|largest)", re.I)

# Words that carry no retrieval signal. Company names are added per company.
_STOPWORDS = set("""
a about across all an and any are as at be been being biggest but by can compare did do does for from
had has have how i in investors investor is it its main me most of on or over people say saying said
show tell than that the their them there these they this those to top us was what when where which
who why will with would you discussion discussions discussed discussing talking evidence posts post
platform platforms last past days day week weeks month months changed seven three
topic topics theme themes concern concerns concerned worried negative positive sentiment many
were collected supports support reddit youtube bluesky think thinking feel about
""".split())


@dataclass
class Plan:
    question: str
    intent: str
    ticker: str
    start: datetime
    end: datetime
    platform: str | None
    sentiment: str | None
    limit: int
    use_sql: bool
    use_retrieval: bool
    keywords: list[str]
    notes: list[str]


def _number(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token.lower()]


def detect_ticker(question: str) -> str | None:
    found = []
    for ticker, cfg in companies().items():
        names = {ticker, *cfg["aliases"], *cfg["strong_terms"]}
        if any(re.search(rf"(?<![\w$]){re.escape(n)}(?!\w)", question, re.I) for n in names):
            found.append(ticker)
    return found[0] if len(found) == 1 else None


def plan(question: str, default_ticker: str, default_start: datetime, default_end: datetime,
         default_platform: str | None = None, now: datetime | None = None) -> Plan:
    now = now or datetime.now(timezone.utc)
    notes: list[str] = []
    q = question.strip()

    intent = next((name for name, pattern in _INTENTS if re.search(pattern, q, re.I)), "SUMMARY")

    ticker = detect_ticker(q) or default_ticker
    if ticker != default_ticker:
        notes.append(f"Company taken from the question: {ticker}.")

    start, end = default_start, default_end
    match = _LAST_N.search(q)
    days = None
    if match:
        days = _number(match.group(1)) * {"day": 1, "week": 7, "month": 30}[match.group(2).lower()]
    elif re.search(r"\b(this|last|past) week\b", q, re.I):
        days = 7
    elif re.search(r"\b(this|last|past) month\b", q, re.I):
        days = 30
    if days:
        start, end = now - timedelta(days=days), now
        notes.append(f"Date range taken from the question: last {days} days.")

    named = [p for p in PLATFORMS if re.search(rf"\b{p}\b", q, re.I)]
    platform = named[0] if len(named) == 1 and intent != "COMPARISON" else default_platform
    if intent == "COMPARISON":
        platform = None  # a comparison needs every platform

    negative, positive = bool(_NEGATIVE.search(q)), bool(_POSITIVE.search(q))
    sentiment = "negative" if negative and not positive else "positive" if positive and not negative else None
    match = _TOP_N.search(q)
    limit = _number(next(g for g in match.groups() if g)) if match else 3

    company_words = {w.lower() for n in (ticker, *companies()[ticker]["aliases"]) for w in re.findall(r"\w+", n)}
    words = [re.sub(r"'s$", "", w) for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", q)]
    keywords = [w for w in words if w.lower() not in _STOPWORDS and w.lower() not in company_words]

    return Plan(
        question=q, intent=intent, ticker=ticker, start=start, end=end, platform=platform,
        sentiment=sentiment, limit=max(1, min(limit, 10)),
        use_sql=True,  # scope metrics are always attached so the answer can state its coverage
        use_retrieval=intent != "COUNT",
        keywords=keywords, notes=notes,
    )
