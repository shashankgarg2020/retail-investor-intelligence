"""Company relevance, stage 1: deterministic rules.

Decides whether a candidate post is really about the company. Obvious cases are
settled here. Ambiguous ones come back as 'possibly_relevant' and are passed to
the LLM check in src/rag/llm.py. Every decision keeps its score, method and
reason so it can be audited.

Relevance is deliberately independent of sentiment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from config.settings import company_config

RULES_VERSION = "rules-v2"

RELEVANT = "relevant"
POSSIBLY = "possibly_relevant"
IRRELEVANT = "irrelevant"


@dataclass
class Relevance:
    label: str
    score: float
    method: str
    reason: str


def _pattern(terms: list[str], ignore_case: bool = True) -> re.Pattern[str] | None:
    """Match on whole terms. '$' is treated as part of a cashtag."""
    if not terms:
        return None
    alternatives = sorted((re.escape(t) for t in terms), key=len, reverse=True)
    return re.compile(r"(?<![\w$])(?:" + "|".join(alternatives) + r")(?!\w)",
                      re.IGNORECASE if ignore_case else 0)


@lru_cache
def _patterns(ticker: str) -> dict[str, re.Pattern[str] | None]:
    cfg = company_config()
    company = cfg["companies"][ticker]
    return {
        "strong": _pattern(company["strong_terms"]),
        "strong_exact": _pattern(company.get("strong_terms_exact_case", []), ignore_case=False),
        "weak": _pattern(company["weak_terms"]),
        "context": _pattern(company["context_terms"]),
        "negative": _pattern(company["negative_terms"]),
        "finance": _pattern(cfg["finance_terms"]),
        "business": _pattern(cfg.get("business_terms", [])),
    }


def _hits(pattern: re.Pattern[str] | None, text: str) -> list[str]:
    if pattern is None:
        return []
    return sorted({m.group(0).lower() for m in pattern.finditer(text)})


def _strong(p: dict[str, re.Pattern[str] | None], text: str) -> list[str]:
    return sorted(set(_hits(p["strong"], text)) | set(_hits(p["strong_exact"], text)))


def resolve(ticker: str, text: str, parent_title: str | None = None) -> Relevance:
    """Classify one post. `text` should include the title when there is one."""
    p = _patterns(ticker)
    strong, weak = _strong(p, text), _hits(p["weak"], text)
    context, negative = _hits(p["context"], text), _hits(p["negative"], text)
    finance, business = _hits(p["finance"], text), _hits(p["business"], text)

    def out(label: str, score: float, reason: str) -> Relevance:
        return Relevance(label, score, RULES_VERSION, reason)

    if strong:
        if finance:
            return out(RELEVANT, 0.95, f"company identifier {strong} with investor vocabulary {finance[:3]}")
        if negative:
            # The name is there, but so is a sign it means something else (a music act, a person).
            return out(POSSIBLY, 0.45, f"company identifier {strong} next to disambiguator {negative[:3]}")
        return out(RELEVANT, 0.85, f"company identifier {strong}")

    if weak:
        if negative:
            return out(IRRELEVANT, 0.05, f"ambiguous term {weak} with disambiguator {negative[:3]}")
        if context and finance:
            return out(RELEVANT, 0.80, f"ambiguous term {weak} with company context {context[:3]} "
                                       f"and investor vocabulary {finance[:3]}")
        if context or finance:
            return out(POSSIBLY, 0.50, f"ambiguous term {weak} with partial support {(context or finance)[:3]}")
        if business:
            # Product news (sales, delays, launch dates) can move a company; let the LLM judge.
            return out(POSSIBLY, 0.40, f"product term {weak} with business-event words {business[:3]}")
        return out(IRRELEVANT, 0.15, f"only product or ambiguous term {weak}, no investor context")

    # Nothing in the text itself. A reply or comment can still be about the company
    # when the thing it replies to clearly is.
    if parent_title:
        parent_strong = _strong(p, parent_title)
        if parent_strong and finance:
            return out(RELEVANT, 0.70, f"comment on company content {parent_strong} "
                                       f"with investor vocabulary {finance[:3]}")
        if parent_strong:
            return out(POSSIBLY, 0.40, f"comment on company content {parent_strong}, no explicit mention")
        if _hits(p["weak"], parent_title) and finance:
            return out(POSSIBLY, 0.35, "comment with investor vocabulary on product-related content")

    return out(IRRELEVANT, 0.05, "no company identifier found")
