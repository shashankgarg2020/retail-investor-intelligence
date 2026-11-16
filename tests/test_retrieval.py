from datetime import datetime, timedelta, timezone

from src.intelligence.topics import keyword_hits
from src.intelligence.trends import period_change
from src.rag.planner import plan
from src.rag.query_engine import validate_citations
from src.rag.response_schema import AssistantDraft, Theme
from src.rag.retriever import rrf

NOW = datetime(2026, 11, 15, tzinfo=timezone.utc)
START = NOW - timedelta(days=30)


def make(question, ticker="VEON"):
    return plan(question, ticker, START, NOW, now=NOW)


# -------------------------------------------------------------------------- RRF

def test_rrf_rewards_agreement_between_retrievers():
    scores = rrf([[1, 2, 3], [3, 4, 1]])
    ranked = sorted(scores, key=scores.get, reverse=True)
    assert ranked[:2] == [1, 3]          # found by both
    assert set(ranked[2:]) == {2, 4}     # found by one


def test_rrf_formula():
    assert rrf([[7]], k=60) == {7: 1 / 61}
    assert rrf([[], []]) == {}


# ---------------------------------------------------------------------- planner

def test_planner_intents():
    assert make("What are the most discussed topics about VEON?").intent == "TOPIC"
    assert make("Compare negative sentiment across all three platforms.").intent == "COMPARISON"
    assert make("How many posts were collected?").intent == "COUNT"
    assert make("What changed over the last seven days?").intent == "TREND"
    assert make("Give me an overview of the discussion").intent == "SUMMARY"


def test_count_questions_do_not_use_retrieval():
    p = make("What percentage of posts are negative?")
    assert p.intent == "COUNT" and p.use_sql and not p.use_retrieval


def test_planner_extracts_filters():
    p = make("What are Reddit users worried about in the last 14 days?")
    assert p.platform == "reddit" and p.sentiment == "negative"
    assert (p.end - p.start).days == 14


def test_planner_takes_company_from_question_over_ui_default():
    assert make("What are investors saying about TTWO's GTA VI release?").ticker == "TTWO"
    assert make("Any concerns about Secure Trust Bank?").ticker == "STB"
    assert make("What are investors concerned about?").ticker == "VEON"


def test_comparison_never_filters_to_one_platform():
    assert make("Compare Reddit with the other platforms").platform is None


def test_planner_keywords_drop_company_and_filler_words():
    p = make("What are investors saying about TTWO's GTA VI release?")
    assert p.keywords == ["GTA", "VI", "release"]


def test_mixed_sentiment_words_mean_no_filter():
    assert make("Compare positive and negative discussions").sentiment is None


# -------------------------------------------------------------------- citations

def draft(*id_lists):
    themes = [Theme(title=f"t{i}", summary="s", stance="negative", evidence_ids=list(ids))
              for i, ids in enumerate(id_lists)]
    return AssistantDraft(answer="a", themes=themes, limitations=[], insufficient_evidence=False)


def test_invented_citations_are_removed():
    themes, dropped = validate_citations(draft([1, 2, 999], [3]), {1, 2, 3})
    assert [t.evidence_ids for t in themes] == [[1, 2], [3]] and dropped == 1


def test_theme_with_no_valid_evidence_is_dropped():
    themes, dropped = validate_citations(draft([404], [1]), {1})
    assert len(themes) == 1 and themes[0].evidence_ids == [1] and dropped == 1


# ----------------------------------------------------------------------- trends

def test_growth_from_zero_is_absolute_not_infinite():
    change = period_change(8, 0)
    assert change.percent is None and change.absolute == 8 and change.direction == "new"
    assert "no posts in previous period" in change.display


def test_small_counts_are_not_significant():
    assert not period_change(2, 1).significant      # +100% of almost nothing
    assert period_change(12, 6).significant
    assert period_change(12, 6).percent == 100.0
    assert period_change(3, 9).direction == "down"
    assert period_change(0, 0).direction == "flat"


# ----------------------------------------------------------------------- topics

def test_topic_keywords_are_company_specific():
    assert "gta_vi" in keyword_hits("TTWO", "GTA VI delayed again")
    assert "gta_vi" not in keyword_hits("VEON", "GTA VI delayed again")
    assert keyword_hits("VEON", "net debt and refinancing risk")["debt_balance_sheet"] == 2  # "net debt" (longest match) and "refinancing"
    assert "motor_finance_redress" in keyword_hits("STB", "FCA redress on motor finance")
