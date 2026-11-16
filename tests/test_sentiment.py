"""Sentiment: FinBERT wrapper, the LLM investor-stance labeller, and the briefing that reads them.

No model download and no API call: FinBERT's pipeline and the LLM client are replaced by fakes.
"""
from types import SimpleNamespace

import pytest

from src.analytics import briefing
from src.intelligence import llm_analysis, sentiment
from src.intelligence.llm_analysis import AspectView, PostLabel, PostLabels
from src.rag.llm import LLMError

# ---------------------------------------------------------------------- FinBERT

def test_finbert_labels_are_lowercased_and_scored(monkeypatch):
    seen = {}

    def fake_pipeline(texts, **kwargs):
        seen["texts"], seen["kwargs"] = texts, kwargs
        return [{"label": "Positive", "score": 0.91}, {"label": "Negative", "score": 0.7}]

    monkeypatch.setattr(sentiment, "_classifier", lambda: fake_pipeline)
    out = sentiment.classify(["good results", "x" * 5000])
    assert out == [("positive", 0.91), ("negative", 0.7)]
    assert len(seen["texts"][1]) == sentiment.MAX_CHARS          # long posts are bounded
    assert seen["kwargs"]["truncation"] is True


def test_finbert_empty_input_skips_the_model(monkeypatch):
    monkeypatch.setattr(sentiment, "_classifier", lambda: pytest.fail("model should not load"))
    assert sentiment.classify([]) == []


# ------------------------------------------------------------ LLM investor stance

class FakeLLM:
    """Returns canned labels per batch; records calls."""

    def __init__(self, batches):
        self.batches, self.calls = list(batches), []

    def structured(self, **kwargs):
        self.calls.append(kwargs)
        out = self.batches.pop(0)
        if isinstance(out, Exception):
            raise out
        return SimpleNamespace(output=PostLabels(labels=out), model="claude-haiku-5-5",
                               input_tokens=100, output_tokens=50, cost_usd=0.0001, seconds=0.5)


def post(i, text="Take-Two forecasts $8 billion in net bookings"):
    return {"id": i, "platform": "bluesky", "text": text, "title": None, "parent_title": None}


def test_stance_aspects_and_type_are_kept():
    llm = FakeLLM([[PostLabel(post_id=1, stance="positive", post_type="news", reason="bookings forecast up",
                              aspects=[AspectView(aspect="bookings_sales", stance="positive")])]])
    results, usage = llm_analysis.analyse("TTWO", [post(1)], llm=llm)
    a = results[1]
    assert (a.sentiment, a.post_type, a.aspects) == ("positive", "news", [("bookings_sales", "positive")])
    assert a.model == "llm:claude-haiku-5-5"
    assert usage["calls"] == 1 and usage["input_tokens"] == 100
    # The prompt lists the company's configured aspects, and the call uses the cheap model.
    assert "bookings_sales" in llm.calls[0]["user"] and llm.calls[0]["fast"] is True


def test_unknown_ids_and_aspects_are_rejected():
    llm = FakeLLM([[
        PostLabel(post_id=999, stance="negative", post_type="opinion", reason="not sent"),
        PostLabel(post_id=1, stance="negative", post_type="opinion", reason="worried",
                  aspects=[AspectView(aspect="made_up", stance="negative"),
                           AspectView(aspect="gta_vi", stance="negative"),
                           AspectView(aspect="gta_vi", stance="positive")]),
    ]])
    results, _ = llm_analysis.analyse("TTWO", [post(1)], llm=llm)
    assert set(results) == {1}                              # never accept an id we did not send
    assert results[1].aspects == [("gta_vi", "negative")]   # unknown aspect dropped, first duplicate kept


def test_failed_batch_is_skipped_and_others_continue():
    posts = [post(i) for i in range(1, 13)]                 # two batches of 10 and 2
    llm = FakeLLM([LLMError("boom"),
                   [PostLabel(post_id=11, stance="neutral", post_type="question", reason="asks"),
                    PostLabel(post_id=12, stance="negative", post_type="opinion", reason="sold")]])
    results, usage = llm_analysis.analyse("TTWO", posts, llm=llm)
    assert set(results) == {11, 12} and usage["calls"] == 1


def test_no_posts_no_call():
    llm = FakeLLM([])
    results, usage = llm_analysis.analyse("VEON", [], llm=llm)
    assert results == {} and usage["calls"] == 0 and llm.calls == []


# --------------------------------------------------------------------- briefing

def overview(posts, pos, neu, neg, **extra):
    net = (pos - neg) / posts if posts else None
    return {"posts": posts, "positive": pos, "neutral": neu, "negative": neg,
            "positive_pct": 100 * pos / posts if posts else None,
            "neutral_pct": 100 * neu / posts if posts else None,
            "negative_pct": 100 * neg / posts if posts else None,
            "post_weighted_sentiment": net, "small_sample": posts < 30, "synthetic_posts": 0, **extra}


@pytest.mark.parametrize("pos,neu,neg,expected", [
    (30, 10, 5, "mostly positive"), (5, 10, 30, "mostly negative"),
    (12, 20, 10, "mixed"), (3, 40, 2, "mostly neutral"),
])
def test_tone_words(pos, neu, neg, expected):
    assert briefing.tone(overview(pos + neu + neg, pos, neu, neg)) == expected


def test_briefing_quotes_its_numbers_and_flags_worries():
    aspects = [
        {"aspect": "debt", "label": "Debt and leverage", "posts": 6, "positive": 1, "neutral": 0, "negative": 5, "net": -0.67},
        {"aspect": "revenue_growth", "label": "Revenue and growth", "posts": 4, "positive": 4, "neutral": 0, "negative": 0, "net": 1.0},
        {"aspect": "management", "label": "Management", "posts": 1, "positive": 0, "neutral": 0, "negative": 1, "net": -1.0},
    ]
    types = [{"post_type": "news", "posts": 30}, {"post_type": "opinion", "posts": 10}]
    out = briefing.compose("VEON Ltd.", overview(40, 12, 20, 8), overview(20, 4, 12, 4), [], aspects, [], [], types)
    assert out["headline"].startswith("40 relevant posts about VEON Ltd. (+100% versus the previous period (20))")
    titles = [p["title"] for p in out["points"]]
    assert "Main worry: Debt and leverage" in titles
    assert "Main strength: Revenue and growth" in titles
    assert "Main worry: Management" not in titles            # one mention is not a finding
    assert "Mostly news, not opinion" in titles
    assert out["kpis"]["posts_previous"] == 20


def platform(name, posts, pos, neg, synthetic=0):
    return {"platform": name, "posts": posts, "positive_pct": 100 * pos / posts,
            "negative_pct": 100 * neg / posts, "synthetic_posts": synthetic}


def test_briefing_labels_synthetic_platforms_and_avoids_self_contradiction():
    platforms = [platform("reddit", 5, 1, 4, synthetic=5), platform("bluesky", 20, 10, 2)]
    out = briefing.compose("VEON", overview(25, 11, 8, 6), overview(0, 0, 0, 0), [], [], [], platforms, [])
    detail = next(p["detail"] for p in out["points"] if p["title"] == "Platforms differ")
    assert detail.startswith("Reddit (synthetic demo data) is the most negative")

    same = [platform("bluesky", 20, 10, 8), platform("youtube", 10, 2, 1)]
    out = briefing.compose("VEON", overview(30, 12, 9, 9), overview(0, 0, 0, 0), [], [], [], same, [])
    detail = next(p["detail"] for p in out["points"] if p["title"] == "Platforms differ")
    assert "both the most negative" in detail


def test_opinion_mode_does_not_say_mostly_news():
    types = [{"post_type": "news", "posts": 30}, {"post_type": "opinion", "posts": 10}]
    out = briefing.compose("VEON", overview(10, 5, 3, 2), overview(0, 0, 0, 0), [], [], [], [], types,
                           opinions_only=True)
    assert "Mostly news, not opinion" not in [p["title"] for p in out["points"]]


def test_briefing_with_no_posts_says_so():
    out = briefing.compose("STB", overview(0, 0, 0, 0), overview(0, 0, 0, 0), [], [], [], [], [])
    assert "No relevant posts" in out["headline"] and out["points"] == []
