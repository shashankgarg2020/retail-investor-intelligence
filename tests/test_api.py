"""API contract tests: routing, validation and response shape.

The database is replaced: `get_conn` yields a dummy and the metric functions return
canned rows, so these tests check the HTTP layer, not SQL (SQL is exercised by
`scripts/evaluate.py latency` against the real database).
"""
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.analytics import briefing, insights, metrics
from src.api import deps
from src.api.main import app
from src.api.routes import chat as chat_route
from src.rag.response_schema import ChatResponse

OVERVIEW = {"posts": 40, "positive": 12, "neutral": 20, "negative": 8, "positive_pct": 30.0,
            "neutral_pct": 50.0, "negative_pct": 20.0, "post_weighted_sentiment": 0.1,
            "engagement_weighted_sentiment": 0.2, "synthetic_posts": 0, "small_sample": False}
calls: dict = {}


@pytest.fixture(autouse=True)
def fake_database(monkeypatch):
    calls.clear()

    def record(name, value):
        def fn(conn, f, *args, **kwargs):
            calls[name] = f
            return value
        return fn

    app.dependency_overrides[deps.get_conn] = lambda: iter([object()])
    monkeypatch.setattr(metrics, "overview", record("overview", OVERVIEW))
    monkeypatch.setattr(metrics, "by_platform", record("by_platform", []))
    monkeypatch.setattr(metrics, "sentiment_trend", record("trend", []))
    monkeypatch.setattr(metrics, "topics", record("topics", []))
    monkeypatch.setattr(metrics, "aspects", record("aspects", []))
    monkeypatch.setattr(metrics, "post_types", record("post_types", [{"post_type": "opinion", "posts": 40}]))
    monkeypatch.setattr(metrics, "coverage", lambda conn, t, s, e: {"collected": 50})
    monkeypatch.setattr(metrics, "freshness", lambda conn, t: [])
    monkeypatch.setattr(insights, "emerging_concerns", record("concerns", []))
    yield
    app.dependency_overrides.clear()


client = TestClient(app)


def test_overview_shape_and_filters():
    r = client.get("/analytics/veon/overview", params={"start": "2026-10-01", "end": "2026-10-31",
                                                      "platform": "bluesky", "opinions_only": "true"})
    assert r.status_code == 200
    body = r.json()
    assert {"overview", "by_platform", "trend", "coverage", "freshness", "emerging_concerns",
            "post_types"} <= set(body)
    f = calls["overview"]
    assert f.ticker == "VEON"                                   # ticker is normalised
    assert f.platform == "bluesky" and f.opinions_only is True
    # Inclusive end date becomes an exclusive bound at the next midnight UTC.
    assert f.end == datetime(2026, 11, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("path,params,status", [
    ("/analytics/NOPE/overview", {}, 404),
    ("/analytics/VEON/overview", {"platform": "myspace"}, 422),
    ("/analytics/VEON/overview", {"start": "2026-10-10", "end": "2026-10-01"}, 422),
    ("/analytics/VEON/overview", {"sentiment": "angry"}, 422),
])
def test_bad_input_is_rejected(path, params, status):
    assert client.get(path, params=params).status_code == status


def test_briefing_endpoint_returns_sentences():
    body = client.get("/analytics/TTWO/briefing").json()
    assert body["headline"].startswith("40 relevant posts about Take-Two")
    assert body["tone"] == briefing.tone(OVERVIEW)
    assert "kpis" in body and isinstance(body["points"], list)


def test_aspects_endpoint():
    body = client.get("/analytics/STB/aspects").json()
    assert body == {"aspects": [], "overview": OVERVIEW}


def test_chat_validates_and_calls_engine(monkeypatch):
    captured = {}

    def fake_answer(conn, question, ticker, start, end, platform, embed):
        captured.update(question=question, ticker=ticker, platform=platform)
        return ChatResponse(answer="ok", themes=[], evidence=[], insufficient_evidence=False, intent="SUMMARY",
                            scope={}, posts_analysed=40, platform_breakdown=[], metrics={}, limitations=[],
                            generated_by="test", retrieved_count=0, dropped_citations=0, planner_notes=[],
                            timings_seconds={})

    monkeypatch.setattr(chat_route.query_engine, "answer", fake_answer)
    r = client.post("/chat", json={"question": "What are investors worried about?", "ticker": "veon"})
    assert r.status_code == 200 and r.json()["answer"] == "ok"
    assert captured["ticker"] == "VEON"
    assert client.post("/chat", json={"question": "hi", "ticker": "VEON"}).status_code == 422       # too short
    assert client.post("/chat", json={"question": "Any news?", "ticker": "VEON",
                                      "platform": "myspace"}).status_code == 422


def test_quality_endpoint_without_results(monkeypatch, tmp_path):
    import src.api.main as main
    monkeypatch.setattr(main, "EVAL_RESULTS", tmp_path / "missing.json")
    assert client.get("/quality").json() == {"available": False}


# ---------------------------------------------------------------------- refresh

class FakeProcess:
    def __init__(self, *args, **kwargs):
        self.returncode = None
        FakeProcess.started += 1

    def poll(self):
        return self.returncode


@pytest.fixture
def refresh_env(monkeypatch, tmp_path):
    from config.settings import get_settings
    from src.api.routes import refresh
    FakeProcess.started = 0
    monkeypatch.setattr(refresh.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(refresh, "LOG_FILE", tmp_path / "refresh.log")
    monkeypatch.setattr(refresh, "data_times", lambda conn, since: {
        "last_collected": None, "last_processed": "2026-11-16T10:00:00+00:00", "new_posts": 3 if since else 0})
    monkeypatch.setattr(refresh, "_process", None)
    monkeypatch.setattr(refresh, "_run", {"state": "idle", "started_at": None, "finished_at": None})
    settings = get_settings()
    monkeypatch.setattr(settings, "refresh_token", "")
    monkeypatch.setattr(settings, "refresh_cooldown_min", 15.0)
    return refresh, settings


def test_refresh_runs_once_then_cools_down(refresh_env):
    refresh, _ = refresh_env
    idle = client.get("/refresh/status").json()
    assert idle["state"] == "idle" and idle["new_posts"] is None and idle["last_updated"]

    started = client.post("/refresh")
    assert started.status_code == 202 and started.json()["state"] == "running"
    assert client.post("/refresh").status_code == 409            # one run at a time
    assert FakeProcess.started == 1

    refresh._process.returncode = 0                              # the subprocess finishes
    done = client.get("/refresh/status").json()
    assert done["state"] == "succeeded" and done["new_posts"] == 3
    assert done["cooldown_seconds_left"] > 0
    assert client.post("/refresh").status_code == 429            # cooldown protects quotas


def test_failed_refresh_reports_last_log_line(refresh_env):
    refresh, _ = refresh_env
    client.post("/refresh")
    refresh.LOG_FILE.write_text("collecting\nERROR youtube quota exceeded\n", encoding="utf-8")
    refresh._process.returncode = 1
    body = client.get("/refresh/status").json()
    assert body["state"] == "failed" and body["message"] == "ERROR youtube quota exceeded"


def test_refresh_token_is_enforced_when_set(refresh_env):
    _, settings = refresh_env
    settings.refresh_token = "s3cret"
    assert client.post("/refresh").status_code == 403
    assert client.post("/refresh", headers={"X-Refresh-Token": "wrong"}).status_code == 403
    assert client.post("/refresh", headers={"X-Refresh-Token": "s3cret"}).status_code == 202
