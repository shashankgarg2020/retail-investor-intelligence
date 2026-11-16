"""Adapter tests against hand-written payloads in each platform's response shape.

No network: HTTP is served by httpx.MockTransport.
"""
from datetime import datetime, timezone

import httpx
import pytest

from config.settings import Settings, platform_config
from src.ingestion.bluesky import BlueskyAdapter
from src.ingestion.http import HttpClient, HttpError
from src.ingestion.reddit import RedditAdapter
from src.ingestion.youtube import YouTubeAdapter


def client(handler, **kwargs) -> HttpClient:
    return HttpClient(transport=httpx.MockTransport(handler), sleep=lambda _: None, **kwargs)


def adapter(cls, handler, **settings):
    return cls(Settings(_env_file=None, **settings), platform_config()[cls.platform], client(handler))


# ------------------------------------------------------------------ HTTP client

def test_retries_rate_limits_then_succeeds():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "1"}) if len(calls) < 3 else httpx.Response(200, json={"ok": True})

    http = client(handler)
    assert http.get_json("https://api.test/x") == {"ok": True}
    assert len(calls) == 3
    assert (http.requests_ok, http.requests_failed) == (1, 0)


def test_gives_up_after_max_retries():
    http = client(lambda r: httpx.Response(503), max_retries=2)
    with pytest.raises(HttpError) as err:
        http.get_json("https://api.test/x")
    assert err.value.status == 503
    assert http.requests_failed == 1


def test_does_not_retry_client_errors():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403, text="forbidden")

    with pytest.raises(HttpError):
        client(handler).get_json("https://api.test/x")
    assert len(calls) == 1


def test_retries_timeouts():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={})

    assert client(handler).get_json("https://api.test/x") == {}


def test_error_message_does_not_leak_query_string():
    http = client(lambda r: httpx.Response(400, text="bad"))
    with pytest.raises(HttpError) as err:
        http.get_json("https://api.test/x", params={"key": "SECRET"})
    assert "SECRET" not in str(err.value)


# ---------------------------------------------------------------------- Bluesky

BSKY_POST = {
    "uri": "at://did:plc:abc/app.bsky.feed.post/3kxyz",
    "author": {"did": "did:plc:abc", "handle": "investor.test"},
    "record": {"text": "$VEON earnings look   strong &amp; debt is falling", "createdAt": "2026-11-10T12:00:00.000Z",
               "langs": ["en"]},
    "likeCount": 4, "replyCount": 1, "repostCount": 2, "quoteCount": 1,
}


def test_bluesky_normalise():
    post = adapter(BlueskyAdapter, lambda r: httpx.Response(200)).normalise(BSKY_POST, "VEON")
    assert post.platform == "bluesky" and post.company_ticker == "VEON"
    assert post.external_id == BSKY_POST["uri"]
    assert post.text == "$VEON earnings look strong & debt is falling"
    assert post.url == "https://bsky.app/profile/did:plc:abc/post/3kxyz"
    assert (post.likes, post.comments, post.shares, post.views) == (4, 1, 3, None)
    assert post.created_at == datetime(2026, 11, 10, 12, tzinfo=timezone.utc)
    assert post.language == "en" and post.content_type == "post"


def test_bluesky_reply_is_a_comment_and_missing_counts_stay_none():
    raw = {**BSKY_POST, "record": {**BSKY_POST["record"], "reply": {"parent": {}}}}
    for key in ("likeCount", "replyCount", "repostCount", "quoteCount"):
        raw.pop(key)
    post = adapter(BlueskyAdapter, lambda r: httpx.Response(200)).normalise(raw, "VEON")
    assert post.content_type == "comment"
    assert (post.likes, post.comments, post.shares) == (None, None, None)


def test_bluesky_pagination_cursor_and_since():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        cursor = request.url.params.get("cursor")
        return httpx.Response(200, json={"posts": [BSKY_POST], "cursor": None if cursor else "page2"})

    a = adapter(BlueskyAdapter, handler)
    since = datetime(2026, 11, 1, tzinfo=timezone.utc)
    first = a.fetch_posts("VEON stock", None, since)
    assert a.get_next_page(first) == "page2"
    second = a.fetch_posts("VEON stock", "page2", since)
    assert a.get_next_page(second) is None
    assert seen[0]["since"] == "2026-11-01T00:00:00Z" and seen[1]["cursor"] == "page2"


def test_bluesky_uses_session_when_credentials_present():
    paths = []

    def handler(request):
        paths.append((request.url.host, request.url.path, request.headers.get("authorization")))
        if request.url.path.endswith("createSession"):
            return httpx.Response(200, json={"accessJwt": "jwt"})
        return httpx.Response(200, json={"posts": []})

    a = adapter(BlueskyAdapter, handler, bluesky_handle="me.test", bluesky_app_password="pw")
    a.fetch_posts("q", None, None)
    assert paths[0][1].endswith("createSession")
    assert paths[1] == ("bsky.social", "/xrpc/app.bsky.feed.searchPosts", "Bearer jwt")


# ----------------------------------------------------------------------- Reddit

REDDIT_POST = {"_kind": "post", "id": "abc1", "title": "Is $TTWO overvalued before GTA VI?",
               "selftext": "Looking at the P/E...", "author": "user1", "created_utc": 1794000000,
               "permalink": "/r/stocks/comments/abc1/x/", "score": 42, "num_comments": 7, "subreddit": "stocks"}


def test_reddit_normalise_post_and_comment():
    a = adapter(RedditAdapter, lambda r: httpx.Response(200))
    post = a.normalise(REDDIT_POST, "TTWO")
    assert post.external_id == "t3_abc1" and post.content_type == "post"
    assert post.title == "Is $TTWO overvalued before GTA VI?" and post.text == "Looking at the P/E..."
    assert (post.likes, post.comments, post.shares, post.views) == (42, 7, None, None)
    assert post.url == "https://www.reddit.com/r/stocks/comments/abc1/x/"

    comment = a.normalise({"_kind": "comment", "_parent_id": "abc1", "_parent_title": REDDIT_POST["title"],
                           "id": "c9", "body": "Bookings guidance was fine.", "author": "[deleted]",
                           "created_utc": 1794000100, "permalink": "/r/stocks/comments/abc1/x/c9/", "score": 3}, "TTWO")
    assert comment.content_type == "comment" and comment.parent_external_id == "t3_abc1"
    assert comment.parent_title == REDDIT_POST["title"] and comment.author_id is None


def test_reddit_skips_removed_content_and_uses_title_for_link_posts():
    a = adapter(RedditAdapter, lambda r: httpx.Response(200))
    assert a.normalise({"_kind": "comment", "_parent_id": "x", "id": "c", "body": "[removed]",
                        "created_utc": 1, "permalink": "/c"}, "TTWO") is None
    link = a.normalise({**REDDIT_POST, "selftext": ""}, "TTWO")
    assert link.text == REDDIT_POST["title"]


def test_reddit_fetch_gets_token_then_searches_and_reads_comments():
    def handler(request):
        if request.url.path == "/api/v1/access_token":
            return httpx.Response(200, json={"access_token": "tok"})
        assert request.headers["authorization"] == "Bearer tok"
        if request.url.path == "/search":
            return httpx.Response(200, json={"data": {"after": "t3_next", "children": [
                {"kind": "t3", "data": {k: v for k, v in REDDIT_POST.items() if k != "_kind"}}]}})
        return httpx.Response(200, json=[{}, {"data": {"children": [
            {"kind": "t1", "data": {"id": "c1", "body": "hi there", "created_utc": 1794000100,
                                    "permalink": "/c1", "score": 1}},
            {"kind": "more", "data": {}}]}}])

    a = adapter(RedditAdapter, handler, reddit_client_id="id", reddit_client_secret="s", reddit_user_agent="ua")
    page = a.fetch_posts("TTWO stock", None, None)
    assert [i["_kind"] for i in page.items] == ["post", "comment"]
    assert a.get_next_page(page) == "t3_next"


def test_adapters_report_missing_credentials():
    assert not adapter(RedditAdapter, lambda r: httpx.Response(200)).is_configured()
    assert not adapter(YouTubeAdapter, lambda r: httpx.Response(200)).is_configured()
    assert adapter(BlueskyAdapter, lambda r: httpx.Response(200)).is_configured()


# ---------------------------------------------------------------------- YouTube

YT_VIDEO = {"_kind": "video", "id": "vid1",
            "snippet": {"title": "VEON Q3 earnings review", "description": "Revenue up, debt down.",
                        "publishedAt": "2026-11-09T08:00:00Z", "channelId": "ch1", "channelTitle": "Fin Chan"},
            "statistics": {"viewCount": "1500", "commentCount": "12"}}  # likes hidden by the creator


def test_youtube_normalise_video_keeps_hidden_likes_as_none():
    post = adapter(YouTubeAdapter, lambda r: httpx.Response(200)).normalise(YT_VIDEO, "VEON")
    assert post.content_type == "video" and post.external_id == "vid1"
    assert (post.likes, post.comments, post.views) == (None, 12, 1500)
    assert post.url == "https://www.youtube.com/watch?v=vid1"


def test_youtube_normalise_comment_links_to_parent():
    raw = {"_kind": "comment", "_parent_id": "vid1", "_parent_title": "VEON Q3 earnings review",
           "snippet": {"totalReplyCount": 2, "topLevelComment": {"id": "cm1", "snippet": {
               "textDisplay": "Debt is still too high", "likeCount": 5, "publishedAt": "2026-11-09T09:00:00Z",
               "authorChannelId": {"value": "ch9"}}}}}
    post = adapter(YouTubeAdapter, lambda r: httpx.Response(200)).normalise(raw, "VEON")
    assert post.content_type == "comment" and post.parent_external_id == "vid1"
    assert post.url.endswith("watch?v=vid1&lc=cm1") and (post.likes, post.comments) == (5, 2)


def test_youtube_fetch_survives_disabled_comments():
    def handler(request):
        path = request.url.path
        if path.endswith("/search"):
            return httpx.Response(200, json={"items": [{"id": {"videoId": "vid1"}}], "nextPageToken": "tok2"})
        if path.endswith("/videos"):
            return httpx.Response(200, json={"items": [{k: v for k, v in YT_VIDEO.items() if k != "_kind"}]})
        return httpx.Response(403, json={"error": {"errors": [{"reason": "commentsDisabled"}]}})

    a = adapter(YouTubeAdapter, handler, youtube_api_key="k")
    page = a.fetch_posts("VEON earnings", None, None)
    assert [i["_kind"] for i in page.items] == ["video"]
    assert a.get_next_page(page) == "tok2"
