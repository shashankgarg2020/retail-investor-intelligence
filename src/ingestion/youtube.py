"""YouTube adapter: Data API v3. Videos (title + description) and top-level comments.

Quota: search.list costs 100 units, videos.list and commentThreads.list cost 1 each.
The free daily quota is 10,000 units, so pages are kept small.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.database.models import CanonicalPost
from src.ingestion.base import BaseAdapter, Health, Page
from src.ingestion.http import HttpError
from src.processing.normalizer import clean_text, parse_timestamp, text_hash, to_int

API = "https://www.googleapis.com/youtube/v3"
MAX_DESCRIPTION_CHARS = 2000


class YouTubeAdapter(BaseAdapter):
    platform = "youtube"

    def is_configured(self) -> bool:
        return bool(self.settings.youtube_api_key)

    def _get(self, resource: str, **params: Any) -> dict[str, Any]:
        return self.http.get_json(f"{API}/{resource}", params={**params, "key": self.settings.youtube_api_key})

    def health_check(self) -> Health:
        try:
            self._get("videos", part="id", chart="mostPopular", maxResults=1)  # 1 unit
            return Health(True, "videos endpoint reachable")
        except HttpError as exc:
            return Health(False, str(exc))

    def fetch_posts(self, query: str, cursor: str | None, since: datetime | None) -> Page:
        params: dict[str, Any] = dict(part="id", q=query, type="video", order="date",
                                      maxResults=self.config["page_size"], relevanceLanguage="en")
        if since:
            params["publishedAfter"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        if cursor:
            params["pageToken"] = cursor
        search = self._get("search", **params)
        video_ids = [item["id"]["videoId"] for item in search.get("items", [])]
        if not video_ids:
            return Page(next_cursor=None)

        # search.list has no statistics and truncates descriptions; videos.list has both.
        videos = self._get("videos", part="snippet,statistics", id=",".join(video_ids)).get("items", [])
        items: list[dict[str, Any]] = [{"_kind": "video", **v} for v in videos]

        def comment_count(v: dict[str, Any]) -> int:
            return to_int(v.get("statistics", {}).get("commentCount")) or 0

        for video in sorted(videos, key=comment_count, reverse=True)[: self.config["comment_videos_per_page"]]:
            if comment_count(video) == 0:
                continue
            try:
                threads = self._get("commentThreads", part="snippet", videoId=video["id"],
                                    maxResults=self.config["comments_per_video"],
                                    order="relevance", textFormat="plainText")
            except HttpError:
                continue  # comments disabled on this video
            items.extend(
                {"_kind": "comment", "_parent_id": video["id"],
                 "_parent_title": video["snippet"].get("title"), **t}
                for t in threads.get("items", [])
            )
        return Page(items=items, next_cursor=search.get("nextPageToken"))

    def normalise(self, raw: dict[str, Any], company_ticker: str) -> CanonicalPost | None:
        if raw["_kind"] == "comment":
            top = raw["snippet"]["topLevelComment"]
            snippet = top["snippet"]
            body = clean_text(snippet.get("textDisplay"))
            if not body:
                return None
            return CanonicalPost(
                platform="youtube", company_ticker=company_ticker, external_id=top["id"],
                content_type="comment", text=body, text_hash=text_hash(body),
                author_id=(snippet.get("authorChannelId") or {}).get("value"),
                parent_external_id=raw["_parent_id"],
                parent_title=clean_text(raw.get("_parent_title")) or None,
                created_at=parse_timestamp(snippet["publishedAt"]),
                url=f"https://www.youtube.com/watch?v={raw['_parent_id']}&lc={top['id']}",
                likes=to_int(snippet.get("likeCount")),
                comments=to_int(raw["snippet"].get("totalReplyCount")),
            )
        snippet, stats = raw["snippet"], raw.get("statistics", {})
        title = clean_text(snippet.get("title"))
        description = clean_text(snippet.get("description"))[:MAX_DESCRIPTION_CHARS]
        text = description or title
        if not text:
            return None
        return CanonicalPost(
            platform="youtube", company_ticker=company_ticker, external_id=raw["id"],
            content_type="video", title=title or None, text=text,
            text_hash=text_hash(f"{title} {description}"),
            author_id=snippet.get("channelId"),
            created_at=parse_timestamp(snippet["publishedAt"]),
            url=f"https://www.youtube.com/watch?v={raw['id']}",
            # Creators can hide like counts; a missing key stays None.
            likes=to_int(stats.get("likeCount")), comments=to_int(stats.get("commentCount")),
            views=to_int(stats.get("viewCount")),
            language=(snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage") or "")[:2] or None,
            extra={"channel": snippet.get("channelTitle")},
        )
