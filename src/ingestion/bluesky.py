"""Bluesky adapter: AT Protocol post search.

Works without credentials through the public AppView. If a handle and app
password are configured it uses an authenticated session instead, which has
its own rate limits and does not depend on the public endpoint staying open.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.database.models import CanonicalPost
from src.ingestion.base import BaseAdapter, Health, Page
from src.ingestion.http import HttpError
from src.processing.normalizer import clean_text, parse_timestamp, text_hash, to_int

PUBLIC_API = "https://api.bsky.app/xrpc"
PDS = "https://bsky.social/xrpc"


class BlueskyAdapter(BaseAdapter):
    platform = "bluesky"
    _jwt: str | None = None

    def is_configured(self) -> bool:
        return True  # public search needs no credentials

    def _authenticated(self) -> bool:
        return bool(self.settings.bluesky_handle and self.settings.bluesky_app_password)

    def _search(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._authenticated():
            return self.http.get_json(f"{PUBLIC_API}/app.bsky.feed.searchPosts", params=params)
        if self._jwt is None:
            session = self.http.post_json(
                f"{PDS}/com.atproto.server.createSession",
                json={"identifier": self.settings.bluesky_handle,
                      "password": self.settings.bluesky_app_password},
            )
            self._jwt = session["accessJwt"]
        return self.http.get_json(f"{PDS}/app.bsky.feed.searchPosts", params=params,
                                  headers={"Authorization": f"Bearer {self._jwt}"})

    def health_check(self) -> Health:
        try:
            self._search({"q": "test", "limit": 1})
            mode = "authenticated" if self._authenticated() else "public, unauthenticated"
            return Health(True, f"search endpoint reachable ({mode})")
        except (HttpError, KeyError) as exc:
            return Health(False, str(exc))

    def fetch_posts(self, query: str, cursor: str | None, since: datetime | None) -> Page:
        params: dict[str, Any] = {"q": query, "limit": self.config["page_size"], "sort": "latest"}
        if since:
            params["since"] = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        if cursor:
            params["cursor"] = cursor
        data = self._search(params)
        return Page(items=data.get("posts", []), next_cursor=data.get("cursor"))

    def normalise(self, raw: dict[str, Any], company_ticker: str) -> CanonicalPost | None:
        record = raw.get("record", {})
        body = clean_text(record.get("text"))
        # Link cards often hold the headline while the post text is only a teaser.
        external = (record.get("embed") or {}).get("external") or {}
        card_title = clean_text(external.get("title")) or None
        if not body and not card_title:
            return None
        did = raw["author"]["did"]
        rkey = raw["uri"].rsplit("/", 1)[-1]
        reposts, quotes = to_int(raw.get("repostCount")), to_int(raw.get("quoteCount"))
        shares = None if reposts is None and quotes is None else (reposts or 0) + (quotes or 0)
        langs = record.get("langs") or []
        return CanonicalPost(
            platform="bluesky", company_ticker=company_ticker, external_id=raw["uri"],
            content_type="comment" if record.get("reply") else "post",
            title=card_title, text=body or card_title, text_hash=text_hash(body or card_title),
            author_id=did,
            created_at=parse_timestamp(record["createdAt"]),
            url=f"https://bsky.app/profile/{did}/post/{rkey}",
            likes=to_int(raw.get("likeCount")), comments=to_int(raw.get("replyCount")), shares=shares,
            language=langs[0][:2].lower() if langs else None,
            extra={"handle": raw["author"].get("handle")},
        )
