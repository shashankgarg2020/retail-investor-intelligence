"""Reddit adapter: official Data API with application-only OAuth."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from src.database.models import CanonicalPost
from src.ingestion.base import BaseAdapter, Health, Page
from src.ingestion.http import HttpError
from src.processing.normalizer import clean_text, parse_timestamp, text_hash, to_int

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API = "https://oauth.reddit.com"
_GONE = {"[removed]", "[deleted]", ""}


class RedditAdapter(BaseAdapter):
    platform = "reddit"
    _token: str | None = None

    def is_configured(self) -> bool:
        s = self.settings
        return bool(s.reddit_client_id and s.reddit_client_secret and s.reddit_user_agent)

    def _headers(self) -> dict[str, str]:
        if self._token is None:
            data = self.http.post_json(
                TOKEN_URL,
                auth=(self.settings.reddit_client_id, self.settings.reddit_client_secret),
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": self.settings.reddit_user_agent},
            )
            self._token = data["access_token"]
        return {"Authorization": f"Bearer {self._token}", "User-Agent": self.settings.reddit_user_agent}

    def health_check(self) -> Health:
        try:
            self.http.get_json(f"{API}/search", headers=self._headers(),
                               params={"q": "test", "limit": 1, "raw_json": 1})
            return Health(True, "search endpoint reachable")
        except (HttpError, KeyError) as exc:
            return Health(False, str(exc))

    def fetch_posts(self, query: str, cursor: str | None, since: datetime | None) -> Page:
        params = {"q": query, "sort": "new", "type": "link", "t": "year",
                  "limit": self.config["page_size"], "raw_json": 1}
        if cursor:
            params["after"] = cursor
        listing = self.http.get_json(f"{API}/search", headers=self._headers(), params=params)["data"]
        submissions = [child["data"] for child in listing["children"] if child.get("kind") == "t3"]
        items = [{"_kind": "post", **s} for s in submissions]

        # Comments carry much of the discussion. Fetch them for the most-commented posts only.
        busiest = sorted(submissions, key=lambda s: s.get("num_comments") or 0, reverse=True)
        for sub in busiest[: self.config["comments_for_top_n_posts"]]:
            if not sub.get("num_comments"):
                continue
            try:
                items.extend(self._fetch_comments(sub))
            except HttpError:
                continue  # one unavailable thread must not lose the page
        return Page(items=items, next_cursor=listing.get("after"))

    def _fetch_comments(self, submission: dict[str, Any]) -> list[dict[str, Any]]:
        data = self.http.get_json(
            f"{API}/comments/{submission['id']}", headers=self._headers(),
            params={"limit": self.config["comments_per_post"], "depth": 1, "sort": "top", "raw_json": 1},
        )
        return [
            {"_kind": "comment", "_parent_id": submission["id"], "_parent_title": submission.get("title"),
             **child["data"]}
            for child in data[1]["data"]["children"] if child.get("kind") == "t1"
        ]

    def normalise(self, raw: dict[str, Any], company_ticker: str) -> CanonicalPost | None:
        author = raw.get("author")
        author = None if author in (None, "[deleted]") else author
        common = dict(
            platform="reddit", company_ticker=company_ticker, author_id=author,
            created_at=parse_timestamp(raw["created_utc"]),
            url="https://www.reddit.com" + raw["permalink"],
            likes=to_int(raw.get("score")),  # net upvotes
            extra={"subreddit": raw.get("subreddit")},
        )
        if raw["_kind"] == "comment":
            body = clean_text(raw.get("body"))
            if body in _GONE:
                return None
            return CanonicalPost(
                external_id=f"t1_{raw['id']}", content_type="comment", text=body,
                text_hash=text_hash(body), parent_external_id=f"t3_{raw['_parent_id']}",
                parent_title=clean_text(raw.get("_parent_title")) or None, **common,
            )
        title = clean_text(raw.get("title"))
        body = clean_text(raw.get("selftext"))
        if body in _GONE:
            body = ""
        text = body or title  # link posts have no body; the title is the content
        if not text:
            return None
        return CanonicalPost(
            external_id=f"t3_{raw['id']}", content_type="post", title=title or None, text=text,
            text_hash=text_hash(f"{title} {body}"), comments=to_int(raw.get("num_comments")), **common,
        )
