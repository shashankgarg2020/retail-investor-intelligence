from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.database.models import CanonicalPost
from src.processing.deduplication import find_duplicates
from src.processing.normalizer import clean_text, parse_timestamp, text_hash, to_int


def test_clean_text_decodes_entities_and_collapses_whitespace():
    assert clean_text("VEON &amp; Kyivstar\n\n  results​ ") == "VEON & Kyivstar results"
    assert clean_text(None) == ""


def test_text_hash_ignores_case_links_and_punctuation():
    a = text_hash("VEON beats expectations! https://example.com/a?utm=1")
    b = text_hash("veon beats   expectations https://other.example/b")
    assert a == b
    assert a != text_hash("VEON misses expectations")


@pytest.mark.parametrize("value", [1763164800, "2025-11-15T00:00:00Z", "2025-11-15T00:00:00.000+00:00"])
def test_parse_timestamp_returns_aware_utc(value):
    assert parse_timestamp(value) == datetime(2025, 11, 15, tzinfo=timezone.utc)


def test_parse_timestamp_converts_offsets_to_utc():
    assert parse_timestamp("2025-11-15T05:30:00+05:30") == datetime(2025, 11, 15, tzinfo=timezone.utc)


def test_missing_engagement_stays_none_not_zero():
    assert to_int(None) is None
    assert to_int("") is None
    assert to_int("12") == 12
    assert to_int(0) == 0


def _post(**overrides):
    base = dict(platform="reddit", external_id="t3_x", company_ticker="VEON", content_type="post",
                text="hello", text_hash="h", created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
                url="https://example.com")
    return CanonicalPost(**{**base, **overrides})


def test_canonical_post_defaults_engagement_to_none():
    post = _post()
    assert (post.likes, post.comments, post.shares, post.views) == (None, None, None, None)
    assert post.is_synthetic is False


def test_canonical_post_rejects_naive_timestamps_and_unknown_platforms():
    with pytest.raises(ValidationError):
        _post(created_at=datetime(2025, 1, 1))
    with pytest.raises(ValidationError):
        _post(platform="twitter")
    with pytest.raises(ValidationError):
        _post(text="")


LONG = "VEON reported strong quarterly results and raised its full year guidance today"


def test_duplicates_point_at_the_earliest_post():
    posts = [{"id": 1, "text": LONG, "text_hash": "a"}, {"id": 2, "text": LONG, "text_hash": "a"},
             {"id": 3, "text": LONG + " again", "text_hash": "b"}]
    assert find_duplicates(posts) == {2: 1}


def test_short_identical_posts_are_separate_opinions():
    posts = [{"id": 1, "text": "bullish", "text_hash": "a"}, {"id": 2, "text": "bullish", "text_hash": "a"}]
    assert find_duplicates(posts) == {}


def test_duplicates_respect_previously_stored_hashes():
    assert find_duplicates([{"id": 9, "text": LONG, "text_hash": "a"}], known={"a": 4}) == {9: 4}
