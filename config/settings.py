"""Central settings: environment variables plus the YAML configuration files."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

PLATFORMS = ("reddit", "youtube", "bluesky")
SUPPORTED_LANGUAGES = ("en",)

SENTIMENT_MODEL = "ProsusAI/finbert"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    database_url: str = ""

    anthropic_api_key: str = ""
    llm_model: str = "claude-haiku-5-5"
    llm_fast_model: str = "claude-haiku-5-5"

    youtube_api_key: str = ""
    bluesky_handle: str = ""
    bluesky_app_password: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = ""

    api_base_url: str = "http://127.0.0.1:8000"

    # "Refresh data" button: optional shared secret, and minimum minutes between runs
    # (protects the YouTube quota and LLM spend on a public deployment).
    refresh_token: str = ""
    refresh_cooldown_min: float = 15.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache
def company_config() -> dict[str, Any]:
    """Full companies.yaml: {'companies': {...}, 'finance_terms': [...]}."""
    return _load_yaml("companies.yaml")


def companies() -> dict[str, dict[str, Any]]:
    return company_config()["companies"]


@lru_cache
def platform_config() -> dict[str, Any]:
    raw = _load_yaml("platform_config.yaml")
    defaults = raw["defaults"]
    return {p: {**defaults, **raw.get(p, {})} for p in PLATFORMS}


@lru_cache
def topic_config() -> dict[str, Any]:
    return _load_yaml("topics.yaml")


def topics_for(ticker: str) -> dict[str, dict[str, Any]]:
    """Shared topics plus the company's own topics, keyed by topic id."""
    cfg = topic_config()
    return {**cfg["shared"], **cfg["companies"].get(ticker, {})}
