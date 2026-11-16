"""LLM provider abstraction. The rest of the code only knows `structured()`.

Swapping provider means writing another class with the same method.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from config.settings import get_settings

T = TypeVar("T", bound=BaseModel)

# Server-side refusal fallback: if a safety classifier declines a request, the API
# re-runs it on a fallback model inside the same call instead of returning nothing.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# USD per million tokens (input, output). Used only to report measured cost.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class LLMUnavailable(Exception):
    """No credentials configured. Callers fall back to non-LLM behaviour."""


class LLMError(Exception):
    """The call was made but produced nothing usable."""


@dataclass
class LLMResult:
    output: BaseModel
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float

    @property
    def cost_usd(self) -> float | None:
        price = PRICES.get(self.model)
        if price is None:
            return None
        return (self.input_tokens * price[0] + self.output_tokens * price[1]) / 1_000_000


class ClaudeClient:
    def __init__(self, api_key: str, model: str, fast_model: str):
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.fast_model = fast_model

    def structured(self, *, system: str, user: str, schema: type[T], fast: bool = False,
                   max_tokens: int = 8000, effort: str = "medium") -> LLMResult:
        """One request, answer validated against `schema` by the SDK."""
        model = self.fast_model if fast else self.model
        started = time.perf_counter()
        common = dict(model=model, max_tokens=max_tokens, system=system,
                      messages=[{"role": "user", "content": user}],
                      output_format=schema, output_config={"effort": effort})
        try:
            if fast:
                response = self._client.messages.parse(**common)
            else:
                response = self._client.beta.messages.parse(
                    **common, betas=[FALLBACK_BETA], fallbacks="default")
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable(f"Anthropic API key rejected: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Rate limited by the Anthropic API; try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Could not reach the Anthropic API.") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise LLMError(f"No valid structured output (stop_reason={response.stop_reason}).")
        return LLMResult(
            output=response.parsed_output, model=response.model,
            input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
            seconds=time.perf_counter() - started,
        )


@lru_cache
def get_llm() -> ClaudeClient:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set.")
    return ClaudeClient(settings.anthropic_api_key, settings.llm_model, settings.llm_fast_model)
