"""Sentiment: FinBERT, a finance-oriented pretrained classifier (docs/blueprint.md §8.1).

The score returned is the model's softmax output. It is not a calibrated
probability and is never described as one.
"""
from __future__ import annotations

from functools import lru_cache

from config.settings import SENTIMENT_MODEL

LABELS = ("positive", "neutral", "negative")
MAX_CHARS = 2000  # the tokenizer truncates to 512 tokens anyway; this just bounds work


def available() -> bool:
    """FinBERT needs transformers and PyTorch (requirements-ml.txt); the cloud build omits them."""
    from importlib.util import find_spec
    return find_spec("transformers") is not None and find_spec("torch") is not None


@lru_cache
def _classifier():
    # Imported lazily: torch is slow to load and the API process never needs it.
    from transformers import pipeline
    return pipeline("text-classification", model=SENTIMENT_MODEL, device=-1)


def classify(texts: list[str], batch_size: int = 16) -> list[tuple[str, float]]:
    """Return (label, score) for each text, in order."""
    if not texts:
        return []
    outputs = _classifier()([t[:MAX_CHARS] for t in texts], batch_size=batch_size,
                            truncation=True, max_length=512)
    return [(o["label"].lower(), float(o["score"])) for o in outputs]
