"""Embeddings: a small local model, one vector per post (docs/blueprint.md §9.2).

BAAI/bge-small-en-v1.5 runs through fastembed (ONNX Runtime) rather than
sentence-transformers. The vectors are the same (cosine 1.00 on a check of both), but
the serving path needs no PyTorch: the deployed app stays small enough for free hosting
and the model loads in seconds.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from config.settings import EMBEDDING_MODEL

# BGE models are trained with this prefix on the query side only.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_CHARS = 2000


@lru_cache
def _model():
    from fastembed import TextEmbedding
    return TextEmbedding(EMBEDDING_MODEL)


def _unit(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


def embed_documents(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Unit-length vectors, so cosine similarity is a dot product."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    vectors = np.array(list(_model().embed([t[:MAX_CHARS] for t in texts], batch_size=batch_size)))
    return _unit(vectors)


def embed_query(text: str) -> np.ndarray:
    return _unit(np.array(next(iter(_model().embed([QUERY_PREFIX + text])))))
