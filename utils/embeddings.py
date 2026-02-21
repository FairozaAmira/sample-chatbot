"""Embedding utilities for the RAG system."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Iterable, List

from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class FallbackEmbeddings(Embeddings):
    """Deterministic embedding generator that avoids external downloads."""

    def _embed(self, text: str) -> List[float]:
        vector = [0.0] * 32
        for index, char in enumerate(text.encode("utf-8")):
            vector[index % len(vector)] += (char % 31) / 100.0
        return vector

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:  # type: ignore[override]
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        return self._embed(text)


@lru_cache()
def get_embeddings() -> Embeddings:
    """Return a cached embedding model instance."""

    try:
        return HuggingFaceEmbeddings(model_name=DEFAULT_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - optional runtime failure path
        LOGGER.warning("Falling back to deterministic embeddings due to error: %s", exc)
        return FallbackEmbeddings()
