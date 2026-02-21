"""Vector store utilities for persistence and retrieval."""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path
from typing import Iterable, List

from langchain.docstore.document import Document

# Chroma requires sqlite >= 3.35. Some slim images ship older sqlite versions.
if sqlite3.sqlite_version_info < (3, 35, 0):
    import pysqlite3  # type: ignore[import-not-found]

    sys.modules["sqlite3"] = pysqlite3

from langchain_community.vectorstores import Chroma

from .config import get_settings
from .embeddings import get_embeddings

LOGGER = logging.getLogger(__name__)


def get_vector_store(persist_directory: Path | None = None) -> Chroma:
    """Return a Chroma vector store instance with persistent storage."""

    settings = get_settings()
    directory = persist_directory or settings.chroma_persist_directory
    directory.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Using Chroma persist directory: %s", directory)
    return Chroma(persist_directory=str(directory), embedding_function=get_embeddings())


def upsert_documents(documents: Iterable[Document]) -> int:
    """Upsert documents into the vector store and return the number of records."""

    docs: List[Document] = list(documents)
    if not docs:
        LOGGER.info("No documents provided for upsert.")
        return 0
    LOGGER.info("Upserting %d documents into Chroma store", len(docs))
    vector_store = get_vector_store()
    vector_store.add_documents(docs)
    return len(docs)
