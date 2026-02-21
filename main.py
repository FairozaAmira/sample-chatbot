"""FastAPI application entrypoint for the enterprise RAG chatbot."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

try:
    import chromadb.telemetry.product.posthog as _chroma_ph_module

    _chroma_ph_module.posthog.capture = lambda *_a, **_kw: None  # type: ignore[attr-defined]
except Exception:
    pass

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from routers import chatbot_router, crawler_router, evaluation_router
from utils.config import Settings, get_settings

LOGGER = logging.getLogger(__name__)


@lru_cache()
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        summary="Enterprise-ready Retrieval-Augmented Generation chatbot platform.",
    )

    app.include_router(chatbot_router)
    app.include_router(crawler_router)
    app.include_router(evaluation_router)

    @app.get("/health", tags=["health"])
    async def health_check() -> JSONResponse:
        return JSONResponse({"status": "ok", "app": settings.app_name})

    return app


app = create_app()
