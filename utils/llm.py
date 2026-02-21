"""LLM utilities including a placeholder implementation for offline development."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from .config import get_settings

LOGGER = logging.getLogger(__name__)


class PlaceholderLLM(LLM):
    """A minimalistic LLM stub used when no external model is configured."""

    model_name: str = "placeholder-gpt5-mini"

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "placeholder"

    def _call(  # type: ignore[override]
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop:
            LOGGER.debug("Stop tokens provided to PlaceholderLLM but are ignored: %s", stop)
        LOGGER.info("Generating placeholder response for prompt of length %s", len(prompt))
        return (
            "[Placeholder LLM Response]\n"
            "This environment is configured without external LLM access.\n"
            "Prompt summary: "
            f"{prompt[:200]}{'...' if len(prompt) > 200 else ''}"
        )


def get_llm() -> LLM:
    """Return an LLM instance based on configuration."""

    settings = get_settings()
    if settings.llm_api_key:
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.llm_model,
                openai_api_key=settings.llm_api_key,
                temperature=0.3,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            LOGGER.warning(
                "langchain-openai is not installed; falling back to placeholder LLM: %s",
                exc,
            )
    return PlaceholderLLM()
