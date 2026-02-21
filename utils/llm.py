"""LLM utilities including a placeholder implementation for offline development."""
from __future__ import annotations

import logging
from typing import Any, List, Optional

import requests

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


class OllamaLLM(LLM):
    """Lightweight Ollama client for local open-source model inference."""

    model_name: str = "llama3.2"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 300.0

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "ollama"

    def _call(  # type: ignore[override]
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop:
            LOGGER.debug("Stop tokens provided to OllamaLLM but are ignored: %s", stop)
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        try:
            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            return str(data.get("response", "")).strip()
        except Exception as exc:
            LOGGER.warning("Ollama request failed (%s). Falling back to placeholder output.", exc)
            return (
                "[Ollama Unavailable - Placeholder Response]\n"
                "Could not reach local Ollama server.\n"
                "Prompt summary: "
                f"{prompt[:200]}{'...' if len(prompt) > 200 else ''}"
            )


def get_llm() -> LLM:
    """Return an LLM instance based on configuration."""

    settings = get_settings()
    provider = settings.llm_provider.lower().strip()

    if provider == "ollama":
        LOGGER.info("Using Ollama provider at %s with model %s", settings.ollama_base_url, settings.llm_model)
        return OllamaLLM(
            model_name=settings.llm_model,
            base_url=settings.ollama_base_url,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

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
    LOGGER.info("Using placeholder LLM (no external provider configured).")
    return PlaceholderLLM()
