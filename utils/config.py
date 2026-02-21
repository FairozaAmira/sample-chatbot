"""Configuration utilities for the enterprise chatbot project."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    app_name: str = "Enterprise RAG Chatbot"
    app_env: str = Field(default="development", description="Application environment.")
    # Placeholder API key for demonstration purposes.
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for the primary language model provider.",
    )
    llm_model: str = Field(default="gpt5-mini", description="Default LLM model identifier.")
    llm_provider: str = Field(
        default="placeholder",
        description="LLM provider to use: placeholder, openai, or ollama.",
    )
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for a local Ollama server.",
    )
    llm_cost_per_1k_tokens: float = Field(
        default=0.0,
        description="Estimated USD cost per 1,000 tokens for the configured LLM.",
    )

    enable_deepeval: bool = Field(
        default=True,
        description="Toggle to enable deepeval-based metrics when configuration allows.",
    )

    chroma_persist_directory: Path = Field(
        default=Path(".chroma_store"),
        description="Directory where Chroma vector store is persisted.",
    )
    data_directory: Path = Field(
        default=Path("data"),
        description="Root directory for local document ingestion.",
    )
    qa_cache_directory: Path = Field(
        default=Path("generated_qa"),
        description="Directory where generated Q&A pairs are stored.",
    )

    class Config:
        env_prefix = "CHATBOT_"
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Return a cached instance of application settings."""

    settings = Settings()
    settings.chroma_persist_directory.mkdir(parents=True, exist_ok=True)
    settings.data_directory.mkdir(parents=True, exist_ok=True)
    settings.qa_cache_directory.mkdir(parents=True, exist_ok=True)
    return settings
