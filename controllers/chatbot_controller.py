"""Controller for chatbot interactions."""
from __future__ import annotations

from utils.rag import RAGPipeline
from utils.schemas import ChatbotRequest, ChatbotResponse


def handle_chatbot_request(payload: ChatbotRequest) -> ChatbotResponse:
    """Process a chatbot query using the RAG pipeline."""

    pipeline = RAGPipeline()
    return pipeline.answer(query=payload.query, domain=payload.domain, top_k=payload.top_k)
