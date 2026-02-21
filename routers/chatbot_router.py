"""API router for chatbot interactions."""
from __future__ import annotations

from fastapi import APIRouter

from controllers import handle_chatbot_request
from utils.schemas import ChatbotRequest, ChatbotResponse

router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])


@router.post("", response_model=ChatbotResponse, summary="Query the enterprise chatbot")
async def chatbot_endpoint(payload: ChatbotRequest) -> ChatbotResponse:
    """Return an answer plus citations for the supplied query."""

    return handle_chatbot_request(payload)
