"""API router for chatbot evaluation."""
from __future__ import annotations

from fastapi import APIRouter

from controllers import handle_evaluation_request
from utils.schemas import EvaluationRequest, EvaluationResponse

router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])


@router.post("", response_model=EvaluationResponse, summary="Evaluate chatbot answer quality")
async def evaluation_endpoint(payload: EvaluationRequest) -> EvaluationResponse:
    """Return evaluation metrics for a chatbot answer."""

    return handle_evaluation_request(payload)
