"""Controller handling chatbot evaluation requests."""
from __future__ import annotations

from datetime import datetime, timezone

from utils.config import get_settings
from utils.costs import estimate_cost
from utils.evaluation import EvaluationService
from utils.identifiers import generate_request_id
from utils.schemas import CostBreakdown, EvaluationRequest, EvaluationResponse
from utils.timing import track_duration

_service = EvaluationService()


def handle_evaluation_request(payload: EvaluationRequest) -> EvaluationResponse:
    """Evaluate a chatbot response using configured metrics."""

    with track_duration() as elapsed_ms:
        summary = _service.evaluate(
            question=payload.question,
            answer=payload.answer,
            references=payload.references,
        )

    settings = get_settings()
    cost_breakdown = CostBreakdown()
    api_cost = None
    if settings.llm_api_key:
        estimate = estimate_cost(payload.question, payload.answer, settings.llm_cost_per_1k_tokens)
        cost_breakdown = CostBreakdown(**estimate.__dict__)
        api_cost = estimate.estimated_cost_usd

    return EvaluationResponse(
        id=generate_request_id(),
        created_at=datetime.now(timezone.utc),
        time_taken_ms=elapsed_ms(),
        api_cost=api_cost,
        cost=cost_breakdown,
        metrics=summary.metrics,
        average_score=summary.average_score,
    )
