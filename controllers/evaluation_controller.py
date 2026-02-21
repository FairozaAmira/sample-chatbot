"""Controller handling chatbot evaluation requests."""
from __future__ import annotations

from datetime import datetime, timezone

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

    return EvaluationResponse(
        id=generate_request_id(),
        created_at=datetime.now(timezone.utc),
        processing_time_ms=elapsed_ms(),
        cost=CostBreakdown(),
        metrics=summary.metrics,
        average_score=summary.average_score,
    )
