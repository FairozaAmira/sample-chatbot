"""Pydantic schemas shared across controllers and routers."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatDomain(str, Enum):
    """Supported knowledge domains for the chatbot."""

    knowledge = "knowledge"
    tender = "tender"
    finance = "finance"


class Citation(BaseModel):
    """Represents contextual evidence backing a chatbot answer."""

    source: str
    snippet: str
    score: Optional[float] = None


class CostBreakdown(BaseModel):
    """Tracks token usage and monetary cost of an interaction."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class BaseAPIResponse(BaseModel):
    """Common metadata returned by all API endpoints."""

    id: str = Field(..., description="Primary key identifier for the response")
    created_at: datetime = Field(..., description="UTC timestamp when the response was generated")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    cost: CostBreakdown = Field(default_factory=CostBreakdown)


class ChatbotRequest(BaseModel):
    """Incoming request payload for chatbot interactions."""

    query: str = Field(..., min_length=3, description="User's question or instruction")
    domain: ChatDomain = Field(default=ChatDomain.knowledge)
    top_k: int = Field(default=4, ge=1, le=10)


class ChatbotResponse(BaseAPIResponse):
    """Response payload for chatbot interactions."""

    answer: str
    citations: List[Citation] = Field(default_factory=list)


class CrawlRequest(BaseModel):
    """Incoming payload for the data crawler endpoint."""

    refresh_index: bool = Field(default=True, description="Whether to rebuild the index from scratch")
    max_qas_per_document: int = Field(default=2, ge=0, le=10)


class CrawlResponse(BaseAPIResponse):
    """Response payload describing ingestion results."""

    documents_indexed: int
    qa_pairs_generated: int
    qa_output_path: Optional[str] = None


class EvaluationReference(BaseModel):
    """Reference materials for evaluation."""

    context: str
    source: Optional[str] = None


class EvaluationRequest(BaseModel):
    """Incoming payload for evaluation endpoint."""

    question: str
    answer: str
    references: List[EvaluationReference]


class EvaluationMetricResult(BaseModel):
    """Individual metric outcome from evaluation."""

    metric: str
    score: float
    passed: bool
    feedback: Optional[str] = None


class EvaluationResponse(BaseAPIResponse):
    """Aggregated evaluation result."""

    metrics: List[EvaluationMetricResult]
    average_score: float
