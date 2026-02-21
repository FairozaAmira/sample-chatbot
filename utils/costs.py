"""Cost estimation utilities for API usage."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostEstimate:
    """Represents an estimated token usage and cost."""

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_usd: float


def _estimate_tokens(text: str) -> int:
    """Approximate token count from a text blob using a heuristic."""

    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_cost(prompt: str, completion: str, rate_per_1k: float = 0.0) -> CostEstimate:
    """Return token usage and rough cost given rate per thousand tokens."""

    input_tokens = _estimate_tokens(prompt)
    output_tokens = _estimate_tokens(completion)
    total_tokens = input_tokens + output_tokens
    estimated_cost_usd = (total_tokens / 1000) * rate_per_1k
    return CostEstimate(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=round(estimated_cost_usd, 6),
    )
