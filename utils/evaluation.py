"""Evaluation utilities leveraging deepeval with graceful degradation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from .config import get_settings
from .schemas import EvaluationMetricResult, EvaluationReference

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency branch
    from deepeval.dataset import EvaluationSample
    from deepeval.evaluate import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        FaithfulnessMetric,
    )

    HAS_DEEPEVAL = True
except Exception as exc:  # pragma: no cover - offline fallback
    LOGGER.warning("deepeval is unavailable: %s", exc)
    HAS_DEEPEVAL = False


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics and average score."""

    metrics: List[EvaluationMetricResult]
    average_score: float


class EvaluationService:
    """High-level evaluator for chatbot answers."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def evaluate(self, question: str, answer: str, references: List[EvaluationReference]) -> EvaluationSummary:
        if HAS_DEEPEVAL and self.settings.enable_deepeval and self.settings.llm_api_key:
            return self._evaluate_with_deepeval(question, answer, references)
        LOGGER.info("Falling back to heuristic evaluation; deepeval unavailable or not configured.")
        return self._evaluate_with_heuristics(question, answer, references)

    # ------------------------------------------------------------------
    # Deepeval-backed evaluation
    # ------------------------------------------------------------------
    def _evaluate_with_deepeval(
        self, question: str, answer: str, references: List[EvaluationReference]
    ) -> EvaluationSummary:
        contexts = [ref.context for ref in references if ref.context.strip()]
        if not contexts:
            contexts = ["No reference context supplied."]

        samples = [
            EvaluationSample(
                input=question,
                actual_output=answer,
                expected_output="\n\n".join(contexts),
                context=contexts,
            )
        ]

        metrics = [
            FaithfulnessMetric(model=self.settings.llm_model, api_key=self.settings.llm_api_key or ""),
            ContextualPrecisionMetric(model=self.settings.llm_model, api_key=self.settings.llm_api_key or ""),
            AnswerRelevancyMetric(model=self.settings.llm_model, api_key=self.settings.llm_api_key or ""),
        ]

        try:
            evaluation_result = evaluate(samples=samples, metrics=metrics)
        except Exception as exc:  # pragma: no cover - runtime fallback
            LOGGER.warning("deepeval execution failed; switching to heuristics: %s", exc)
            return self._evaluate_with_heuristics(question, answer, references)

        metric_results: List[EvaluationMetricResult] = []
        total_score = 0.0
        for metric in evaluation_result.results[0].metrics:  # type: ignore[attr-defined]
            score = getattr(metric, "score", 0.0)
            passed = getattr(metric, "passed", bool(score >= 0.5))
            feedback = getattr(metric, "reason", None)
            total_score += score
            metric_results.append(
                EvaluationMetricResult(
                    metric=getattr(metric, "name", metric.__class__.__name__),
                    score=score,
                    passed=passed,
                    feedback=feedback,
                )
            )

        average_score = total_score / max(len(metric_results), 1)
        return EvaluationSummary(metrics=metric_results, average_score=round(average_score, 4))

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------
    def _evaluate_with_heuristics(
        self, question: str, answer: str, references: List[EvaluationReference]
    ) -> EvaluationSummary:
        answer_lower = answer.lower()
        reference_text = " ".join(ref.context.lower() for ref in references)
        overlap_ratio = self._calculate_overlap(answer_lower, reference_text)
        question_terms = set(token for token in question.lower().split() if len(token) > 3)
        answer_terms = set(token for token in answer_lower.split())
        coverage_ratio = len(question_terms & answer_terms) / max(len(question_terms), 1)

        metric_results = [
            EvaluationMetricResult(
                metric="ReferenceOverlap",
                score=round(overlap_ratio, 4),
                passed=overlap_ratio >= 0.45,
                feedback="Checks for hallucination via reference overlap.",
            ),
            EvaluationMetricResult(
                metric="QuestionCoverage",
                score=round(coverage_ratio, 4),
                passed=coverage_ratio >= 0.4,
                feedback="Ensures answer addresses major terms in the question.",
            ),
        ]

        average_score = sum(metric.score for metric in metric_results) / len(metric_results)
        return EvaluationSummary(metrics=metric_results, average_score=round(average_score, 4))

    @staticmethod
    def _calculate_overlap(answer: str, reference: str) -> float:
        if not answer.strip() or not reference.strip():
            return 0.0
        answer_tokens = set(answer.split())
        reference_tokens = set(reference.split())
        return len(answer_tokens & reference_tokens) / max(len(answer_tokens), 1)


__all__ = ["EvaluationService", "EvaluationSummary"]
