"""Eval data-models for the LexReview evaluation harness.

Classes
-------
EvalSample
    A single ground-truth Q&A pair with relevant chunk IDs.
MetricResult
    A named metric value with optional breakdown details.
EvalReport
    Aggregated evaluation report across a set of EvalSamples.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EvalSample(BaseModel):
    """A single ground-truth evaluation sample.

    Attributes:
        sample_id:             Unique identifier for the sample.
        question:              Legal question to pose to the agent.
        ground_truth_answer:   Expected answer (used by faithfulness judge).
        relevant_chunk_ids:    Set of chunk IDs that are definitively relevant.
        category:              Optional topic category for breakdown analysis.
    """

    sample_id: str = Field(..., description="Unique sample identifier.")
    question: str = Field(..., description="Legal query to evaluate against.")
    ground_truth_answer: str = Field(
        ..., description="Gold-standard answer for faithfulness scoring."
    )
    relevant_chunk_ids: list[str] = Field(
        ..., description="Ground-truth set of relevant chunk IDs."
    )
    category: str = Field(
        default="general", description="Topic category for evaluation breakdown."
    )


class MetricResult(BaseModel):
    """A single named metric value with optional breakdown details.

    Attributes:
        name:    Metric name (e.g. ``"precision@5"``).
        value:   Numeric metric value.
        details: Optional per-sample or per-category breakdown.
    """

    name: str = Field(..., description="Metric identifier.")
    value: float = Field(..., description="Numeric metric value.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-sample or per-category breakdown details.",
    )


class EvalReport(BaseModel):
    """Aggregated evaluation report across a set of EvalSamples.

    Attributes:
        metrics:          List of aggregated MetricResult objects.
        sample_count:     Number of samples evaluated.
        avg_latency_ms:   Average agent pipeline latency in ms.
        categories:       Per-category metric breakdowns if available.
    """

    metrics: list[MetricResult] = Field(default_factory=list)
    sample_count: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
    categories: dict[str, list[MetricResult]] = Field(default_factory=dict)
