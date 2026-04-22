"""RAGEvaluator — orchestrates retrieval + faithfulness evaluation.

:class:`RAGEvaluator` wraps a :class:`~src.lexreview.agent.legal_rag_agent.LegalRAGAgent`
and a :class:`~src.lexreview.eval.faithfulness.FaithfulnessJudge` to run a
batch of :class:`~src.lexreview.eval.models.EvalSample` instances and
aggregate all metrics into an :class:`~src.lexreview.eval.models.EvalReport`.

Typical usage::

    evaluator = RAGEvaluator(agent, judge)
    report = evaluator.evaluate(samples, k=5)
    for metric in report.metrics:
        print(metric.name, metric.value)
"""

from __future__ import annotations

import time
from collections import defaultdict

from src.lexreview.agent.legal_rag_agent import LegalRAGAgent
from src.lexreview.eval.faithfulness import FaithfulnessJudge
from src.lexreview.eval.metrics import (
    citation_accuracy,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.lexreview.eval.models import EvalReport, EvalSample, MetricResult
from src.utils.logger import get_logger

log = get_logger(__name__)


class RAGEvaluator:
    """Batch evaluator for the LexReview RAG pipeline.

    Runs each :class:`EvalSample` through the agent, computes retrieval
    metrics against ground-truth chunk IDs, and optionally scores
    faithfulness with an LLM judge.

    Args:
        agent: The :class:`LegalRAGAgent` to evaluate.
        judge: Optional :class:`FaithfulnessJudge` for faithfulness scoring.
               When ``None``, faithfulness metrics are skipped.
        k:     Default cut-off rank for precision/recall/nDCG (default 5).

    Example::

        evaluator = RAGEvaluator(agent, judge, k=5)
        report = evaluator.evaluate(samples)
    """

    def __init__(
        self,
        agent: LegalRAGAgent,
        judge: FaithfulnessJudge | None = None,
        k: int = 5,
    ) -> None:
        self._agent = agent
        self._judge = judge
        self._k = k

    def evaluate(
        self,
        samples: list[EvalSample],
        k: int | None = None,
    ) -> EvalReport:
        """Evaluate the agent on all *samples* and return an EvalReport.

        Args:
            samples: List of :class:`EvalSample` ground-truth instances.
            k:       Rank cut-off override; uses constructor *k* if ``None``.

        Returns:
            :class:`EvalReport` with aggregated metrics and per-category breakdown.
        """
        effective_k = k if k is not None else self._k
        log.info(
            "RAGEvaluator: starting evaluation",
            extra={"samples": len(samples), "k": effective_k},
        )

        per_sample: list[dict[str, float]] = []
        category_metrics: dict[str, list[dict[str, float]]] = defaultdict(list)
        total_latency = 0.0

        for sample in samples:
            t0 = time.perf_counter()
            try:
                response = self._agent.answer(query=sample.question)
            except Exception as exc:
                log.warning(
                    "RAGEvaluator: agent failed for sample",
                    extra={"sample_id": sample.sample_id, "error": str(exc)},
                )
                continue

            elapsed_ms = (time.perf_counter() - t0) * 1000
            total_latency += elapsed_ms

            relevant = set(sample.relevant_chunk_ids)
            retrieved_ids = [c.chunk_id for c in response.citations]
            cited_ids = [c.chunk_id for c in response.citations]

            sample_metrics: dict[str, float] = {
                "precision": precision_at_k(retrieved_ids, relevant, effective_k),
                "recall": recall_at_k(retrieved_ids, relevant, effective_k),
                "mrr": mrr(retrieved_ids, relevant),
                "ndcg": ndcg_at_k(retrieved_ids, relevant, effective_k),
                "citation_accuracy": citation_accuracy(cited_ids, relevant),
            }

            if self._judge is not None:
                context_texts = [c.content for c in response.citations]
                faith_result = self._judge.score(
                    answer=response.answer, context_chunks=context_texts
                )
                sample_metrics["faithfulness"] = float(faith_result["score"])

            per_sample.append(sample_metrics)
            category_metrics[sample.category].append(sample_metrics)

            log.debug(
                "RAGEvaluator: sample done",
                extra={"sample_id": sample.sample_id, **sample_metrics},
            )

        if not per_sample:
            return EvalReport(sample_count=0)

        # ── Aggregate ────────────────────────────────────────────────────────
        metric_names = list(per_sample[0].keys())
        aggregated: list[MetricResult] = []
        for name in metric_names:
            values = [s[name] for s in per_sample if name in s]
            avg = sum(values) / len(values) if values else 0.0
            per_sample_details = {
                f"sample_{i}": round(v, 4) for i, v in enumerate(values)
            }
            aggregated.append(
                MetricResult(
                    name=f"{name}@{effective_k}" if name not in {"mrr", "faithfulness", "citation_accuracy"} else name,
                    value=round(avg, 4),
                    details=per_sample_details,
                )
            )

        # Per-category breakdown
        category_reports: dict[str, list[MetricResult]] = {}
        for cat, cat_samples in category_metrics.items():
            cat_results: list[MetricResult] = []
            for name in metric_names:
                vals = [s[name] for s in cat_samples if name in s]
                avg = sum(vals) / len(vals) if vals else 0.0
                cat_results.append(
                    MetricResult(
                        name=f"{name}@{effective_k}" if name not in {"mrr", "faithfulness", "citation_accuracy"} else name,
                        value=round(avg, 4),
                        details={},
                    )
                )
            category_reports[cat] = cat_results

        avg_latency = total_latency / len(per_sample)
        log.info(
            "RAGEvaluator: evaluation complete",
            extra={
                "samples": len(per_sample),
                "avg_latency_ms": round(avg_latency, 2),
                "metrics": {m.name: m.value for m in aggregated},
            },
        )
        return EvalReport(
            metrics=aggregated,
            sample_count=len(per_sample),
            avg_latency_ms=round(avg_latency, 2),
            categories=category_reports,
        )
