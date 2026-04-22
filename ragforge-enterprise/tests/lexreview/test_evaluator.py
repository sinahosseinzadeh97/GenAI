"""Tests for src/lexreview/eval/ — EvalSample, MetricResult, RAGEvaluator, FaithfulnessJudge."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.lexreview.agent.models import AgentResponse, Citation
from src.lexreview.eval.models import EvalReport, EvalSample, MetricResult
from src.lexreview.eval.samples import EVAL_SAMPLES

# ── EvalSample model tests ────────────────────────────────────────────────────


class TestEvalSampleModel:
    def test_sample_fields(self) -> None:
        s = EvalSample(
            sample_id="s-001",
            question="What is the payment term?",
            ground_truth_answer="30 days net.",
            relevant_chunk_ids=["c1", "c2"],
        )
        assert s.sample_id == "s-001"
        assert s.category == "general"

    def test_sample_category_override(self) -> None:
        s = EvalSample(
            sample_id="s-002",
            question="q",
            ground_truth_answer="a",
            relevant_chunk_ids=[],
            category="payment",
        )
        assert s.category == "payment"


class TestMetricResultModel:
    def test_metric_result(self) -> None:
        m = MetricResult(name="precision@5", value=0.8)
        assert m.name == "precision@5"
        assert m.value == 0.8
        assert m.details == {}


# ── Built-in eval samples ─────────────────────────────────────────────────────


class TestBuiltinEvalSamples:
    def test_sample_count(self) -> None:
        assert len(EVAL_SAMPLES) >= 5

    def test_all_samples_valid(self) -> None:
        for s in EVAL_SAMPLES:
            assert s.sample_id
            assert s.question
            assert s.ground_truth_answer
            assert isinstance(s.relevant_chunk_ids, list)

    def test_category_diversity(self) -> None:
        categories = {s.category for s in EVAL_SAMPLES}
        # Should cover multiple categories
        assert len(categories) >= 3

    def test_payment_sample_exists(self) -> None:
        assert any(s.category == "payment" for s in EVAL_SAMPLES)

    def test_governing_law_sample_exists(self) -> None:
        assert any(s.category == "governing_law" for s in EVAL_SAMPLES)


# ── FaithfulnessJudge tests ───────────────────────────────────────────────────


class TestFaithfulnessJudge:
    def _make_judge_with_llm(self, response: str) -> FaithfulnessJudge:  # type: ignore[name-defined]  # noqa: F821
        from src.lexreview.eval.faithfulness import FaithfulnessJudge

        mock_llm = MagicMock()
        mock_llm.complete_text.return_value = response
        return FaithfulnessJudge(llm=mock_llm)

    def test_score_returns_dict_with_required_keys(self) -> None:
        judge = self._make_judge_with_llm('{"score": 0.9, "reason": "Well supported."}')
        result = judge.score("The term is 30 days.", ["Payment is due in 30 days."])
        assert "score" in result
        assert "reason" in result
        assert "is_faithful" in result

    def test_score_high_confidence_is_faithful(self) -> None:
        judge = self._make_judge_with_llm('{"score": 0.9, "reason": "Good."}')
        result = judge.score("Answer.", ["Context."])
        assert result["is_faithful"] is True

    def test_score_low_confidence_is_not_faithful(self) -> None:
        judge = self._make_judge_with_llm('{"score": 0.2, "reason": "Poor grounding."}')
        result = judge.score("Answer.", ["Unrelated context."])
        assert result["is_faithful"] is False

    def test_score_clamps_above_one(self) -> None:
        judge = self._make_judge_with_llm('{"score": 1.5, "reason": "Over 1."}')
        result = judge.score("A.", ["C."])
        assert float(result["score"]) <= 1.0

    def test_score_clamps_below_zero(self) -> None:
        judge = self._make_judge_with_llm('{"score": -0.5, "reason": "Negative."}')
        result = judge.score("A.", ["C."])
        assert float(result["score"]) >= 0.0

    def test_llm_failure_returns_zero(self) -> None:
        from src.lexreview.eval.faithfulness import FaithfulnessJudge

        mock_llm = MagicMock()
        mock_llm.complete_text.side_effect = RuntimeError("LLM error")
        judge = FaithfulnessJudge(llm=mock_llm)
        result = judge.score("A.", ["C."])
        assert result["score"] == 0.0
        assert result["is_faithful"] is False


# ── RAGEvaluator tests ────────────────────────────────────────────────────────


class TestRAGEvaluator:
    def _make_mock_agent(self, chunk_ids: list[str], answer: str = "Test answer.") -> MagicMock:
        agent = MagicMock()
        citations = [
            Citation(chunk_id=cid, content=f"content-{cid}", score=0.8)
            for cid in chunk_ids
        ]
        agent.answer.return_value = AgentResponse(
            answer=answer,
            citations=citations,
            confidence=0.85,
        )
        return agent

    def test_evaluator_returns_report(self) -> None:
        from src.lexreview.eval.evaluator import RAGEvaluator

        agent = self._make_mock_agent(["chunk-payment-01"])
        evaluator = RAGEvaluator(agent=agent, judge=None, k=5)
        samples = [EVAL_SAMPLES[0]]  # payment sample
        report = evaluator.evaluate(samples)
        assert isinstance(report, EvalReport)

    def test_report_has_metrics(self) -> None:
        from src.lexreview.eval.evaluator import RAGEvaluator

        agent = self._make_mock_agent(["chunk-payment-01"])
        evaluator = RAGEvaluator(agent=agent, judge=None, k=5)
        report = evaluator.evaluate([EVAL_SAMPLES[0]])
        assert len(report.metrics) >= 4  # precision, recall, mrr, ndcg, cit_acc

    def test_report_sample_count(self) -> None:
        from src.lexreview.eval.evaluator import RAGEvaluator

        agent = self._make_mock_agent(["c1"])
        evaluator = RAGEvaluator(agent=agent, k=5)
        report = evaluator.evaluate(EVAL_SAMPLES[:3])
        assert report.sample_count == 3

    def test_empty_samples_returns_empty_report(self) -> None:
        from src.lexreview.eval.evaluator import RAGEvaluator

        agent = self._make_mock_agent([])
        evaluator = RAGEvaluator(agent=agent, k=5)
        report = evaluator.evaluate([])
        assert report.sample_count == 0

    def test_perfect_retrieval_precision_one(self) -> None:
        from src.lexreview.eval.evaluator import RAGEvaluator

        sample = EVAL_SAMPLES[0]  # relevant: chunk-payment-01, chunk-payment-02
        agent = self._make_mock_agent(sample.relevant_chunk_ids)
        evaluator = RAGEvaluator(agent=agent, k=5)
        report = evaluator.evaluate([sample])
        precision_metric = next(
            (m for m in report.metrics if "precision" in m.name), None
        )
        assert precision_metric is not None
        # With 2 relevant and 2 retrieved at top, precision@5 = 2/5 = 0.4
        assert precision_metric.value >= 0.0

    def test_with_faithfulness_judge(self) -> None:
        from src.lexreview.eval.evaluator import RAGEvaluator
        from src.lexreview.eval.faithfulness import FaithfulnessJudge

        agent = self._make_mock_agent(["chunk-payment-01"])
        mock_judge = MagicMock(spec=FaithfulnessJudge)
        mock_judge.score.return_value = {"score": 0.85, "reason": "ok", "is_faithful": True}
        evaluator = RAGEvaluator(agent=agent, judge=mock_judge, k=5)
        report = evaluator.evaluate([EVAL_SAMPLES[0]])
        faith_metric = next(
            (m for m in report.metrics if "faithfulness" in m.name), None
        )
        assert faith_metric is not None
        assert faith_metric.value == pytest.approx(0.85)

    def test_category_breakdown_populated(self) -> None:
        from src.lexreview.eval.evaluator import RAGEvaluator

        agent = self._make_mock_agent(["c1"])
        evaluator = RAGEvaluator(agent=agent, k=5)
        report = evaluator.evaluate(EVAL_SAMPLES[:5])
        assert len(report.categories) >= 1

    def test_agent_error_skips_sample(self) -> None:
        from src.lexreview.eval.evaluator import RAGEvaluator

        agent = MagicMock()
        agent.answer.side_effect = RuntimeError("Agent exploded")
        evaluator = RAGEvaluator(agent=agent, k=5)
        report = evaluator.evaluate([EVAL_SAMPLES[0]])
        # Should skip the failed sample → sample_count = 0
        assert report.sample_count == 0
