"""eval sub-package — metrics, faithfulness judge, evaluator, samples."""

from src.lexreview.eval.evaluator import RAGEvaluator
from src.lexreview.eval.faithfulness import FaithfulnessJudge
from src.lexreview.eval.metrics import (
    citation_accuracy,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src.lexreview.eval.models import EvalReport, EvalSample, MetricResult
from src.lexreview.eval.samples import EVAL_SAMPLES

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "citation_accuracy",
    "FaithfulnessJudge",
    "RAGEvaluator",
    "EvalSample",
    "MetricResult",
    "EvalReport",
    "EVAL_SAMPLES",
]
