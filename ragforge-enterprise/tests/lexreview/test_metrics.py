"""Tests for src/lexreview/eval/metrics.py — all 5 metric functions."""

from __future__ import annotations

import math

import pytest

from src.lexreview.eval.metrics import (
    citation_accuracy,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == pytest.approx(1.0)

    def test_none_relevant(self) -> None:
        assert precision_at_k(["x", "y", "z"], {"a", "b"}, k=3) == pytest.approx(0.0)

    def test_partial_relevant(self) -> None:
        result = precision_at_k(["a", "b", "c"], {"a", "c"}, k=3)
        assert result == pytest.approx(2 / 3)

    def test_k_larger_than_retrieved(self) -> None:
        # Only 2 items but k=5
        result = precision_at_k(["a", "b"], {"a", "b"}, k=5)
        assert result == pytest.approx(2 / 5)

    def test_k_zero_returns_zero(self) -> None:
        assert precision_at_k(["a", "b"], {"a"}, k=0) == 0.0

    def test_empty_retrieved(self) -> None:
        assert precision_at_k([], {"a"}, k=5) == pytest.approx(0.0)

    def test_cut_off_respected(self) -> None:
        # "c" is at rank 3, k=2 should not count it
        result = precision_at_k(["x", "y", "c"], {"c"}, k=2)
        assert result == pytest.approx(0.0)


class TestRecallAtK:
    def test_all_relevant_retrieved(self) -> None:
        assert recall_at_k(["a", "b"], {"a", "b"}, k=5) == pytest.approx(1.0)

    def test_none_relevant_retrieved(self) -> None:
        assert recall_at_k(["x", "y"], {"a", "b"}, k=5) == pytest.approx(0.0)

    def test_partial_recall(self) -> None:
        result = recall_at_k(["a", "x"], {"a", "b"}, k=5)
        assert result == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_k_truncates_retrieved(self) -> None:
        # "b" is at rank 2, k=1 cuts it off
        result = recall_at_k(["a", "b"], {"b"}, k=1)
        assert result == pytest.approx(0.0)


class TestMRR:
    def test_first_position_hit(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_position_hit(self) -> None:
        assert mrr(["x", "a", "c"], {"a"}) == pytest.approx(0.5)

    def test_third_position_hit(self) -> None:
        assert mrr(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_no_hit_returns_zero(self) -> None:
        assert mrr(["x", "y", "z"], {"a", "b"}) == pytest.approx(0.0)

    def test_empty_retrieved(self) -> None:
        assert mrr([], {"a"}) == pytest.approx(0.0)

    def test_multiple_relevant_uses_first(self) -> None:
        # First relevant at rank 2
        assert mrr(["x", "a", "b"], {"a", "b"}) == pytest.approx(0.5)


class TestNDCGAtK:
    def test_perfect_ranking(self) -> None:
        # All top-k are relevant → nDCG = 1.0
        result = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert result == pytest.approx(1.0)

    def test_no_relevant_returns_zero(self) -> None:
        assert ndcg_at_k(["x", "y"], {"a", "b"}, k=2) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self) -> None:
        assert ndcg_at_k(["a", "b"], set(), k=2) == 0.0

    def test_partial_ranking(self) -> None:
        # Relevant at ranks 1 and 3
        result = ndcg_at_k(["a", "x", "c"], {"a", "c"}, k=3)
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # iDCG = 1/log2(2) + 1/log2(3) ≈ 1 + 0.631 = 1.631
        expected = (1.0 + 0.5) / (1.0 + 1.0 / math.log2(3))
        assert result == pytest.approx(expected, rel=1e-4)

    def test_all_relevant_at_bottom(self) -> None:
        # Relevant only at last position
        result = ndcg_at_k(["x", "x", "a"], {"a"}, k=3)
        assert result < 1.0 and result > 0.0


class TestCitationAccuracy:
    def test_all_citations_relevant(self) -> None:
        assert citation_accuracy(["a", "b"], {"a", "b", "c"}) == pytest.approx(1.0)

    def test_no_citations_relevant(self) -> None:
        assert citation_accuracy(["x", "y"], {"a", "b"}) == pytest.approx(0.0)

    def test_partial(self) -> None:
        assert citation_accuracy(["a", "x"], {"a", "b"}) == pytest.approx(0.5)

    def test_empty_citations_returns_zero(self) -> None:
        assert citation_accuracy([], {"a", "b"}) == 0.0

    def test_empty_relevant_returns_zero_accuracy(self) -> None:
        # None of the cited IDs are in an empty relevant set
        assert citation_accuracy(["a", "b"], set()) == pytest.approx(0.0)
