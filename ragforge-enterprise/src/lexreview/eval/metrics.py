"""Pure-Python retrieval and faithfulness metrics for RAG evaluation.

All functions are stateless and accept plain Python lists so they can be
tested without any external framework.

Functions
---------
precision_at_k
recall_at_k
mrr
ndcg_at_k
citation_accuracy
"""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant:  Set of ground-truth relevant chunk IDs.
        k:         Cut-off rank.

    Returns:
        Precision@k in [0.0, 1.0].

    Example:
        >>> precision_at_k(["a", "b", "c"], {"a", "c"}, k=3)
        0.6666666666666666
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top-k retrieved.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant:  Set of ground-truth relevant chunk IDs.
        k:         Cut-off rank.

    Returns:
        Recall@k in [0.0, 1.0].  Returns ``0.0`` when *relevant* is empty.

    Example:
        >>> recall_at_k(["a", "b", "c"], {"a", "d"}, k=3)
        0.5
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first relevant item.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant:  Set of ground-truth relevant chunk IDs.

    Returns:
        MRR score in (0.0, 1.0].  Returns ``0.0`` when no relevant item is found.

    Example:
        >>> mrr(["x", "a", "b"], {"a"})
        0.5
    """
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Discounted Cumulative Gain at rank k (binary relevance).

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant:  Set of ground-truth relevant chunk IDs.
        k:         Cut-off rank.

    Returns:
        DCG@k.
    """
    dcg = 0.0
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank k.

    Ideal DCG is computed assuming the top-``|relevant|`` positions are all hits.

    Args:
        retrieved: Ordered list of retrieved chunk IDs.
        relevant:  Set of ground-truth relevant chunk IDs.
        k:         Cut-off rank.

    Returns:
        nDCG@k in [0.0, 1.0].  Returns ``0.0`` when *relevant* is empty.

    Example:
        >>> ndcg_at_k(["a", "b", "c"], {"a", "c"}, k=3)  # doctest: +ELLIPSIS
        0.92...
    """
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    ideal_retrieved = [f"__ideal_{i}" for i in range(ideal_hits)]
    ideal_relevant = set(ideal_retrieved)
    idcg = _dcg_at_k(ideal_retrieved, ideal_relevant, k)
    if idcg == 0.0:
        return 0.0
    return _dcg_at_k(retrieved, relevant, k) / idcg


def citation_accuracy(cited_ids: list[str], relevant: set[str]) -> float:
    """Fraction of cited chunks that are actually relevant.

    This measures citation precision — are the sources cited in the answer
    actually ground-truth relevant chunks?

    Args:
        cited_ids: Chunk IDs that appear in the agent's citations.
        relevant:  Set of ground-truth relevant chunk IDs.

    Returns:
        Citation accuracy in [0.0, 1.0].  Returns ``0.0`` when no citations.

    Example:
        >>> citation_accuracy(["a", "b"], {"a", "c"})
        0.5
    """
    if not cited_ids:
        return 0.0
    hits = sum(1 for cid in cited_ids if cid in relevant)
    return hits / len(cited_ids)
