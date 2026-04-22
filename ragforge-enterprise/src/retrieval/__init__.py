"""Retrieval layer for RAGForge Enterprise.

Exposes dense, sparse, and hybrid retrievers together with the
CrossEncoder reranker. Import the concrete classes directly from this
package rather than from their individual modules.

Example::

    from src.retrieval import DenseRetriever, HybridRetriever, CrossEncoderReranker
"""

from src.retrieval.base import BaseRetriever, RetrievalError
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, RerankerError
from src.retrieval.sparse_retriever import SparseRetriever

__all__ = [
    "BaseRetriever",
    "RetrievalError",
    "DenseRetriever",
    "SparseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "RerankerError",
]
