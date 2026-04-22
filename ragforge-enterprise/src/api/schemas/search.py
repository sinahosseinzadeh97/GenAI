"""Pydantic request/response schemas for search endpoints.

All models use Pydantic v2 validation.

Classes:
    SearchRequest:       Incoming search payload.
    SearchResultSchema:  A single result item in the response.
    SearchResponse:      Full search response envelope.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request body for POST /search and POST /search/hybrid.

    Attributes:
        query:           Natural-language query string (3–500 characters).
        top_k:           Number of results to return (1–50, default 10).
        collection_name: Qdrant collection to search.
        filters:         Optional key/value metadata filters.
        rerank:          Whether to apply cross-encoder reranking.
    """

    query: str = Field(..., min_length=3, max_length=500, description="Search query text.")
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of results to return (1–50).",
    )
    collection_name: str = Field(..., description="Target Qdrant collection name.")
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata filter map (key → exact value).",
    )
    rerank: bool = Field(
        default=True,
        description="Whether to apply cross-encoder reranking.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "query": "invoice processing workflow",
            "top_k": 5,
            "collection_name": "ragforge_docs",
            "rerank": True,
        }
    }}


class SearchResultSchema(BaseModel):
    """A single result item in a search response.

    Attributes:
        chunk_id: Unique identifier of the chunk.
        content:  Raw text content of the retrieved chunk.
        score:    Similarity or RRF score (float).
        metadata: Arbitrary metadata dict stored alongside the vector.
    """

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Full search response envelope.

    Attributes:
        query:               Echo of the input query.
        results:             Ordered list of :class:`SearchResultSchema`.
        total_found:         Total number of results returned.
        latency_ms:          End-to-end server-side latency in milliseconds.
        retrieval_strategy:  ``"dense"`` or ``"hybrid"``.
        reranked:            Whether cross-encoder reranking was applied.
    """

    query: str
    results: list[SearchResultSchema]
    total_found: int
    latency_ms: float
    retrieval_strategy: str  # "dense" | "hybrid"
    reranked: bool
