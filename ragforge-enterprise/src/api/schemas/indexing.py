"""Pydantic request/response schemas for indexing endpoints.

Classes:
    IndexRequest:  Incoming indexing payload (base64-encoded file content).
    IndexResponse: Indexing result summary.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    """Request body for POST /index.

    Attributes:
        collection_name:     Target Qdrant collection name (created if absent).
        chunking_strategy:   One of ``"fixed"``, ``"recursive"``, ``"semantic"``.
        filename:            Original filename (used to determine parser).
        file_content_base64: Base64-encoded raw file bytes.
    """

    collection_name: str = Field(..., description="Target Qdrant collection name.")
    chunking_strategy: str = Field(
        default="recursive",
        description="One of: 'fixed', 'recursive', 'semantic'.",
    )
    filename: str = Field(..., description="Original filename (e.g. 'report.pdf').")
    file_content_base64: str = Field(
        ..., description="Base64-encoded file content."
    )


class IndexResponse(BaseModel):
    """Indexing run summary returned by POST /index.

    Attributes:
        collection_name:          Collection that was upserted into.
        total_chunks_processed:   Total chunks extracted from the document.
        total_indexed:            Chunks successfully stored.
        total_failed:             Chunks that failed to index.
        success_rate:             Fraction of chunks indexed (0.0–1.0).
        embedding_time_seconds:   Time spent on embedding.
        indexing_time_seconds:    Time spent on Qdrant upsert.
        avg_embedding_throughput: Chunks embedded per second.
        embedder_model:           Model identifier used for embedding.
    """

    collection_name: str
    total_chunks_processed: int
    total_indexed: int
    total_failed: int
    success_rate: float
    embedding_time_seconds: float
    indexing_time_seconds: float
    avg_embedding_throughput: float
    embedder_model: str
