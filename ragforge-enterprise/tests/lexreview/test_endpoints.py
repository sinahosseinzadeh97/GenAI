"""Async integration tests for the four LexReview endpoints + /health.

Tests use ``httpx.AsyncClient`` with FastAPI's ``ASGITransport`` so the full
ASGI middleware stack (CORS, lifespan) is exercised without a live server.

Heavy dependencies (LLM calls, Qdrant, embedder, cross-encoder reranker) are
mocked at the module-path level so no real network or model I/O occurs.

Authentication:
    ``verify_api_key`` is a no-op when ``settings.api_key`` is empty (dev
    mode).  All authenticated tests therefore send ``X-API-Key: ""``; the
    header is still required by FastAPI's ``Header(...)`` declaration.
"""

from __future__ import annotations

import datetime
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

# ── App import (also exercises lifespan startup) ──────────────────────────────

from src.api.main import app
from src.lexreview.agent.models import AgentResponse, Citation
from src.lexreview.extraction.models import Clause, LegalEntities
from src.vectorstore.schema import SearchResult

# ── Constants ─────────────────────────────────────────────────────────────────

_AUTH_HEADERS: dict[str, str] = {"X-API-Key": ""}
_LONG_TEXT = (
    "This Agreement is entered into as of January 1, 2024, by and between "
    "Acme Corporation ('Client') and Globex Ltd ('Provider').  The Provider "
    "agrees to deliver software development services as described in Exhibit A. "
    "Payment shall be made within thirty (30) days of invoice receipt."
)


# ── Shared async client fixture ───────────────────────────────────────────────


@pytest_asyncio.fixture()
async def async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Yield an ``httpx.AsyncClient`` pointed at the FastAPI ASGI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ── Helper factories ──────────────────────────────────────────────────────────


def _make_agent_response() -> AgentResponse:
    return AgentResponse(
        answer="The governing law clause specifies Delaware law.",
        citations=[
            Citation(
                chunk_id="chunk-001",
                content="Governed by the laws of the State of Delaware.",
                score=0.92,
                source="contract.pdf",
            )
        ],
        confidence=0.88,
        reasoning_steps=[
            "[UNDERSTAND] User asks about the governing law.",
            "[REASON] Chunk chunk-001 directly addresses jurisdiction.",
            "[ANSWER] Delaware law governs this agreement.",
        ],
        latency_ms=145.0,
    )


def _make_search_result(chunk_id: str = "chunk-001") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content="Party A shall indemnify Party B against all third-party claims.",
        score=0.87,
        metadata={"source": "nda.pdf", "page": 3},
        rank=1,
    )


# ── /health ───────────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """GET /health must return 200 and ``{"status": "ok"}`` — no auth needed."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "service" in data


# ── POST /lexreview/query ─────────────────────────────────────────────────────


class TestQueryEndpoint:
    @pytest.mark.asyncio
    async def test_query_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """POST /lexreview/query: 200 with ``answer``, ``citations``, ``confidence``."""
        mock_agent = MagicMock()
        mock_agent.answer.return_value = _make_agent_response()

        # Patch the internal builder so no Qdrant / embedder / reranker I/O occurs.
        with patch(
            "src.lexreview.api.router._build_agent",
            return_value=mock_agent,
        ):
            response = await async_client.post(
                "/lexreview/query",
                json={
                    "query": "What is the governing law clause in this contract?",
                    "collection_name": "contracts",
                    "top_k": 5,
                    "rerank": False,
                },
                headers=_AUTH_HEADERS,
            )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert "reasoning_steps" in data
        assert "latency_ms" in data
        assert "query" in data
        assert "collection_name" in data
        # Sanity-check values flow through from the mocked agent.
        assert "Delaware" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["citations"][0]["chunk_id"] == "chunk-001"


# ── POST /lexreview/extract ───────────────────────────────────────────────────


class TestExtractEndpoint:
    @pytest.mark.asyncio
    async def test_extract_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """POST /lexreview/extract: 200 with ``entities``, ``clauses``, ``char_count``."""
        mock_entities = LegalEntities(
            parties=["Acme Corporation", "Globex Ltd"],
            dates=["January 1, 2024"],
            amounts=["$50,000"],
        )
        mock_clauses = [
            Clause(
                type="payment",
                text="Payment shall be made within thirty (30) days of invoice receipt.",
                span=(0, 64),
                confidence=0.91,
            )
        ]

        with (
            patch(
                "src.lexreview.api.router.LegalNER.extract",
                return_value=mock_entities,
            ),
            patch(
                "src.lexreview.api.router.RegexExtractor.extract",
                return_value=LegalEntities(amounts=["$50,000"]),
            ),
            patch(
                "src.lexreview.api.router.RegexExtractor.merge",
                return_value=mock_entities,
            ),
            patch(
                "src.lexreview.api.router.ClauseDetector.detect",
                return_value=mock_clauses,
            ),
        ):
            response = await async_client.post(
                "/lexreview/extract",
                json={
                    "text": _LONG_TEXT,
                    "extract_entities": True,
                    "detect_clauses": True,
                },
                headers=_AUTH_HEADERS,
            )

        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "clauses" in data
        assert "char_count" in data
        assert data["char_count"] == len(_LONG_TEXT)
        # Verify entity data propagates correctly.
        assert "Acme Corporation" in data["entities"]["parties"]
        assert len(data["clauses"]) == 1
        assert data["clauses"][0]["type"] == "payment"


# ── POST /lexreview/index ─────────────────────────────────────────────────────


class TestIndexEndpoint:
    @pytest.mark.asyncio
    async def test_index_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """POST /lexreview/index: 200 with ``indexed_count``, ``collection_name``, ``latency_ms``."""
        sample_texts = [
            "Party A shall indemnify Party B against all third-party claims.",
            "This agreement is governed by the laws of the State of Delaware.",
        ]

        mock_embedder = MagicMock()
        # Return two fake 3-dim embedding vectors (shape matches len(sample_texts)).
        mock_embedder.embed.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]

        mock_store = MagicMock()
        mock_store.upsert_chunks.return_value = None

        with (
            # The router uses lazy local imports inside the endpoint body, so
            # we patch the names in their *original* modules.
            patch(
                "src.embedding.bge_embedder.BGEEmbedder",
                return_value=mock_embedder,
            ),
            patch(
                "src.vectorstore.qdrant_store.QdrantStore",
                return_value=mock_store,
            ),
        ):
            response = await async_client.post(
                "/lexreview/index",
                json={
                    "texts": sample_texts,
                    "metadatas": [{"source": "nda.pdf"}, {"source": "nda.pdf"}],
                    "collection_name": "contracts",
                },
                headers=_AUTH_HEADERS,
            )

        assert response.status_code == 200
        data = response.json()
        assert "indexed_count" in data
        assert "collection_name" in data
        assert "latency_ms" in data
        assert data["indexed_count"] == len(sample_texts)
        assert data["collection_name"] == "contracts"
        assert data["latency_ms"] >= 0.0


# ── POST /lexreview/search ────────────────────────────────────────────────────


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_endpoint(self, async_client: httpx.AsyncClient) -> None:
        """POST /lexreview/search: 200 with ``query``, ``results``, ``total_found``."""
        mock_results = [
            _make_search_result("chunk-001"),
            _make_search_result("chunk-002"),
        ]

        mock_embedder = MagicMock()
        mock_store = MagicMock()
        mock_dense = MagicMock()
        mock_sparse = MagicMock()
        mock_sparse.corpus_size = 0
        mock_hybrid = MagicMock()
        mock_hybrid.retrieve.return_value = mock_results

        with (
            # Lazy local imports — patch in the originating modules.
            patch("src.embedding.bge_embedder.BGEEmbedder", return_value=mock_embedder),
            patch("src.vectorstore.qdrant_store.QdrantStore", return_value=mock_store),
            patch("src.retrieval.dense_retriever.DenseRetriever", return_value=mock_dense),
            patch("src.retrieval.sparse_retriever.SparseRetriever", return_value=mock_sparse),
            patch("src.retrieval.hybrid_retriever.HybridRetriever", return_value=mock_hybrid),
        ):
            response = await async_client.post(
                "/lexreview/search",
                json={
                    "query": "indemnification clause",
                    "collection_name": "contracts",
                    "top_k": 5,
                    "rerank": False,
                },
                headers=_AUTH_HEADERS,
            )

        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total_found" in data
        assert "latency_ms" in data
        assert "reranked" in data
        assert data["query"] == "indemnification clause"
        assert data["total_found"] == 2
        assert len(data["results"]) == 2
        # Verify individual result shape.
        first = data["results"][0]
        assert "chunk_id" in first
        assert "content" in first
        assert "score" in first
        assert "metadata" in first
