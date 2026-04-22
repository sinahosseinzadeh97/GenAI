"""Tests for src/lexreview/api/ — FastAPI endpoints via TestClient."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.lexreview.agent.models import AgentResponse, Citation
from src.lexreview.extraction.models import Clause, LegalEntities


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient with the full FastAPI app."""
    from src.api.main import app

    return TestClient(app)


# ── POST /health ───────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


# ── POST /lexreview/extract ────────────────────────────────────────────────────


class TestExtractEndpoint:
    def test_extract_entities_and_clauses(self, client: TestClient) -> None:
        with (
            patch(
                "src.lexreview.api.router.LegalNER.extract",
                return_value=LegalEntities(parties=["Acme Corp."], dates=["2024-01-01"]),
            ),
            patch(
                "src.lexreview.api.router.RegexExtractor.extract",
                return_value=LegalEntities(amounts=["$50,000"]),
            ),
            patch(
                "src.lexreview.api.router.RegexExtractor.merge",
                return_value=LegalEntities(
                    parties=["Acme Corp."],
                    dates=["2024-01-01"],
                    amounts=["$50,000"],
                ),
            ),
            patch(
                "src.lexreview.api.router.ClauseDetector.detect",
                return_value=[
                    Clause(
                        type="indemnification",
                        text="Party A shall indemnify Party B.",
                        span=(0, 33),
                        confidence=0.85,
                    )
                ],
            ),
        ):
            response = client.post(
                "/lexreview/extract",
                json={
                    "text": "Party A shall indemnify Party B against all losses arising from breach.",
                    "extract_entities": True,
                    "detect_clauses": True,
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
        assert "clauses" in data
        assert data["char_count"] > 0

    def test_extract_text_too_short_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/lexreview/extract",
            json={"text": "Hi"},
        )
        assert response.status_code == 422

    def test_extract_entities_only(self, client: TestClient) -> None:
        with (
            patch(
                "src.lexreview.api.router.LegalNER.extract",
                return_value=LegalEntities(jurisdictions=["Delaware"]),
            ),
            patch(
                "src.lexreview.api.router.RegexExtractor.extract",
                return_value=LegalEntities(),
            ),
            patch(
                "src.lexreview.api.router.RegexExtractor.merge",
                return_value=LegalEntities(jurisdictions=["Delaware"]),
            ),
            patch(
                "src.lexreview.api.router.ClauseDetector.detect",
                return_value=[],
            ),
        ):
            response = client.post(
                "/lexreview/extract",
                json={"text": "Governed by the laws of Delaware under applicable statutes.", "detect_clauses": False},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["clauses"] == []

    def test_extract_ner_runtime_error_falls_back_to_regex(self, client: TestClient) -> None:
        """When spaCy is unavailable, extract should fall back to regex."""
        with (
            patch(
                "src.lexreview.api.router.LegalNER.extract",
                side_effect=RuntimeError("spaCy not installed"),
            ),
            patch(
                "src.lexreview.api.router.RegexExtractor.extract",
                return_value=LegalEntities(amounts=["$100"]),
            ),
            patch(
                "src.lexreview.api.router.ClauseDetector.detect",
                return_value=[],
            ),
        ):
            response = client.post(
                "/lexreview/extract",
                json={"text": "Payment of $100 is due upon signing of this agreement."},
            )
        assert response.status_code == 200


# ── POST /lexreview/query ─────────────────────────────────────────────────────


class TestQueryEndpoint:
    def _mock_agent_response(self) -> AgentResponse:
        return AgentResponse(
            answer="The governing law is Delaware.",
            citations=[
                Citation(
                    chunk_id="chunk-001",
                    content="Governed by Delaware law.",
                    score=0.92,
                    source="nda.pdf",
                )
            ],
            confidence=0.88,
            reasoning_steps=["[UNDERSTAND] Governing law question.", "[ANSWER] Delaware."],
            latency_ms=280.0,
        )

    @pytest.fixture()
    def client_with_mock_agent(self) -> TestClient:
        """TestClient with the agent dependency overridden."""
        from src.api.main import app
        from src.lexreview.api.router import get_agent

        mock_agent = MagicMock()
        mock_agent.answer.return_value = self._mock_agent_response()
        app.dependency_overrides[get_agent] = lambda: mock_agent
        test_client = TestClient(app)
        yield test_client
        app.dependency_overrides.pop(get_agent, None)

    def test_query_returns_200(self, client_with_mock_agent: TestClient) -> None:
        response = client_with_mock_agent.post(
            "/lexreview/query",
            json={
                "query": "What is the governing law?",
                "collection_name": "contracts",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data

    def test_query_too_short_returns_422(self, client_with_mock_agent: TestClient) -> None:
        response = client_with_mock_agent.post(
            "/lexreview/query",
            json={"query": "Hi", "collection_name": "contracts"},
        )
        assert response.status_code == 422

    def test_query_missing_collection_returns_422(self, client_with_mock_agent: TestClient) -> None:
        response = client_with_mock_agent.post(
            "/lexreview/query",
            json={"query": "What are the payment terms?"},
        )
        assert response.status_code == 422


# ── POST /lexreview/search ─────────────────────────────────────────────────────


class TestSearchEndpoint:
    def test_search_missing_collection_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/lexreview/search",
            json={"query": "payment terms"},
        )
        assert response.status_code == 422

    def test_search_invalid_top_k_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/lexreview/search",
            json={"query": "payment terms", "collection_name": "contracts", "top_k": 0},
        )
        assert response.status_code == 422


# ── POST /lexreview/index ─────────────────────────────────────────────────────


class TestIndexEndpoint:
    def test_index_missing_texts_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/lexreview/index",
            json={"texts": [], "collection_name": "contracts"},
        )
        assert response.status_code == 422

    def test_index_missing_collection_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/lexreview/index",
            json={"texts": ["Some contract text..."]},
        )
        assert response.status_code == 422
