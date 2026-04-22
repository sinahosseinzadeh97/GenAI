"""Tests for src/lexreview/agent/ — LegalRAGAgent, LLMClient, prompts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.lexreview.agent.models import AgentResponse, Citation
from src.lexreview.agent.prompts import SYSTEM_PROMPT, build_prompt
from src.vectorstore.schema import SearchResult

# ── SearchResult factory ───────────────────────────────────────────────────────


def _make_result(chunk_id: str, content: str = "sample content", score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        content=content,
        score=score,
        metadata={"source": "test.pdf"},
        rank=1,
    )


# ── Prompt tests ───────────────────────────────────────────────────────────────


class TestPrompts:
    def test_system_prompt_not_empty(self) -> None:
        assert len(SYSTEM_PROMPT) > 100

    def test_build_prompt_returns_two_messages(self) -> None:
        messages = build_prompt("What is the payment term?", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_prompt_includes_query(self) -> None:
        query = "What is the governing law?"
        messages = build_prompt(query, [])
        assert query in messages[1]["content"]

    def test_build_prompt_includes_chunk_ids(self) -> None:
        results = [_make_result("chunk-abc"), _make_result("chunk-xyz")]
        messages = build_prompt("Test query", results)
        user_content = messages[1]["content"]
        assert "chunk-abc" in user_content
        assert "chunk-xyz" in user_content

    def test_build_prompt_no_context_placeholder(self) -> None:
        messages = build_prompt("Test?", [])
        assert "No context retrieved" in messages[1]["content"]

    def test_build_prompt_includes_source(self) -> None:
        result = _make_result("c1", content="text", score=0.8)
        result.metadata["source"] = "nda.pdf"
        messages = build_prompt("q", [result])
        assert "nda.pdf" in messages[1]["content"]


# ── AgentResponse model tests ─────────────────────────────────────────────────


class TestAgentResponseModel:
    def test_default_values(self) -> None:
        resp = AgentResponse(answer="The answer is X.")
        assert resp.confidence == 0.0
        assert resp.citations == []
        assert resp.reasoning_steps == []
        assert resp.latency_ms == 0.0

    def test_with_citations(self) -> None:
        c = Citation(chunk_id="c1", content="Some text", score=0.9)
        resp = AgentResponse(answer="Answer", citations=[c])
        assert resp.citations[0].chunk_id == "c1"

    def test_confidence_bounds(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            AgentResponse(answer="x", confidence=1.5)

    def test_citation_source_optional(self) -> None:
        c = Citation(chunk_id="c1", content="text", score=0.5)
        assert c.source is None


# ── LegalRAGAgent tests (fully mocked) ────────────────────────────────────────


class TestLegalRAGAgent:
    @pytest.fixture()
    def mock_retriever(self) -> MagicMock:
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            _make_result("chunk-001", content="Party A shall indemnify Party B."),
            _make_result("chunk-002", content="Governed by Delaware law."),
        ]
        return retriever

    @pytest.fixture()
    def mock_reranker(self) -> MagicMock:
        reranker = MagicMock()
        reranker.rerank.return_value = [
            _make_result("chunk-001", content="Party A shall indemnify Party B.", score=0.95),
        ]
        return reranker

    @pytest.fixture()
    def mock_llm(self) -> MagicMock:
        llm = MagicMock()
        llm.complete.return_value = (
            "[UNDERSTAND] The user asks about indemnification.\n"
            "[RETRIEVE] Chunk chunk-001 is most relevant.\n"
            "[REASON] Chunk chunk-001 directly addresses indemnification obligations.\n"
            "[ANSWER] Party A shall indemnify Party B against all losses.\n"
            "[CITE] [CHUNK: chunk-001]"
        )
        return llm

    def test_answer_returns_agent_response(
        self,
        mock_retriever: MagicMock,
        mock_reranker: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        from src.lexreview.agent.legal_rag_agent import LegalRAGAgent

        agent = LegalRAGAgent(
            retriever=mock_retriever,
            reranker=mock_reranker,
            llm=mock_llm,
        )
        response = agent.answer("What is the indemnification clause?")
        assert isinstance(response, AgentResponse)

    def test_answer_includes_answer_text(
        self,
        mock_retriever: MagicMock,
        mock_reranker: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        from src.lexreview.agent.legal_rag_agent import LegalRAGAgent

        agent = LegalRAGAgent(
            retriever=mock_retriever,
            reranker=mock_reranker,
            llm=mock_llm,
        )
        response = agent.answer("What is the indemnification clause?")
        assert "indemnify" in response.answer.lower()

    def test_answer_cot_steps_extracted(
        self,
        mock_retriever: MagicMock,
        mock_reranker: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        from src.lexreview.agent.legal_rag_agent import LegalRAGAgent

        agent = LegalRAGAgent(
            retriever=mock_retriever,
            reranker=mock_reranker,
            llm=mock_llm,
        )
        response = agent.answer("What is the indemnification clause?")
        assert len(response.reasoning_steps) >= 1
        assert any("UNDERSTAND" in s or "REASON" in s for s in response.reasoning_steps)

    def test_citations_mapped_from_chunk_ids(
        self,
        mock_retriever: MagicMock,
        mock_reranker: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        from src.lexreview.agent.legal_rag_agent import LegalRAGAgent

        agent = LegalRAGAgent(
            retriever=mock_retriever,
            reranker=mock_reranker,
            llm=mock_llm,
        )
        response = agent.answer("What is the indemnification clause?")
        cited_ids = [c.chunk_id for c in response.citations]
        assert "chunk-001" in cited_ids

    def test_latency_ms_positive(
        self,
        mock_retriever: MagicMock,
        mock_reranker: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        from src.lexreview.agent.legal_rag_agent import LegalRAGAgent

        agent = LegalRAGAgent(
            retriever=mock_retriever,
            reranker=mock_reranker,
            llm=mock_llm,
        )
        response = agent.answer("Any query?")
        assert response.latency_ms >= 0.0

    def test_retriever_called_with_correct_top_k(
        self,
        mock_retriever: MagicMock,
        mock_reranker: MagicMock,
        mock_llm: MagicMock,
    ) -> None:
        from src.lexreview.agent.legal_rag_agent import LegalRAGAgent

        agent = LegalRAGAgent(
            retriever=mock_retriever,
            reranker=mock_reranker,
            llm=mock_llm,
            retrieval_top_k=15,
        )
        agent.answer("Query")
        mock_retriever.retrieve.assert_called_once_with(
            query="Query", top_k=15, filters=None
        )

    def test_no_cot_structure_fallback_to_full_response(
        self,
        mock_retriever: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        from src.lexreview.agent.legal_rag_agent import LegalRAGAgent

        plain_llm = MagicMock()
        plain_llm.complete.return_value = "This is a plain answer without CoT."

        agent = LegalRAGAgent(
            retriever=mock_retriever,
            reranker=mock_reranker,
            llm=plain_llm,
        )
        response = agent.answer("Query?")
        assert "plain answer" in response.answer
        assert response.reasoning_steps == []


# ── LLMClient tests (mocked openai) ───────────────────────────────────────────


class TestLLMClient:
    def test_complete_returns_string(self) -> None:
        """LLMClient.complete() returns the content string from the mocked SDK response.

        We patch ``_sync_client`` directly on the already-constructed instance
        because ``openai.OpenAI`` is resolved at ``__init__`` time, so patching
        the constructor has no effect after the object exists.
        """
        from src.lexreview.agent.llm_client import LLMClient

        # Construct with an explicit fake key so no real network call is made.
        client = LLMClient(base_url="http://localhost", model="test-model", api_key="sk-test")

        # Replace the internal sync client with a mock — this is what complete() calls.
        mock_sync = MagicMock()
        mock_sync.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Test response"))]
        )
        client._sync_client = mock_sync

        result = client.complete([{"role": "user", "content": "Hello"}])
        assert result == "Test response"
        mock_sync.chat.completions.create.assert_called_once()

    def test_model_property(self) -> None:
        from src.lexreview.agent.llm_client import LLMClient

        client = LLMClient(model="gpt-4o", base_url="http://x", api_key="k")
        assert client.model == "gpt-4o"
