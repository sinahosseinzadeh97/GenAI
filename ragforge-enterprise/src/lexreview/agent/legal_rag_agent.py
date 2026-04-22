"""LegalRAGAgent — the core retrieval-augmented generation pipeline.

Pipeline::

    query
      → HybridRetriever.retrieve(top_k=20)
      → CrossEncoderReranker.rerank(top_k=10)
      → build_prompt(query, top-10 chunks)
      → LLMClient.complete()
      → parse CoT + citations
      → AgentResponse

Typical usage::

    from src.lexreview.agent.legal_rag_agent import LegalRAGAgent

    agent = LegalRAGAgent(retriever=hybrid, reranker=reranker, llm=llm_client)
    response = agent.answer("What is the governing law clause?", collection_name="contracts")
"""

from __future__ import annotations

import re
import time
from typing import Any

from src.lexreview.agent.llm_client import LLMClient
from src.lexreview.agent.models import AgentResponse, Citation
from src.lexreview.agent.prompts import build_prompt
from src.retrieval.base import BaseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.logger import get_logger
from src.vectorstore.schema import SearchResult

log = get_logger(__name__)

# ── CoT section extractor ─────────────────────────────────────────────────────

_STEP_LABELS = ["[UNDERSTAND]", "[RETRIEVE]", "[REASON]", "[ANSWER]", "[CITE]"]
_STEP_RE = re.compile(
    r"\[(UNDERSTAND|RETRIEVE|REASON|ANSWER|CITE)\](.*?)(?=\[(?:UNDERSTAND|RETRIEVE|REASON|ANSWER|CITE)\]|$)",
    re.DOTALL,
)
_CHUNK_ID_RE = re.compile(r"\[CHUNK:\s*([^\]]+)\]")


def _parse_cot(
    raw: str,
    provider: str = "unknown",
) -> tuple[str, list[str], list[str]]:
    """Extract (answer, reasoning_steps, cited_chunk_ids) from a CoT response.

    Args:
        raw:      Raw LLM response string.
        provider: LLM provider name (forwarded from the caller for diagnostics).

    Returns:
        Tuple of (answer_text, reasoning_steps_list, chunk_id_list).
    """
    steps: list[str] = []
    answer = ""
    cited_ids: list[str] = []

    for match in _STEP_RE.finditer(raw):
        label = match.group(1)
        content = match.group(2).strip()
        if label == "ANSWER":
            answer = content
        elif label == "CITE":
            cited_ids = _CHUNK_ID_RE.findall(content)
        else:
            steps.append(f"[{label}] {content}")

    # Fallback: if no structured CoT found, use whole response as answer
    if not answer:
        # CoT tags absent — model ignored the prompt format.
        log.warning(
            "CoT format not detected in LLM output — falling back to raw answer; citations will be empty",
            extra={"raw_length": len(raw), "provider": provider},
        )
        answer = raw.strip()
        steps = []

    return answer, steps, cited_ids


def _compute_confidence(answer: str, reranked: list[SearchResult]) -> float:
    """Compute a heuristic confidence score in [0, 1] from raw cross-encoder logits.

    Note: the normalisation range [-10, 10] is calibrated for ms-marco-MiniLM models.
    Results may be misleading for other cross-encoder architectures.

    Args:
        answer:   Agent's synthesised answer string.
        reranked: Reranked retrieval results used for generation.

    Returns:
        Heuristic confidence in [0.0, 1.0]; not a calibrated probability.
    """
    if not answer or not reranked:
        return 0.0
    top_scores = [r.score for r in reranked[:3]]
    # Heuristic: assumes cross-encoder logits in [-10, 10]; calibration varies by model.
    normalised = [max(0.0, min(1.0, (s + 10) / 20)) for s in top_scores]
    return round(sum(normalised) / len(normalised), 4)


class LegalRAGAgent:
    """End-to-end legal RAG agent with Chain-of-Thought reasoning.

    Orchestrates retrieval → reranking → LLM generation and returns a
    structured :class:`~src.lexreview.agent.models.AgentResponse` with
    citations and CoT reasoning trace.

    Args:
        retriever:        A :class:`~src.retrieval.base.BaseRetriever` instance.
        reranker:         A :class:`~src.retrieval.reranker.CrossEncoderReranker`.
        llm:              A :class:`~src.lexreview.agent.llm_client.LLMClient`.
        retrieval_top_k:  Candidates fetched from the retriever (default 20).
        rerank_top_k:     Final passages sent to the LLM (default 10).

    Example::

        agent = LegalRAGAgent(retriever=hybrid, reranker=reranker, llm=client)
        resp = agent.answer("What are the payment terms?", collection_name="nda")
        print(resp.answer)
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: CrossEncoderReranker,
        llm: LLMClient,
        retrieval_top_k: int = 20,
        rerank_top_k: int = 10,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._retrieval_top_k = retrieval_top_k
        self._rerank_top_k = rerank_top_k

        log.info(
            "LegalRAGAgent initialised",
            extra={
                "retriever": type(retriever).__name__,
                "reranker": type(reranker).__name__,
                "retrieval_top_k": retrieval_top_k,
                "rerank_top_k": rerank_top_k,
            },
        )

    def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Run the full RAG pipeline for *query* and return an AgentResponse.

        Args:
            query:   Legal question from the user.
            filters: Optional Qdrant metadata filters forwarded to the retriever.

        Returns:
            :class:`~src.lexreview.agent.models.AgentResponse` with answer,
            citations, confidence, CoT steps, and latency.
        """
        t_start = time.perf_counter()

        # ── 1. Retrieval ──────────────────────────────────────────────────────
        log.info("LegalRAGAgent: retrieving", extra={"query": query[:100]})
        raw_results: list[SearchResult] = self._retriever.retrieve(
            query=query, top_k=self._retrieval_top_k, filters=filters
        )
        log.debug("Retrieved", extra={"count": len(raw_results)})

        # ── 2. Reranking ──────────────────────────────────────────────────────
        reranked: list[SearchResult] = self._reranker.rerank(
            query=query, results=raw_results, top_k=self._rerank_top_k
        )
        log.debug("Reranked", extra={"count": len(reranked)})

        # ── 3. Prompt assembly ────────────────────────────────────────────────
        messages = build_prompt(query=query, contexts=reranked)

        # ── 4. LLM generation ─────────────────────────────────────────────────
        raw_response = self._llm.complete(messages)

        # ── 5. Parse CoT ──────────────────────────────────────────────────────
        answer, reasoning_steps, cited_ids = _parse_cot(raw_response, provider=self._llm.provider)

        # ── 6. Build citations ────────────────────────────────────────────────
        result_map = {r.chunk_id: r for r in reranked}
        citations: list[Citation] = []
        for cid in dict.fromkeys(cited_ids):  # preserve order, dedupe
            if cid in result_map:
                r = result_map[cid]
                citations.append(
                    Citation(
                        chunk_id=r.chunk_id,
                        content=r.content,
                        score=r.score,
                        source=r.metadata.get("source"),
                    )
                )

        # ── 7. Confidence ─────────────────────────────────────────────────────
        confidence = _compute_confidence(answer, reranked)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        log.info(
            "LegalRAGAgent: answer generated",
            extra={
                "query": query[:80],
                "citations": len(citations),
                "confidence": confidence,
                "latency_ms": latency_ms,
            },
        )

        # ── 8. EU AI Act — low-confidence audit warning ───────────────────────
        # Art. 9 + Annex III: high-risk systems must log when output reliability
        # is below acceptable thresholds so that human reviewers are alerted.
        _AI_ACT_CONFIDENCE_THRESHOLD = 0.6
        if confidence < _AI_ACT_CONFIDENCE_THRESHOLD:
            log.warning(
                "EU AI Act: low-confidence legal output detected — human review required",
                extra={
                    "ai_act_risk": "low_confidence_legal_output",
                    "confidence": confidence,
                    "threshold": _AI_ACT_CONFIDENCE_THRESHOLD,
                    "citations": len(citations),
                },
            )

        return AgentResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            latency_ms=latency_ms,
        )
