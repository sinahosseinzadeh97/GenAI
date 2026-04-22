"""ItalianLegalRAGAgent — pipeline RAG specializzata per il diritto italiano.

Pipeline::

    query
      → HybridRetriever.retrieve(top_k=20)
      → CrossEncoderReranker.rerank(top_k=10)
      → build_prompt_it(query, top-10 chunks)
      → LLMClient.complete()   [provider="anthropic", model="claude-sonnet-4-5"]
      → _parse_cot_it()        [parse 7 sezioni italiane]
      → ItalianAgentResponse

Formato CoT atteso dal modello::

    [COMPRENSIONE]      Riassunto questione giuridica
    [NORME_APPLICABILI] Norme di riferimento
    [GIURISPRUDENZA]    Sentenze rilevanti
    [RAGIONAMENTO]      Ragionamento sillogistico
    [RISPOSTA]          Conclusione giuridica
    [FONTI]             Fonti citate (include [CHUNK: <id>])
    [AVVERTENZE]        Limitazioni e disclaimer professionale

Typical usage::

    from src.italia.agent import ItalianLegalRAGAgent
    from src.lexreview.agent.llm_client import LLMClient

    llm = LLMClient(
        provider="anthropic",
        model="claude-sonnet-4-5",
        max_tokens=4096,
    )
    agent = ItalianLegalRAGAgent(
        retriever=hybrid,
        reranker=reranker,
        llm=llm,
    )
    response = agent.answer(
        "Quali sono i presupposti della responsabilità ex art. 2043 c.c.?"
    )
    print(response.risposta)
"""

from __future__ import annotations

import re
import time
from typing import Any

from src.italia.agent.models_it import ItalianAgentResponse, ItalianCitation
from src.italia.agent.prompts_it import build_prompt_it
from src.lexreview.agent.llm_client import LLMClient
from src.retrieval.base import BaseRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.utils.logger import get_logger
from src.vectorstore.schema import SearchResult

log = get_logger(__name__)

# ── Production LLM defaults ──────────────────────────────────────────────────

#: Provider da usare in produzione per RAGForge Italia.
ITALIA_PROVIDER: str = "anthropic"

#: Modello Claude consigliato per la produzione italiana.
ITALIA_MODEL: str = "claude-sonnet-4-5"

# ── CoT section extractor (italiano) ─────────────────────────────────────────

# I 7 tag italiani nel loro ordine canonico.
_IT_LABELS = (
    "COMPRENSIONE",
    "NORME_APPLICABILI",
    "GIURISPRUDENZA",
    "RAGIONAMENTO",
    "RISPOSTA",
    "FONTI",
    "AVVERTENZE",
)

# Pattern che cattura ciascun blocco tra i tag italiano.
# Usa lookahead per staccarsi al prossimo tag o fine stringa.
_IT_STEP_RE = re.compile(
    r"\[("
    + "|".join(re.escape(lbl) for lbl in _IT_LABELS)
    + r")\](.*?)(?=\[(?:"
    + "|".join(re.escape(lbl) for lbl in _IT_LABELS)
    + r")\]|$)",
    re.DOTALL,
)

# Estrae gli ID dei chunk dalla sezione [FONTI].
_CHUNK_ID_RE = re.compile(r"\[CHUNK:\s*([^\]]+)\]")


def _parse_cot_it(
    raw: str,
    provider: str = "unknown",
) -> tuple[str, dict[str, str], list[str]]:
    """Estrae le sezioni CoT italiane da una risposta grezza del modello.

    Analizza il testo grezzo cercando i sette tag italiani e restituisce
    un dizionario di sezioni indicizzato per nome tag.

    Args:
        raw:      Stringa grezza restituita dall'LLM.
        provider: Nome del provider LLM (usato per diagnostica).

    Returns:
        Tupla ``(risposta, sezioni, chunk_ids)`` dove:

        - ``risposta`` è il testo della sezione ``[RISPOSTA]``
          (o l'intera risposta se il formato non è rilevato);
        - ``sezioni`` è un ``dict`` con tutte le sezioni CoT italiane,
          indicizzate in minuscolo (es. ``"comprensione"``, ``"ragionamento"``);
        - ``chunk_ids`` è la lista degli ID chunk estratti da ``[FONTI]``.
    """
    sezioni: dict[str, str] = {}
    risposta = ""
    chunk_ids: list[str] = []

    for match in _IT_STEP_RE.finditer(raw):
        label = match.group(1)      # es. "RISPOSTA"
        content = match.group(2).strip()
        sezioni[label.lower()] = content

        if label == "RISPOSTA":
            risposta = content
        elif label == "FONTI":
            chunk_ids = _CHUNK_ID_RE.findall(content)

    if not risposta:
        # Il modello ha ignorato il formato CoT italiano.
        log.warning(
            "Formato CoT italiano non rilevato — fallback alla risposta grezza; "
            "le citazioni saranno vuote.",
            extra={"raw_length": len(raw), "provider": provider},
        )
        risposta = raw.strip()
        sezioni = {}

    return risposta, sezioni, chunk_ids


def _compute_confidence_it(risposta: str, reranked: list[SearchResult]) -> float:
    """Calcola un punteggio euristico di confidenza in [0, 1].

    Si basa sui logit del cross-encoder normalizzati nell'intervallo [-10, 10],
    calibrato per modelli ms-marco-MiniLM. Risultati diversi con altri modelli.

    Args:
        risposta: Testo della conclusione giuridica generata.
        reranked: Risultati reranked utilizzati per la generazione.

    Returns:
        Confidenza euristica in [0.0, 1.0]; non è una probabilità calibrata.
    """
    if not risposta or not reranked:
        return 0.0
    top_scores = [r.score for r in reranked[:3]]
    normalised = [max(0.0, min(1.0, (s + 10) / 20)) for s in top_scores]
    return round(sum(normalised) / len(normalised), 4)


class ItalianLegalRAGAgent:
    """Pipeline RAG end-to-end per il diritto italiano con Chain-of-Thought.

    In produzione utilizza **Claude 3.5 Sonnet (Anthropic)** come LLM,
    il prompt di sistema ``ITALIAN_LEGAL_SYSTEM_PROMPT`` ottimizzato per
    il diritto italiano, e un parser CoT a sette sezioni.

    Args:
        retriever:        Istanza di :class:`~src.retrieval.base.BaseRetriever`.
        reranker:         Istanza di :class:`~src.retrieval.reranker.CrossEncoderReranker`.
        llm:              Istanza di :class:`~src.lexreview.agent.llm_client.LLMClient`.
                          Si raccomanda ``provider="anthropic"`` e
                          ``model="claude-sonnet-4-5"`` per la produzione.
        retrieval_top_k:  Candidati recuperati dal retriever (default 20).
        rerank_top_k:     Passaggi finali inviati al modello (default 10).

    Example::

        from src.lexreview.agent.llm_client import LLMClient
        from src.italia.agent import ItalianLegalRAGAgent

        llm = LLMClient(
            provider="anthropic",
            model="claude-sonnet-4-5",
            max_tokens=4096,
        )
        agent = ItalianLegalRAGAgent(retriever=hybrid, reranker=reranker, llm=llm)
        resp = agent.answer("Cos'è la responsabilità precontrattuale?")
        print(resp.risposta)
        print(resp.norme_applicabili)
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
            "ItalianLegalRAGAgent inizializzato",
            extra={
                "retriever": type(retriever).__name__,
                "reranker": type(reranker).__name__,
                "llm_provider": llm.provider,
                "llm_model": llm.model,
                "retrieval_top_k": retrieval_top_k,
                "rerank_top_k": rerank_top_k,
            },
        )

    def answer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> ItalianAgentResponse:
        """Esegue la pipeline RAG italiana e restituisce una risposta strutturata.

        Args:
            query:   Domanda giuridica dell'utente (in italiano).
            filters: Filtri opzionali sui metadati Qdrant (es. tipo_documento,
                     anno_pubblicazione) forwarded al retriever.

        Returns:
            :class:`~src.italia.agent.models_it.ItalianAgentResponse` con
            risposta, citazioni, sezioni CoT, confidenza e latenza.
        """
        t_start = time.perf_counter()

        # ── 1. Recupero ───────────────────────────────────────────────────────
        log.info("ItalianLegalRAGAgent: recupero in corso", extra={"query": query[:100]})
        raw_results: list[SearchResult] = self._retriever.retrieve(
            query=query, top_k=self._retrieval_top_k, filters=filters
        )
        log.debug("Recuperati", extra={"count": len(raw_results)})

        # ── 2. Reranking ──────────────────────────────────────────────────────
        reranked: list[SearchResult] = self._reranker.rerank(
            query=query, results=raw_results, top_k=self._rerank_top_k
        )
        log.debug("Ordinati per rilevanza", extra={"count": len(reranked)})

        # ── 3. Costruzione del prompt italiano ──────────────────────────────
        messages = build_prompt_it(query=query, contexts=reranked)

        # ── 4. Generazione LLM (Claude 3.5 Sonnet in produzione) ────────────
        raw_response = self._llm.complete(messages)

        # ── 5. Parsing del CoT italiano (7 sezioni) ──────────────────────────
        risposta, sezioni, cited_ids = _parse_cot_it(
            raw_response, provider=self._llm.provider
        )

        # ── 6. Costruzione delle citazioni ────────────────────────────────────
        result_map = {r.chunk_id: r for r in reranked}
        citations: list[ItalianCitation] = []
        for cid in dict.fromkeys(cited_ids):  # preserva ordine, deduplicazione
            if cid in result_map:
                r = result_map[cid]
                citations.append(
                    ItalianCitation(
                        chunk_id=r.chunk_id,
                        content=r.content,
                        score=r.score,
                        source=r.metadata.get("source"),
                        riferimento_normativo=r.metadata.get("riferimento_normativo"),
                        tipo_documento=r.metadata.get("tipo_documento"),
                    )
                )

        # ── 7. Confidenza e latenza ───────────────────────────────────────────
        confidence = _compute_confidence_it(risposta, reranked)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        log.info(
            "ItalianLegalRAGAgent: risposta generata",
            extra={
                "query": query[:80],
                "citazioni": len(citations),
                "confidenza": confidence,
                "latency_ms": latency_ms,
                "llm_provider": self._llm.provider,
                "llm_model": self._llm.model,
            },
        )

        return ItalianAgentResponse(
            risposta=risposta,
            citations=citations,
            confidence=confidence,
            comprensione=sezioni.get("comprensione", ""),
            norme_applicabili=sezioni.get("norme_applicabili", ""),
            giurisprudenza=sezioni.get("giurisprudenza", ""),
            ragionamento=sezioni.get("ragionamento", ""),
            fonti_raw=sezioni.get("fonti", ""),
            avvertenze=sezioni.get("avvertenze", ""),
            latency_ms=latency_ms,
        )

    async def aanswer(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> ItalianAgentResponse:
        """Versione asincrona di :meth:`answer` per l'uso nelle API FastAPI.

        Identica a :meth:`answer` ma usa ``LLMClient.acomplete()`` per non
        bloccare l'event-loop di FastAPI durante la chiamata al modello.

        Args:
            query:   Domanda giuridica dell'utente.
            filters: Filtri opzionali sui metadati Qdrant.

        Returns:
            :class:`~src.italia.agent.models_it.ItalianAgentResponse`.
        """
        t_start = time.perf_counter()

        # ── 1. Recupero ───────────────────────────────────────────────────────
        log.info(
            "ItalianLegalRAGAgent (async): recupero in corso",
            extra={"query": query[:100]},
        )
        raw_results: list[SearchResult] = self._retriever.retrieve(
            query=query, top_k=self._retrieval_top_k, filters=filters
        )

        # ── 2. Reranking ──────────────────────────────────────────────────────
        reranked: list[SearchResult] = self._reranker.rerank(
            query=query, results=raw_results, top_k=self._rerank_top_k
        )

        # ── 3. Costruzione del prompt italiano ──────────────────────────────
        messages = build_prompt_it(query=query, contexts=reranked)

        # ── 4. Generazione asincrona ──────────────────────────────────────────
        raw_response = await self._llm.acomplete(messages)

        # ── 5. Parsing CoT italiano ───────────────────────────────────────────
        risposta, sezioni, cited_ids = _parse_cot_it(
            raw_response, provider=self._llm.provider
        )

        # ── 6. Citazioni ──────────────────────────────────────────────────────
        result_map = {r.chunk_id: r for r in reranked}
        citations: list[ItalianCitation] = []
        for cid in dict.fromkeys(cited_ids):
            if cid in result_map:
                r = result_map[cid]
                citations.append(
                    ItalianCitation(
                        chunk_id=r.chunk_id,
                        content=r.content,
                        score=r.score,
                        source=r.metadata.get("source"),
                        riferimento_normativo=r.metadata.get("riferimento_normativo"),
                        tipo_documento=r.metadata.get("tipo_documento"),
                    )
                )

        confidence = _compute_confidence_it(risposta, reranked)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)

        log.info(
            "ItalianLegalRAGAgent (async): risposta generata",
            extra={
                "query": query[:80],
                "citazioni": len(citations),
                "confidenza": confidence,
                "latency_ms": latency_ms,
            },
        )

        return ItalianAgentResponse(
            risposta=risposta,
            citations=citations,
            confidence=confidence,
            comprensione=sezioni.get("comprensione", ""),
            norme_applicabili=sezioni.get("norme_applicabili", ""),
            giurisprudenza=sezioni.get("giurisprudenza", ""),
            ragionamento=sezioni.get("ragionamento", ""),
            fonti_raw=sezioni.get("fonti", ""),
            avvertenze=sezioni.get("avvertenze", ""),
            latency_ms=latency_ms,
        )
