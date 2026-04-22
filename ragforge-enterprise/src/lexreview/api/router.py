"""FastAPI router for LexReview endpoints.

Endpoints
---------
POST /lexreview/query              — LegalRAGAgent Q&A pipeline
POST /lexreview/extract            — spaCy NER + clause detection
POST /lexreview/index              — embed and index texts into Qdrant
POST /lexreview/search             — hybrid retrieval without agent

-- Phase 5 · Italian Market --
POST /lexreview/vigenza            — vigenza (norm validity) check
POST /lexreview/massima            — automatic massimario generation
POST /lexreview/contratto/analisi  — contract clause analyser (PDF upload)
POST /lexreview/231/risk-assessment — D.Lgs. 231/2001 compliance assessment

All heavy objects (agent, NER, qdrant client) are constructed lazily via
FastAPI ``Depends()`` factories so tests can substitute mocks cleanly.
"""

from __future__ import annotations

import datetime
import time
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from src.api.dependencies import verify_api_key
from src.config.settings import Settings, get_settings
from src.lexreview.agent.legal_rag_agent import LegalRAGAgent
from src.lexreview.agent.llm_client import LLMClient
from src.lexreview.api.schemas import (
    ContrattoAnalisiResponse,
    ClausolaAnalisiSchema,
    D231RiskRequest,
    D231RiskResponse,
    ExtractRequest,
    ExtractResponse,
    IndexRequest,
    IndexResponse,
    LexSearchRequest,
    LexSearchResponse,
    LexSearchResultItem,
    MassimaRequest,
    MassimaResponse,
    QueryRequest,
    QueryResponse,
    VigenzaRequest,
    VigenzaResponse,
)
from src.lexreview.extraction.clause_detector import ClauseDetector
from src.lexreview.extraction.models import LegalEntities
from src.lexreview.extraction.ner import LegalNER
from src.lexreview.extraction.regex_extractor import RegexExtractor
from src.utils.audit_log import get_audit_logger
from src.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/lexreview", tags=["lexreview"])


# ── Dependency factories ──────────────────────────────────────────────────────


def _get_settings() -> Settings:
    return get_settings()


def get_llm_client(settings: Annotated[Settings, Depends(_get_settings)]) -> LLMClient:
    return LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def get_ner(settings: Annotated[Settings, Depends(_get_settings)]) -> LegalNER:
    return LegalNER(model_name=settings.spacy_model)


def get_clause_detector() -> ClauseDetector:
    return ClauseDetector()


def get_regex_extractor() -> RegexExtractor:
    return RegexExtractor()


def _build_agent(
    settings: Settings,
    llm: LLMClient,
    collection_name: str,
    sparse: "SparseRetriever | None" = None,
) -> LegalRAGAgent:
    """Build the full agent pipeline lazily.

    Imports retrievers only at call time to avoid hard dependencies during
    module import (important when running without Qdrant available).

    Args:
        sparse: Pre-warmed :class:`SparseRetriever` loaded from Qdrant at
                startup.  When ``None`` (e.g. non-default collection), a fresh
                empty instance is used — acceptable for cold-start fallbacks.
    """
    try:
        from src.embedding.bge_embedder import BGEEmbedder
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import CrossEncoderReranker
        from src.retrieval.sparse_retriever import SparseRetriever
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=collection_name,
        )
        embedder = BGEEmbedder(model_name=settings.embedding_model)
        dense = DenseRetriever(embedder, store, collection_name)
        # Use the caller-supplied pre-warmed SparseRetriever when available.
        # For non-default collections (rare cold-start path) we fall back to
        # an empty index; hybrid RRF degrades gracefully to pure dense.
        if sparse is None:
            sparse = SparseRetriever(chunks=[])
        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        reranker = CrossEncoderReranker()
        return LegalRAGAgent(retriever=hybrid, reranker=reranker, llm=llm)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent initialisation failed: {exc}",
        ) from exc



def get_agent(
    settings: Annotated[Settings, Depends(_get_settings)],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> LegalRAGAgent:
    # collection_name resolved per-request in the endpoint, not here
    return _build_agent(settings, llm, collection_name=settings.qdrant_collection_name)


# ── POST /lexreview/query ─────────────────────────────────────────────────────


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Legal Q&A via RAG agent",
    description=(
        "Run a legal query through the full HybridRetriever → Reranker → LLM "
        "Chain-of-Thought pipeline and return a grounded answer with citations."
    ),
    dependencies=[Depends(verify_api_key)],
)
def query_endpoint(
    http_request: Request,
    request: QueryRequest,
    settings: Annotated[Settings, Depends(_get_settings)],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
    injected_agent: Annotated[LegalRAGAgent, Depends(get_agent)],
) -> QueryResponse:
    """POST /lexreview/query handler.

    Retrieves the shared ``LegalRAGAgent`` from ``app.state.agent`` (constructed
    once at worker startup) to avoid rebuilding the BGE embedding model and the
    full retrieval stack on every request.

    If the caller supplies a ``collection_name`` that differs from the default,
    a fresh agent is built for that collection so that routing is still correct.

    EU AI Act compliance (Annex III — administration of justice)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    An immutable audit record is written to ``logs/audit.jsonl`` for every
    query.  The raw query text is **never** logged; only its SHA-256 hash is
    stored alongside request ID, collection, model, and confidence score.
    """
    log.info("POST /lexreview/query", extra={"query": request.query[:80]})
    try:
        # Prefer the pre-warmed shared agent for the default collection.
        # Fall back to _build_agent() only when the caller requests a
        # non-default collection (rare; accepts the one-time cold-start cost).
        default_collection = settings.qdrant_collection_name
        # In production the lifespan always populates app.state.agent.
        # In test mode (no lifespan) fall back to injected_agent, which tests
        # can substitute via app.dependency_overrides[get_agent].
        state_agent = getattr(http_request.app.state, "agent", None)
        if request.collection_name == default_collection:
            # Default collection: prefer the pre-warmed state agent.
            agent: LegalRAGAgent = state_agent or injected_agent
        else:
            # Non-default collection: build on demand in production; use mock in tests.
            if state_agent is not None:
                log.info(
                    "Non-default collection requested — building agent on demand",
                    extra={"collection": request.collection_name},
                )
                agent = _build_agent(settings, llm, collection_name=request.collection_name)
            else:
                # Test / dev mode: no state agent → use the injected (possibly mocked) agent.
                agent = injected_agent

        response = agent.answer(query=request.query, filters=request.filters)
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Agent error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline failed: {exc}",
        ) from exc

    # ── EU AI Act audit trail (Art. 12 — record-keeping) ─────────────────────
    # Write an immutable record to logs/audit.jsonl.  Raw query is NEVER stored;
    # only its SHA-256 hash is persisted to satisfy GDPR data minimisation.
    request_id: str = getattr(http_request.state, "request_id", str(uuid.uuid4()))
    try:
        get_audit_logger().record(
            request_id=request_id,
            query=request.query,
            collection_name=request.collection_name,
            model=settings.llm_model,
            confidence=response.confidence,
        )
    except Exception as audit_exc:  # pragma: no cover — best-effort, never block the response
        log.error("Audit log write failed", extra={"error": str(audit_exc)})

    return QueryResponse(
        query=request.query,
        collection_name=request.collection_name,
        answer=response.answer,
        citations=response.citations,
        confidence=response.confidence,
        reasoning_steps=response.reasoning_steps,
        latency_ms=response.latency_ms,
    )


# ── POST /lexreview/extract ───────────────────────────────────────────────────


@router.post(
    "/extract",
    response_model=ExtractResponse,
    summary="Extract legal entities and clauses",
    description=(
        "Run spaCy NER + regex extraction and/or keyword-based clause detection "
        "on raw legal text."
    ),
    dependencies=[Depends(verify_api_key)],
)
def extract_endpoint(
    request: ExtractRequest,
    ner: Annotated[LegalNER, Depends(get_ner)],
    regex_ex: Annotated[RegexExtractor, Depends(get_regex_extractor)],
    clause_detector: Annotated[ClauseDetector, Depends(get_clause_detector)],
) -> ExtractResponse:
    """POST /lexreview/extract handler."""
    log.info("POST /lexreview/extract", extra={"text_len": len(request.text)})

    entities = LegalEntities()
    if request.extract_entities:
        try:
            spacy_entities = ner.extract(request.text)
            regex_entities = regex_ex.extract(request.text)
            entities = regex_ex.merge(spacy_entities, regex_entities)
        except RuntimeError as exc:
            # spaCy model not installed — fall back to regex only
            log.warning("LegalNER unavailable, using regex only", extra={"error": str(exc)})
            entities = regex_ex.extract(request.text)

    clauses = []
    if request.detect_clauses:
        detector = ClauseDetector(min_confidence=request.min_confidence)
        clauses = detector.detect(request.text)

    return ExtractResponse(
        entities=entities,
        clauses=clauses,
        char_count=len(request.text),
    )


# ── POST /lexreview/index ─────────────────────────────────────────────────────


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Embed and index texts into Qdrant",
    description=(
        "Embed a list of text strings and upsert them into the specified Qdrant collection."
    ),
    dependencies=[Depends(verify_api_key)],
)
def index_endpoint(
    request: IndexRequest,
    settings: Annotated[Settings, Depends(_get_settings)],
) -> IndexResponse:
    """POST /lexreview/index handler."""
    log.info(
        "POST /lexreview/index",
        extra={"texts": len(request.texts), "collection": request.collection_name},
    )
    t_start = time.perf_counter()

    try:
        from src.embedding.bge_embedder import BGEEmbedder
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=request.collection_name,
        )

        # ── Deduplication guard ───────────────────────────────────────────────
        # If the caller supplied a source_path, check whether it is already
        # represented in the collection.  When it is — and the caller did not
        # explicitly request a forced re-index — return immediately without
        # spending any compute on embedding.
        if request.source_path and not request.force_reindex:
            if store.source_exists(request.collection_name, request.source_path):
                latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
                log.info(
                    "Document already indexed — skipping embedding",
                    extra={
                        "source_path": request.source_path,
                        "collection": request.collection_name,
                    },
                )
                return IndexResponse(
                    indexed_count=0,
                    collection_name=request.collection_name,
                    latency_ms=latency_ms,
                    already_indexed=True,
                    source_path=request.source_path,
                )

        embedder = BGEEmbedder(model_name=settings.embedding_model)

        # Pad metadata if not provided
        metadatas: list[dict[str, Any]] = request.metadatas or [{}] * len(request.texts)
        if len(metadatas) < len(request.texts):
            metadatas += [{}] * (len(request.texts) - len(metadatas))

        embeddings = embedder.embed_batch(request.texts)

        from src.vectorstore.schema import IndexedChunk

        chunks = [
            IndexedChunk(
                # Deterministic ID: same document + page + position → same UUID.
                # This makes upsert idempotent: re-indexing the same file
                # overwrites existing points rather than creating duplicates.
                chunk_id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        f"{meta.get('source_path', '')}:{meta.get('page_number', 0)}:{chunk_index}",
                    )
                ),
                content=text,
                embedding=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                metadata=meta,
                indexed_at=datetime.datetime.utcnow(),
            )
            for chunk_index, (text, emb, meta) in enumerate(
                zip(request.texts, embeddings, metadatas, strict=True)
            )
        ]
        store.upsert_chunks(chunks)

    except Exception as exc:
        log.error("Index error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return IndexResponse(
        indexed_count=len(request.texts),
        collection_name=request.collection_name,
        latency_ms=latency_ms,
        already_indexed=False,
        source_path=request.source_path,
    )


# ── DELETE /lexreview/document ───────────────────────────────────────────────


@router.delete(
    "/document",
    summary="Delete all indexed chunks for a document",
    description=(
        "Remove every chunk whose payload ``source_path`` matches the supplied value "
        "from the specified Qdrant collection. "
        "Because chunk IDs are deterministic, re-indexing the same file after deletion "
        "is fully idempotent. "
        "This endpoint implements the GDPR **Right to Erasure** (Art. 17)."
    ),
    dependencies=[Depends(verify_api_key)],
)
def delete_document_endpoint(
    http_response: Response,
    settings: Annotated[Settings, Depends(_get_settings)],
    source_path: str = Query(..., description="source_path payload value to match and delete"),
    collection_name: str = Query(
        "",
        description="Qdrant collection name (defaults to the configured collection)",
    ),
) -> dict[str, object]:
    """DELETE /lexreview/document handler.

    Implements GDPR Art. 17 — Right to Erasure (\"Right to be Forgotten\").
    The ``X-GDPR-Article: 17`` response header documents the legal basis for
    this operation to support audit and compliance records.
    """
    resolved_collection = collection_name or settings.qdrant_collection_name
    log.info(
        "DELETE /lexreview/document",
        extra={"source_path": source_path, "collection": resolved_collection},
    )
    try:
        from src.vectorstore.qdrant_store import QdrantStore

        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=resolved_collection,
        )
        deleted = store.delete_chunks_by_source(resolved_collection, source_path)
    except Exception as exc:
        log.error("Delete document error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document deletion failed: {exc}",
        ) from exc

    # GDPR Art. 17 — Right to Erasure: document the legal basis in the response.
    http_response.headers["X-GDPR-Article"] = "17"

    return {"deleted_chunks": deleted, "source_path": source_path}


# ── POST /lexreview/search ────────────────────────────────────────────────────


@router.post(
    "/search",
    response_model=LexSearchResponse,
    summary="Hybrid retrieval search (no agent)",
    description=(
        "Run hybrid BM25 + dense retrieval with optional cross-encoder reranking, "
        "returning ranked chunks without agent generation."
    ),
    dependencies=[Depends(verify_api_key)],
)
def search_endpoint(
    http_request: Request,
    request: LexSearchRequest,
    settings: Annotated[Settings, Depends(_get_settings)],
) -> LexSearchResponse:
    """POST /lexreview/search handler.

    Reuses the shared ``BGEEmbedder`` (``app.state.embedder``) and
    ``CrossEncoderReranker`` (``app.state.reranker``) that were loaded once at
    worker startup.  Only ``QdrantStore``, ``DenseRetriever``, and
    ``HybridRetriever`` are constructed per-request because they depend on the
    caller-supplied ``collection_name``.
    """
    log.info("POST /lexreview/search", extra={"query": request.query[:80]})
    t_start = time.perf_counter()

    try:
        from src.embedding.bge_embedder import BGEEmbedder
        from src.retrieval.dense_retriever import DenseRetriever
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.retrieval.reranker import CrossEncoderReranker
        from src.retrieval.sparse_retriever import SparseRetriever
        from src.vectorstore.qdrant_store import QdrantStore

        # Reuse the pre-warmed embedder, reranker, and SparseRetriever from app.state
        # when available (production path).  Fall back to constructing fresh instances
        # when app.state was not populated by the lifespan (e.g. in async test clients).
        embedder = getattr(http_request.app.state, "embedder", None) or BGEEmbedder(
            model_name=settings.embedding_model
        )
        reranker = getattr(http_request.app.state, "reranker", None) or CrossEncoderReranker()
        sparse: SparseRetriever = getattr(
            http_request.app.state, "sparse", None
        ) or SparseRetriever(chunks=[])

        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=request.collection_name,
        )
        dense = DenseRetriever(embedder, store, request.collection_name)
        # Use the shared SparseRetriever populated from Qdrant at startup.
        # For non-default collections this index won't match, but hybrid RRF
        # degrades gracefully to pure dense retrieval in that case.
        hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)
        results = hybrid.retrieve(
            query=request.query, top_k=request.top_k, filters=request.filters
        )

        if request.rerank and results:
            results = reranker.rerank(query=request.query, results=results, top_k=request.top_k)

    except Exception as exc:
        log.error("Search error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return LexSearchResponse(
        query=request.query,
        results=[
            LexSearchResultItem(
                chunk_id=r.chunk_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ],
        total_found=len(results),
        latency_ms=latency_ms,
        reranked=request.rerank,
    )


# ── POST /lexreview/vigenza ───────────────────────────────────────────────────


@router.post(
    "/vigenza",
    response_model=VigenzaResponse,
    summary="Vigenza check — verifica validità di una norma",
    description=(
        "Verifica se una norma italiana era in vigore alla data di riferimento indicata. "
        "Interroga prima la knowledge base Qdrant per la verifica deterministca tramite "
        "metadati strutturati (it_data_vigenza / it_data_abrogazione); se non disponibili, "
        "esegue una chiamata LLM guidata dal catalogo normativo."
    ),
    dependencies=[Depends(verify_api_key)],
    tags=["lexreview", "italia"],
)
def vigenza_endpoint(
    request: VigenzaRequest,
    settings: Annotated[Settings, Depends(_get_settings)],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> VigenzaResponse:
    """POST /lexreview/vigenza handler.

    Checks whether the cited Italian norm was in force on *data_riferimento*.
    The determination uses Qdrant-retrieved structured metadata when available,
    falling back to a guided LLM call otherwise.
    """
    import datetime as _dt

    from src.italia.vigenza import check_vigenza

    log.info(
        "POST /lexreview/vigenza",
        extra={"norma": request.norma, "data_riferimento": request.data_riferimento},
    )
    t_start = time.perf_counter()

    try:
        ref_date = _dt.date.fromisoformat(request.data_riferimento)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"data_riferimento non valida: {exc}",
        ) from exc

    try:
        from src.vectorstore.qdrant_store import QdrantStore
        from src.embedding.bge_embedder import BGEEmbedder
        context_chunks: list[dict] = []
        try:
            _store = QdrantStore(host=settings.qdrant_host, port=settings.qdrant_port, collection_name=settings.qdrant_collection_name)
            _emb = getattr(http_request.app.state, 'embedder', None) or BGEEmbedder(model_name=settings.embedding_model)
            _results = _store.search(query_vector=_emb.embed_single(request.norma), top_k=5)
            context_chunks = [{'content': r.content, 'metadata': r.metadata} for r in _results]
        except Exception:
            pass
        result = check_vigenza(
            norma=request.norma,
            data_riferimento=ref_date,
            llm=llm,
            context_chunks=context_chunks if context_chunks else None,
        )
    except Exception as exc:
        log.error("Vigenza check error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verifica vigenza fallita: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return VigenzaResponse(
        vigente=result.vigente,
        data_entrata_vigore=(
            result.data_entrata_vigore.isoformat() if result.data_entrata_vigore else None
        ),
        data_abrogazione=(
            result.data_abrogazione.isoformat() if result.data_abrogazione else None
        ),
        modificata_da=result.modificata_da,
        testo_vigente=result.testo_vigente,
        fonte=result.fonte,
        latency_ms=latency_ms,
    )


# ── POST /lexreview/massima ───────────────────────────────────────────────────


@router.post(
    "/massima",
    response_model=MassimaResponse,
    summary="Massimario automatico — generazione automatica di massima",
    description=(
        "Genera automaticamente la massima ufficiale (≤ 150 parole), il principio di diritto, "
        "le parole chiave e la classificazione per materia di una sentenza italiana."
    ),
    dependencies=[Depends(verify_api_key)],
    tags=["lexreview", "italia"],
)
def massima_endpoint(
    request: MassimaRequest,
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> MassimaResponse:
    """POST /lexreview/massima handler.

    Generates a structured massima from the supplied sentenza text using the
    LLM configured in :class:`~src.config.settings.Settings`.
    """
    from src.italia.massimario import generate_massima

    log.info(
        "POST /lexreview/massima",
        extra={"testo_len": len(request.testo_sentenza)},
    )
    t_start = time.perf_counter()

    try:
        result = generate_massima(sentenza_text=request.testo_sentenza, llm=llm)
    except Exception as exc:
        log.error("Massima generation error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generazione massima fallita: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return MassimaResponse(
        massima_ufficiale=result.massima_ufficiale,
        principio_di_diritto=result.principio_di_diritto,
        parole_chiave=result.parole_chiave,
        classificazione_materia=result.classificazione_materia,
        latency_ms=latency_ms,
    )


# ── POST /lexreview/contratto/analisi ────────────────────────────────────────

from fastapi import File, UploadFile  # noqa: PLC0415


@router.post(
    "/contratto/analisi",
    response_model=ContrattoAnalisiResponse,
    summary="Analisi clausole contrattuali — vessatorie e nulle",
    description=(
        "Analizza un contratto in formato PDF e identifica clausole vessatorie "
        "(Art. 33-38 Codice del Consumo), clausole nulle (Art. 1418 c.c.), con "
        "risk score per clausola (🔴/🟠/🟡/🟢) e correzioni suggerite in italiano."
    ),
    dependencies=[Depends(verify_api_key)],
    tags=["lexreview", "italia"],
)
async def contratto_analisi_endpoint(
    settings: Annotated[Settings, Depends(_get_settings)],
    llm: Annotated[LLMClient, Depends(get_llm_client)],
    file: UploadFile = File(..., description="Contratto in formato PDF."),
) -> ContrattoAnalisiResponse:
    """POST /lexreview/contratto/analisi handler.

    Accepts a contract PDF via multipart/form-data, extracts plain text with
    pdfplumber, then analyses clauses via
    :func:`~src.italia.contratto_analyzer.analyze_contract`.

    The ``python-multipart`` package must be installed for FastAPI to parse the
    multipart body (already a core dependency for any FastAPI form-upload app).
    """
    from src.italia.contratto_analyzer import ClausolaAnalisi, analyze_contract  # noqa: PLC0415

    t_start = time.perf_counter()

    if not file.content_type or "pdf" not in file.content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Solo file PDF sono accettati (Content-Type: application/pdf).",
        )

    try:
        import io  # noqa: PLC0415

        import pdfplumber  # type: ignore[import-untyped]  # noqa: PLC0415

        raw_bytes = await file.read()
        with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
        contract_text = "\n\n".join(pages_text).strip()
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="pdfplumber non disponibile. Installare con: pip install pdfplumber",
        ) from exc
    except Exception as exc:
        log.error("PDF extraction error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Impossibile estrarre testo dal PDF: {exc}",
        ) from exc

    if not contract_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Il PDF non contiene testo estraibile (potrebbe essere un PDF scansionato).",
        )

    log.info(
        "POST /lexreview/contratto/analisi",
        extra={"file_name": file.filename, "text_len": len(contract_text)},
    )

    try:
        result = analyze_contract(contract_text=contract_text, llm=llm)
    except Exception as exc:
        log.error("Contract analysis error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analisi contratto fallita: {exc}",
        ) from exc

    def _clausola_to_schema(c: ClausolaAnalisi) -> ClausolaAnalisiSchema:
        return ClausolaAnalisiSchema(
            testo_clausola=c.testo_clausola,
            tipo=c.tipo,
            riferimento_normativo=c.riferimento_normativo,
            motivazione=c.motivazione,
            risk_score=c.risk_score.value,
            correzione_suggerita=c.correzione_suggerita,
        )

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return ContrattoAnalisiResponse(
        clausole_vessatorie=[_clausola_to_schema(c) for c in result.clausole_vessatorie],
        clausole_nulle=[_clausola_to_schema(c) for c in result.clausole_nulle],
        risk_score_globale=result.risk_score_globale.value,
        sommario=result.sommario,
        latency_ms=latency_ms,
    )


# ── POST /lexreview/231/risk-assessment ──────────────────────────────────────


@router.post(
    "/231/risk-assessment",
    response_model=D231RiskResponse,
    summary="231 Compliance — valutazione del rischio D.Lgs. 231/2001",
    description=(
        "Valuta il rischio di responsabilità degli enti ai sensi del D.Lgs. 231/2001, "
        "identificando i reati presupposto applicabili al settore e alle attività dell'ente, "
        "e formulando raccomandazioni operative per l'Organismo di Vigilanza (ODV)."
    ),
    dependencies=[Depends(verify_api_key)],
    tags=["lexreview", "italia"],
)
def d231_risk_assessment_endpoint(
    request: D231RiskRequest,
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> D231RiskResponse:
    """POST /lexreview/231/risk-assessment handler.

    Performs a D.Lgs. 231/2001 risk assessment for the described entity via
    :func:`~src.italia.d231_compliance.assess_231_risk`.
    """
    from src.italia.d231_compliance import assess_231_risk  # noqa: PLC0415

    log.info(
        "POST /lexreview/231/risk-assessment",
        extra={"settore": request.settore},
    )
    t_start = time.perf_counter()

    try:
        result = assess_231_risk(
            settore=request.settore,
            descrizione_attivita=request.descrizione_attivita,
            llm=llm,
        )
    except Exception as exc:
        log.error("231 risk assessment error", extra={"error": str(exc)})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Valutazione rischio 231 fallita: {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return D231RiskResponse(
        reati_presupposto=result.reati_presupposto,
        odv_raccomandazioni=result.odv_raccomandazioni,
        risk_score=result.risk_score,
        riferimenti_normativi=result.riferimenti_normativi,
        sintesi=result.sintesi,
        latency_ms=latency_ms,
    )
