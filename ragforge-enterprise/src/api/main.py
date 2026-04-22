"""FastAPI application entry point for RAGForge Enterprise + LexReview.

Run locally::

    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

Swagger UI:  http://localhost:8000/docs
ReDoc:       http://localhost:8000/redoc

Lifespan strategy
-----------------
Heavy objects (BGE embedding model, Qdrant store, cross-encoder reranker, and
the full LegalRAGAgent) are constructed **once per worker** inside the lifespan
context manager and stored on ``app.state``.  Endpoint handlers read from
``request.app.state`` instead of rebuilding the stack per-request, eliminating
the 500–2000 ms cold-start caused by loading the BGE model from disk.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.metrics import get_metrics_response
from src.api.middleware import RequestIDMiddleware
from src.api.middleware_it import ItalianLocalisationMiddleware
from src.config.settings import get_settings
from src.italia.connectors.webhook_router import webhook_router as italia_webhook_router
from src.lexreview.api.router import router as lexreview_router
from src.utils.logger import get_logger

settings = get_settings()

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan handler — builds shared objects once per worker.

    Startup
    ~~~~~~~
    * Constructs one ``LegalRAGAgent`` via :func:`_build_agent` and stores it
      as ``app.state.agent``.  This amortises the BGE model load (500–2000 ms)
      across all requests served by this worker.
    * Also stores ``app.state.embedder`` and ``app.state.reranker`` so the
      ``/lexreview/search`` path can reuse them without rebuilding.

    Shutdown
    ~~~~~~~~
    Logs a clean shutdown message.
    """
    from src.embedding.bge_embedder import BGEEmbedder
    from src.lexreview.agent.llm_client import LLMClient
    from src.lexreview.api.router import _build_agent
    from src.retrieval.reranker import CrossEncoderReranker
    from src.retrieval.sparse_retriever import SparseRetriever
    from src.vectorstore.qdrant_store import QdrantStore

    log.info("RAGForge Enterprise API starting up — building shared agent")

    llm = LLMClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    embedder = BGEEmbedder(model_name=settings.embedding_model)
    reranker = CrossEncoderReranker()

    # _build_agent constructs the full retrieval stack.  When Qdrant is not
    # reachable (e.g. local dev without Docker) we log a warning and continue —
    # the lifespan must not crash, or /health will never be reachable.
    try:
        app.state.agent = _build_agent(
            settings=settings,
            llm=llm,
            collection_name=settings.qdrant_collection_name,
            sparse=sparse,
        )
        log.info("Shared LegalRAGAgent ready")
    except Exception as exc:
        log.warning(
            "Agent initialisation skipped — Qdrant unreachable at startup",
            extra={"error": str(exc)},
        )
        app.state.agent = None  # endpoints will return 503 on demand

    app.state.embedder = embedder
    app.state.reranker = reranker

    # ── Pre-populate the shared SparseRetriever from Qdrant ──────────────────
    # Long-term: migrate to Qdrant native sparse vectors (BM42) to eliminate
    # this RAM index and the startup scroll.
    try:
        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
        )
        fetched_chunks = store.get_all_chunks(settings.qdrant_collection_name)
        sparse = SparseRetriever(chunks=fetched_chunks if fetched_chunks else [])
        if fetched_chunks:
            log.info(
                "SparseRetriever BM25 index warmed up from Qdrant",
                extra={"corpus_size": sparse.corpus_size},
            )
        else:
            log.info("SparseRetriever started with empty corpus (no data in Qdrant yet)")
    except Exception as exc:
        log.warning(
            "SparseRetriever warm-up skipped — Qdrant unreachable at startup",
            extra={"error": str(exc)},
        )
        sparse = SparseRetriever(chunks=[])

    app.state.sparse = sparse
    # ── Build shared agent with pre-warmed sparse retriever ──────────────────
    try:
        app.state.agent = _build_agent(
            settings=settings,
            llm=llm,
            collection_name=settings.qdrant_collection_name,
            sparse=sparse,
        )
        log.info("Shared LegalRAGAgent ready")
    except Exception as exc:
        log.warning(
            "Agent initialisation skipped — Qdrant unreachable at startup",
            extra={"error": str(exc)},
        )
        app.state.agent = None

    log.info("RAGForge Enterprise API startup complete — entering request loop")
    yield
    log.info("RAGForge shutdown")




app = FastAPI(
    title="RAGForge Enterprise API",
    description=(
        "Enterprise-grade RAG document ingestion, retrieval, and legal analysis system. "
        "Includes the LexReview sub-system for legal document Q&A, NER extraction, "
        "and evaluation."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
# NOTE: FastAPI/Starlette applies middleware in *reverse* registration order
# (last registered runs outermost).  Registration order here:
#
#   1. RequestIDMiddleware              (outermost — runs first on every request)
#   2. ItalianLocalisationMiddleware    (second — translates errors, injects disclaimer)
#   3. CORSMiddleware                   (innermost)
#
# This ensures the request ID is set before localisation middleware runs, and
# X-Legal-Disclaimer is injected on every response including CORS pre-flights.

app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    ItalianLocalisationMiddleware,
    disclaimer_text=settings.legal_disclaimer_it,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(lexreview_router)
app.include_router(italia_webhook_router, prefix="/italia")



# ── System endpoints ─────────────────────────────────────────────────────────


@app.get("/health", tags=["system"], summary="Health check")
def health_check() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok", "service": "ragforge-enterprise"}


@app.get(
    "/metrics",
    tags=["system"],
    summary="Prometheus metrics",
    response_description="Prometheus text exposition format",
    include_in_schema=True,
)
def metrics_endpoint() -> object:
    """Expose Prometheus metrics for scraping.

    This endpoint is intentionally **unauthenticated** — the same policy as
    ``/health`` — so that a Prometheus server can reach it without credentials.
    If the metrics endpoint should be private, add network-level access
    controls (e.g. an ingress allow-list) rather than application-level auth.
    """
    return get_metrics_response()
