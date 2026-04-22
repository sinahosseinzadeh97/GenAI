"""FastAPI request/response schemas for LexReview endpoints.

Schemas
-------
QueryRequest / QueryResponse
    POST /lexreview/query — agent-powered legal Q&A.
ExtractRequest / ExtractResponse
    POST /lexreview/extract — spaCy NER + clause detection.
IndexRequest / IndexResponse
    POST /lexreview/index — ingest texts into Qdrant.
LexSearchRequest / LexSearchResponse
    POST /lexreview/search — hybrid retrieval without agent.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from src.lexreview.agent.models import AgentResponse, Citation  # noqa: F401 (re-exported)
from src.lexreview.extraction.models import Clause, LegalEntities  # noqa: F401

# ── Query (agent) ─────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    """Request body for POST /lexreview/query.

    Attributes:
        query:           Legal question (5–1000 chars).
        collection_name: Qdrant collection to query.
        top_k:           Number of candidates to retrieve (default 10).
        rerank:          Whether to apply cross-encoder reranking (default True).
        filters:         Optional Qdrant metadata filter map.
    """

    query: str = Field(..., min_length=5, max_length=1000, description="Legal query.")
    collection_name: str = Field(..., description="Qdrant collection name.")
    top_k: int = Field(default=10, ge=1, le=50)
    rerank: bool = Field(default=True)
    filters: dict[str, Any] | None = Field(default=None)

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the payment terms?",
                "collection_name": "contracts",
                "top_k": 10,
                "rerank": True,
            }
        }
    }


class QueryResponse(BaseModel):
    """Response envelope for POST /lexreview/query."""

    query: str
    collection_name: str
    answer: str
    citations: list[Citation]
    confidence: float
    reasoning_steps: list[str]
    latency_ms: float


# ── Extract (NLP) ─────────────────────────────────────────────────────────────


class ExtractRequest(BaseModel):
    """Request body for POST /lexreview/extract.

    Attributes:
        text:             Raw document or clause text to analyse.
        extract_entities: Run LegalNER + RegexExtractor (default True).
        detect_clauses:   Run ClauseDetector (default True).
        min_confidence:   Minimum clause detection confidence (default 0.3).
    """

    text: str = Field(..., min_length=10, max_length=100_000, description="Document text.")
    extract_entities: bool = Field(default=True)
    detect_clauses: bool = Field(default=True)
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "This Agreement is governed by the laws of the State of Delaware.",
                "extract_entities": True,
                "detect_clauses": True,
            }
        }
    }


class ExtractResponse(BaseModel):
    """Response envelope for POST /lexreview/extract."""

    entities: LegalEntities
    clauses: list[Clause]
    char_count: int


# ── Index ─────────────────────────────────────────────────────────────────────


class IndexRequest(BaseModel):
    """Request body for POST /lexreview/index.

    Attributes:
        texts:           Plain-text strings to embed and index.
        metadatas:       Parallel list of metadata dicts (one per text).
        collection_name: Target Qdrant collection (auto-created if absent).
        source_path:     Canonical document identifier used for deduplication.
                         When supplied, the endpoint checks whether any chunk
                         with this ``source_path`` payload value already exists
                         in the collection and returns early (HTTP 200,
                         ``already_indexed=True``) unless ``force_reindex`` is
                         set.  If omitted the deduplication check is skipped.
        force_reindex:   When ``True``, bypass the deduplication guard and
                         re-embed/re-upsert even if the source is already
                         indexed.  Existing chunks are overwritten in place
                         because chunk IDs are deterministic (Issue C).
    """

    texts: list[str] = Field(..., min_length=1, description="Texts to embed and index.")
    metadatas: list[dict[str, Any]] = Field(default_factory=list)
    collection_name: str = Field(..., description="Target Qdrant collection.")
    source_path: str | None = Field(
        default=None,
        description=(
            "Canonical document identifier used for deduplication. "
            "Omit to skip the deduplication check."
        ),
    )
    force_reindex: bool = Field(
        default=False,
        description=(
            "Re-embed and re-upsert even if the source is already indexed. "
            "Existing chunks are overwritten (idempotent via deterministic IDs)."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": ["Party A agrees to pay Party B..."],
                "metadatas": [{"source_path": "nda_2024.pdf"}],
                "collection_name": "contracts",
                "source_path": "nda_2024.pdf",
                "force_reindex": False,
            }
        }
    }


class IndexResponse(BaseModel):
    """Response envelope for POST /lexreview/index.

    Attributes:
        indexed_count:   Number of chunks actually embedded and upserted.
                         ``0`` when ``already_indexed`` is ``True``.
        collection_name: Qdrant collection that was targeted.
        latency_ms:      Wall-clock time for the operation in milliseconds.
        already_indexed: ``True`` when the deduplication guard fired and the
                         source was already present in the collection.
        source_path:     Echoes back the ``source_path`` that triggered the
                         deduplication check (``None`` when not supplied).
    """

    indexed_count: int
    collection_name: str
    latency_ms: float
    already_indexed: bool = False
    source_path: str | None = None


# ── Search (retrieval only) ───────────────────────────────────────────────────


class LexSearchRequest(BaseModel):
    """Request body for POST /lexreview/search (retrieval without agent).

    Attributes:
        query:           Search query string.
        collection_name: Qdrant collection.
        top_k:           Number of results (default 10).
        rerank:          Apply cross-encoder reranking (default True).
        filters:         Optional metadata filters.
    """

    query: str = Field(..., min_length=3, max_length=500)
    collection_name: str
    top_k: int = Field(default=10, ge=1, le=50)
    rerank: bool = Field(default=True)
    filters: dict[str, Any] | None = Field(default=None)


class LexSearchResultItem(BaseModel):
    """A single search result item."""

    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class LexSearchResponse(BaseModel):
    """Response envelope for POST /lexreview/search."""

    query: str
    results: list[LexSearchResultItem]
    total_found: int
    latency_ms: float
    reranked: bool


# ── Phase 5 — Italian Market Schemas ─────────────────────────────────────────


# 5.1 Vigenza ─────────────────────────────────────────────────────────────────


class VigenzaRequest(BaseModel):
    """Request body for POST /lexreview/vigenza.

    Attributes:
        norma:             Italian norm citation to check,
                           e.g. ``"Art. 18 L. 300/1970"``.
        data_riferimento:  Reference date for the validity check (ISO 8601).
    """

    norma: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Citazione della norma da verificare, es. 'Art. 18 L. 300/1970'.",
    )
    data_riferimento: str = Field(
        ...,
        description="Data di riferimento ISO 8601 (YYYY-MM-DD).",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "norma": "Art. 18 L. 300/1970",
                "data_riferimento": "2024-03-01",
            }
        }
    }


class VigenzaResponse(BaseModel):
    """Response envelope for POST /lexreview/vigenza."""

    vigente: bool
    data_entrata_vigore: str | None = Field(
        default=None,
        description="Data di entrata in vigore (YYYY-MM-DD) o null.",
    )
    data_abrogazione: str | None = Field(
        default=None,
        description="Data di abrogazione (YYYY-MM-DD) o null se ancora vigente.",
    )
    modificata_da: list[str] = Field(
        default_factory=list,
        description="Elenco degli atti che hanno modificato la norma.",
    )
    testo_vigente: str | None = Field(
        default=None,
        description="Testo vigente alla data di riferimento, o null se non disponibile.",
    )
    fonte: str = Field(
        default="llm_inference",
        description="Fonte della determinazione: 'metadata' o 'llm_inference'.",
    )
    latency_ms: float = Field(default=0.0)


# 5.2 Massimario ──────────────────────────────────────────────────────────────


class MassimaRequest(BaseModel):
    """Request body for POST /lexreview/massima.

    Attributes:
        testo_sentenza: Full or partial text of the Italian court judgment.
    """

    testo_sentenza: str = Field(
        ...,
        min_length=100,
        max_length=100_000,
        description="Testo della sentenza da massimizzare.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "testo_sentenza": (
                    "La Corte di Cassazione, Sezione Lavoro, con sentenza n. 12345/2024 "
                    "ha affermato che il licenziamento per giustificato motivo oggettivo..."
                )
            }
        }
    }


class MassimaResponse(BaseModel):
    """Response envelope for POST /lexreview/massima."""

    massima_ufficiale: str = Field(description="Massima ufficiale (max 150 parole).")
    principio_di_diritto: str = Field(description="Principio di diritto in forma sillogistica.")
    parole_chiave: list[str] = Field(
        default_factory=list,
        description="Parole chiave per l'indicizzazione.",
    )
    classificazione_materia: list[str] = Field(
        default_factory=list,
        description="Classificazione per materia del diritto.",
    )
    latency_ms: float = Field(default=0.0)


# 5.3 Contratto ───────────────────────────────────────────────────────────────


class ClausolaAnalisiSchema(BaseModel):
    """A single analysed contract clause."""

    testo_clausola: str
    tipo: str = Field(description="'vessatoria' | 'nulla_cc' | 'irregolare'")
    riferimento_normativo: str
    motivazione: str
    risk_score: str = Field(description="🔴 | 🟠 | 🟡 | 🟢")
    correzione_suggerita: str


class ContrattoAnalisiResponse(BaseModel):
    """Response envelope for POST /lexreview/contratto/analisi."""

    clausole_vessatorie: list[ClausolaAnalisiSchema] = Field(default_factory=list)
    clausole_nulle: list[ClausolaAnalisiSchema] = Field(default_factory=list)
    risk_score_globale: str = Field(description="🔴 | 🟠 | 🟡 | 🟢")
    sommario: str
    latency_ms: float = Field(default=0.0)


# 5.4 231 Compliance ──────────────────────────────────────────────────────────


class D231RiskRequest(BaseModel):
    """Request body for POST /lexreview/231/risk-assessment.

    Attributes:
        settore:               Industry / sector of the company,
                               e.g. ``"edilizia"``, ``"bancario"``.
        descrizione_attivita:  Free-text description of business activities.
    """

    settore: str = Field(
        ...,
        min_length=2,
        max_length=200,
        description="Settore di attività dell'ente, es. 'edilizia', 'bancario'.",
    )
    descrizione_attivita: str = Field(
        ...,
        min_length=20,
        max_length=5_000,
        description="Descrizione delle attività svolte dall'ente.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "settore": "edilizia",
                "descrizione_attivita": (
                    "La società opera nel settore delle costruzioni, gestisce appalti "
                    "pubblici e privati, ha rapporti con la pubblica amministrazione "
                    "per concessioni edilizie e partecipa a gare d'appalto."
                ),
            }
        }
    }


class D231RiskResponse(BaseModel):
    """Response envelope for POST /lexreview/231/risk-assessment."""

    reati_presupposto: list[str] = Field(
        default_factory=list,
        description="Reati presupposto applicabili ai sensi del D.Lgs. 231/2001.",
    )
    odv_raccomandazioni: list[str] = Field(
        default_factory=list,
        description="Raccomandazioni per l'Organismo di Vigilanza.",
    )
    risk_score: float = Field(
        description="Score di rischio complessivo [0.0 – 1.0].",
        ge=0.0,
        le=1.0,
    )
    riferimenti_normativi: list[str] = Field(
        default_factory=list,
        description="Riferimenti normativi agli articoli del D.Lgs. 231/2001.",
    )
    sintesi: str = Field(default="", description="Sintesi esecutiva della valutazione.")
    latency_ms: float = Field(default=0.0)
