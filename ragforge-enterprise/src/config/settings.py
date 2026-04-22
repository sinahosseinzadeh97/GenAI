"""Application-wide settings for RAGForge Enterprise.

All tuneable parameters live here and are loaded from environment variables
(or a ``.env`` file). No magic strings are scattered across modules.

Usage::

    from src.config.settings import get_settings

    cfg = get_settings()
    print(cfg.chunk_size)           # 512
    print(cfg.embedding_model)      # "BAAI/bge-small-en-v1.5"
    print(cfg.qdrant_host)          # "localhost"
    print(cfg.qdrant_collection_name) # "ragforge_docs"
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object for RAGForge Enterprise.

    All values can be overridden via environment variables or a ``.env``
    file located in the project root. Variable names are **case-insensitive**.

    Attributes:
        chunk_size:              Target token count per chunk for fixed-size and
                                 recursive chunkers.
        chunk_overlap:           Number of overlapping tokens between consecutive
                                 chunks (must be < chunk_size).
        similarity_threshold:    Cosine-similarity threshold below which a
                                 sentence boundary is declared by the semantic
                                 chunker. Range: [0.0, 1.0].
        embedding_model:         HuggingFace model identifier for BGE embedder.
        embedding_provider:      Which embedder to use: ``"bge"`` or ``"openai"``.
        embedding_batch_size:    Mini-batch size for embedding calls.
        embedding_normalize:     L2-normalise embedding outputs when ``True``.
        openai_api_base:         Base URL for OpenAI-compatible endpoint.
        openai_api_key:          API key for OpenAI-compatible endpoint.
        openai_embedding_model:  Model name sent in OpenAI embedding requests.
        qdrant_host:             Qdrant server hostname.
        qdrant_port:             Qdrant REST API port (default 6333).
        qdrant_collection_name:  Default collection name.
        qdrant_use_grpc:         Use gRPC transport when ``True``.
        hnsw_m:                  HNSW graph parameter (edges per node).
        hnsw_ef_construct:       HNSW build-time quality knob.
        log_level:               Minimum log level forwarded to the JSON logger.
        data_dir:                Directory scanned in batch-load mode.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, gt=0, description="Target token count per chunk.")
    chunk_overlap: int = Field(
        default=64, ge=0, description="Token overlap between consecutive chunks."
    )
    similarity_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Cosine-similarity cut-off for semantic chunker.",
    )

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="sentence-transformers model identifier for BGE embedder.",
    )
    embedding_provider: str = Field(
        default="bge",
        description="Embedding provider: 'bge' or 'openai'.",
    )
    embedding_batch_size: int = Field(
        default=32,
        gt=0,
        description="Number of texts per embedding mini-batch.",
    )
    embedding_normalize: bool = Field(
        default=True,
        description="If True, output embeddings are L2-normalised to unit length.",
    )

    # ── OpenAI-compatible API (optional fallback) ─────────────────────────────
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for an OpenAI-compatible embeddings endpoint.",
    )
    openai_api_key: str = Field(
        default="",
        description="API key / bearer token for the OpenAI-compatible endpoint.",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model name sent in the OpenAI embeddings request body.",
    )

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field(
        default="localhost",
        description="Hostname of the Qdrant server.",
    )
    qdrant_port: int = Field(
        default=6333,
        gt=0,
        description="REST API port of the Qdrant server.",
    )
    qdrant_collection_name: str = Field(
        default="ragforge_docs",
        description="Default Qdrant collection name.",
    )
    qdrant_use_grpc: bool = Field(
        default=False,
        description="If True, use the gRPC transport (port 6334) for Qdrant.",
    )

    # ── HNSW Index ────────────────────────────────────────────────────────────
    hnsw_m: int = Field(
        default=16,
        gt=0,
        description="HNSW 'm' parameter – number of bi-directional edges per node.",
    )
    hnsw_ef_construct: int = Field(
        default=100,
        gt=0,
        description="HNSW 'ef_construct' – size of the dynamic candidate list during build.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Python logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )
    log_retention_days: int = Field(
        default=90,
        gt=0,
        description=(
            "Number of days to retain application log files before automatic rotation. "
            "Implements GDPR Art. 5(1)(e) — storage limitation. "
            "Applied by the rotating file handler configured in src/utils/logger.py."
        ),
    )

    # ── GDPR — Data Processing Agreement ────────────────────────────────────
    gdpr_controller: str = Field(
        default="",
        description="Data controller name for GDPR Art. 30 records of processing activities.",
    )
    gdpr_dpo_email: str = Field(
        default="",
        description="DPO contact email for GDPR Art. 37-39 Data Protection Officer designation.",
    )

    # ── LexReview — LLM ───────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="openai",
        description="LLM provider: 'openai' (default) or 'ollama'.",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model name sent in LLM completion requests.",
    )
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the OpenAI-compatible LLM endpoint.",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM completions (0.0 = deterministic).",
    )
    llm_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum number of tokens in an LLM completion.",
    )

    # ── LexReview — NLP ───────────────────────────────────────────────────────
    spacy_model: str = Field(
        default="en_core_web_sm",
        description="spaCy pipeline name loaded by LegalNER (e.g. 'en_core_web_sm').",
    )
    judge_llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model used by FaithfulnessJudge for answer grounding.",
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Field(
        default=Path("data/sample_docs"),
        description="Directory scanned in batch-load mode.",
    )

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Sentence-transformers cross-encoder model used for reranking.",
    )
    reranker_enabled: bool = Field(
        default=True,
        description="Enable cross-encoder reranking in the retrieval pipeline.",
    )
    reranker_min_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum cross-encoder score threshold; results below this are discarded.",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description=(
            "Whitelist of allowed CORS origins. "
            "Override via CORS_ALLOWED_ORIGINS env var (comma-separated)."
        ),
    )

    # ── Auth ─────────────────────────────────────────────────────────────────
    api_key: str = Field(
        default="",
        description="Secret key required in X-API-Key header for all protected endpoints. Set via API_KEY env var.",
    )

    # ── Italia Connector Layer ────────────────────────────────────────────────
    italia_rate_limit_rps: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Default max HTTP requests per second for Italian legal source connectors. "
            "Individual connectors may apply a lower limit (e.g. 0.5 rps for scrapers)."
        ),
    )
    italia_request_timeout: int = Field(
        default=45,
        gt=0,
        description="HTTP request timeout in seconds for Italian legal source connectors.",
    )
    italia_collection_name: str = Field(
        default="ragforge_italia",
        description=(
            "Qdrant collection name for the Italian legal knowledge base. "
            "Kept separate from the main 'ragforge_docs' collection so that "
            "a multilingual embedder (BAAI/bge-m3) can be used without "
            "impacting the existing English-centric pipeline."
        ),
    )
    italia_data_dir: str = Field(
        default="data/italia",
        description="Local directory used to cache downloaded Italian legal documents.",
    )
    italia_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description=(
            "HuggingFace embedding model for Italian legal text. "
            "Defaults to BAAI/bge-m3 (multilingual, state-of-the-art for Italian). "
            "Override with 'paraphrase-multilingual-mpnet-base-v2' for a lighter model."
        ),
    )
    italia_dejure_api_key: str = Field(
        default="",
        description=(
            "API key for DeJure (Giuffrè) full-access mode. "
            "Obtain at: https://www.giuffre.it/riviste-e-banche-dati/dejure. "
            "When empty, DeJureConnector uses the public massime fallback."
        ),
    )

    # ── Phase 6 — Italian Localisation ──────────────────────────────────────
    default_language: str = Field(
        default="it",
        description=(
            "Default Accept-Language for the Italian deployment. "
            "When 'it', error messages are returned in Italian even if the "
            "client does not send an Accept-Language header."
        ),
    )
    legal_disclaimer_it: str = Field(
        default=(
            "Le risposte fornite hanno carattere informativo "
            "e non costituiscono parere legale."
        ),
        description=(
            "Italian legal disclaimer text injected as the X-Legal-Disclaimer "
            "response header on all endpoints. Overridable without code changes."
        ),
    )

    # ── Phase 6 — FileNet / Documentum ──────────────────────────────────────
    filenet_base_url: str = Field(
        default="",
        description="IBM FileNet / OpenText Documentum CE REST gateway base URL.",
    )
    filenet_repository: str = Field(
        default="FPOS",
        description="FileNet repository name / ID.",
    )
    filenet_username: str = Field(
        default="",
        description="Service-account username for FileNet / Documentum authentication.",
    )
    filenet_password: str = Field(
        default="",
        description="Service-account password for FileNet / Documentum authentication.",
    )
    filenet_folder_path: str = Field(
        default="/",
        description="CMIS folder path to poll in pull mode.",
    )
    filenet_webhook_secret: str = Field(
        default="",
        description=(
            "Shared HMAC-SHA256 secret for verifying FileNet push-event webhooks. "
            "Also used via FILENET_WEBHOOK_SECRET env var by the webhook endpoint."
        ),
    )

    # ── Phase 6 — LexisNexis Italia ─────────────────────────────────────────
    lexisnexis_base_url: str = Field(
        default="https://api.lexisnexis.it/v2",
        description="LexisNexis Italia REST API base URL.",
    )
    lexisnexis_client_id: str = Field(
        default="",
        description="OAuth 2.0 client ID for LexisNexis Italia API.",
    )
    lexisnexis_client_secret: str = Field(
        default="",
        description="OAuth 2.0 client secret for LexisNexis Italia API.",
    )
    lexisnexis_webhook_secret: str = Field(
        default="",
        description=(
            "Shared HMAC-SHA256 secret for verifying LexisNexis push notifications. "
            "Also used via LEXISNEXIS_WEBHOOK_SECRET env var."
        ),
    )

    # ── Phase 6 — Notartel ─────────────────────────────────────────────────
    notartel_base_url: str = Field(
        default="https://api.notartel.it/v3",
        description="Notartel REST gateway base URL.",
    )
    notartel_token: str = Field(
        default="",
        description=(
            "Bearer token for Notartel IdP authentication. "
            "Tokens expire every 8 hours; set via NOTARTEL_TOKEN env var."
        ),
    )
    notartel_stub: bool = Field(
        default=False,
        description=(
            "When True, NotartelConnector returns fixture data without HTTP calls. "
            "Override via NOTARTEL_STUB=true env var."
        ),
    )

    # ── Phase 6 — SIECIC / SICID (Ministry of Justice) ─────────────────────
    siecic_base_url: str = Field(
        default="https://pst.giustizia.it/api/v1",
        description="Portale dei Servizi Telematici REST API base URL.",
    )
    siecic_api_key: str = Field(
        default="",
        description=(
            "DGSIA-issued API key for SIECIC (civil) / SICID (criminal) access. "
            "Requires formal agreement with the Ministry of Justice (DGSIA)."
        ),
    )
    siecic_stub: bool = Field(
        default=False,
        description=(
            "When True, SiecicSicidConnector returns fixture data without HTTP calls. "
            "Override via SIECIC_STUB=true env var."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, v: int, info: object) -> int:
        """Ensure overlap is strictly less than chunk_size.

        Args:
            v:    The candidate overlap value.
            info: Pydantic validation info carrying sibling field values.

        Returns:
            The validated overlap value.

        Raises:
            ValueError: When *v* >= ``chunk_size``.
        """
        # info.data is populated only when chunk_size has already been validated.
        data = getattr(info, "data", {})
        chunk_size: int = data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be strictly less than chunk_size ({chunk_size})."
            )
        return v

    @field_validator("log_level")
    @classmethod
    def normalise_log_level(cls, v: str) -> str:
        """Upper-case the log level and validate against known Python levels.

        Args:
            v: Raw level string from env / .env file.

        Returns:
            Upper-cased level string.

        Raises:
            ValueError: When *v* is not a recognised Python log level.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}, got: {v!r}.")
        return upper

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        """Validate that the provider is one of the supported values.

        Args:
            v: Raw provider string.

        Returns:
            Lower-cased provider string.

        Raises:
            ValueError: When *v* is not a recognised provider.
        """
        supported = {"bge", "openai"}
        lower = v.lower()
        if lower not in supported:
            raise ValueError(
                f"embedding_provider must be one of {supported}, got: {v!r}."
            )
        return lower


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton :class:`Settings` instance (cached after first call).

    The cache ensures that ``.env`` is parsed only once per process, which
    keeps module-level imports cheap.

    Returns:
        A fully validated :class:`Settings` object.

    Example:
        >>> cfg = get_settings()
        >>> cfg.chunk_size
        512
    """
    return Settings()
