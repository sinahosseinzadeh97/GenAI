"""CLI ingestion script for RAGForge Italia.

Fetches documents from one or all Italian legal sources and indexes them
into a dedicated Qdrant collection using the existing IndexingPipeline.

Usage::

    # Single source
    python scripts/ingest_italia.py --source normattiva --codice codice_civile

    # All critical sources with a date window
    python scripts/ingest_italia.py --source all --from-date 2024-01-01 --limit 100

    # Dry run (fetch only, no indexing)
    python scripts/ingest_italia.py --source cassazione --dry-run --limit 5

Environment variables used:
    ITALIA_COLLECTION_NAME  (default: ragforge_italia)
    EMBEDDING_MODEL         (default: BAAI/bge-m3  — multilingual)
    QDRANT_HOST / QDRANT_PORT
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

# Ensure src is on the path when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.embedding.bge_embedder import BGEEmbedder
from src.indexing.pipeline import IndexingPipeline
from src.ingestion.chunker import FixedSizeChunker
from src.ingestion.cleaner import DocumentCleaner
from src.italia.connectors import CONNECTORS
from src.vectorstore.qdrant_store import QdrantStore
from src.utils.logger import get_logger

log = get_logger(__name__)

_PRIORITY_ORDER = [
    "normattiva",
    "gazzetta",
    "cassazione",
    "eurlex",
    "tar",
    "corte_costituzionale",
    "agcm",
    "bancaditalia",
    "dejure",
]


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        print(f"[ERROR] Invalid date '{s}'. Use YYYY-MM-DD format.", file=sys.stderr)
        sys.exit(1)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RAGForge Italia — Italian legal knowledge ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources:
  normattiva        Italian legislation via NIR/URN API (normattiva.it)
  gazzetta          Gazzetta Ufficiale legislative acts
  cassazione        Corte di Cassazione sentenze (ItalGiurè)
  eurlex            EU Directives and Regulations in Italian
  tar               TAR and Consiglio di Stato sentenze
  corte_costituzionale  Constitutional Court decisions
  agcm              AGCM antitrust provvedimenti
  bancaditalia      Banca d'Italia and IVASS regulations
  dejure            DeJure massime (public fallback; API key for full access)
  all               All sources in priority order

Examples:
  python scripts/ingest_italia.py --source normattiva --codice codice_civile
  python scripts/ingest_italia.py --source cassazione --from-date 2023-01-01 --limit 50
  python scripts/ingest_italia.py --source all --limit 200 --dry-run
        """,
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=list(CONNECTORS.keys()) + ["all"],
        help="Source to ingest from (or 'all' for every source in priority order).",
    )
    parser.add_argument(
        "--from-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date filter (for sources that support date ranges).",
    )
    parser.add_argument(
        "--to-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum documents to fetch per source (default: 50).",
    )
    parser.add_argument(
        "--codice",
        default=None,
        help="Normattiva codice shortcut (e.g. codice_civile, dlgs_231).",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Full-text search query for search-based sources.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (default: ITALIA_COLLECTION_NAME env var).",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-m3",
        help="HuggingFace embedding model (default: BAAI/bge-m3 — multilingual).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse documents but skip indexing into Qdrant.",
    )
    parser.add_argument(
        "--dejure-api-key",
        default=None,
        help="DeJure API key (overrides DEJURE_API_KEY env var).",
    )
    return parser


def _run_source(
    source_name: str,
    args: argparse.Namespace,
    pipeline: IndexingPipeline | None,
    cleaner: DocumentCleaner,
    chunker: FixedSizeChunker,
) -> dict[str, int]:
    """Fetch + clean + chunk + index for a single source. Returns summary dict."""
    print(f"\n{'─' * 60}")
    print(f"  Source: {source_name.upper()}")
    print(f"{'─' * 60}")

    connector_cls = CONNECTORS[source_name]

    # Inject DeJure API key if provided.
    kwargs: dict = {}
    if source_name == "dejure":
        import os  # noqa: PLC0415
        api_key = args.dejure_api_key or os.getenv("DEJURE_API_KEY")
        kwargs = {"api_key": api_key, "fallback_public": True}

    connector = connector_cls(**kwargs)

    fetch_kwargs: dict = {
        "query": args.query,
        "limit": args.limit,
        "from_date": _parse_date(args.from_date),
        "to_date": _parse_date(args.to_date),
    }
    if source_name == "normattiva" and args.codice:
        fetch_kwargs["codice"] = args.codice
    if source_name == "normattiva" and not args.codice and not args.query:
        # Default: fetch the Codice Civile for demo purposes.
        fetch_kwargs["codice"] = "codice_civile"

    try:
        documents = connector.fetch(**fetch_kwargs)
        connector.close()
    except Exception as exc:  # noqa: BLE001
        print(f"  [FETCH ERROR] {exc}")
        return {"source": source_name, "documents": 0, "chunks": 0, "indexed": 0}

    print(f"  Fetched:  {len(documents)} documents")
    if not documents:
        return {"source": source_name, "documents": 0, "chunks": 0, "indexed": 0}

    cleaned = cleaner.clean_batch(documents)
    chunks = chunker.chunk_documents(cleaned)
    print(f"  Chunks:   {len(chunks)}")

    if args.dry_run or pipeline is None:
        print("  Indexing: SKIPPED (dry-run)")
        return {"source": source_name, "documents": len(documents), "chunks": len(chunks), "indexed": 0}

    report = pipeline.run(chunks)
    print(f"  Indexed:  {report.total_indexed}/{report.total_chunks_processed} chunks")
    if report.total_failed > 0:
        print(f"  Failed:   {report.total_failed} chunks")

    return {
        "source": source_name,
        "documents": len(documents),
        "chunks": len(chunks),
        "indexed": report.total_indexed,
    }


def main() -> None:
    """Entry point for the Italia ingestion CLI."""
    parser = _build_argparser()
    args = parser.parse_args()

    settings = get_settings()
    collection = args.collection or getattr(settings, "italia_collection_name", "ragforge_italia")

    print(f"\n{'═' * 60}")
    print(f"  RAGForge Italia — Legal Knowledge Ingestion")
    print(f"{'═' * 60}")
    print(f"  Collection:  {collection}")
    print(f"  Dry-run:     {args.dry_run}")
    print(f"  Limit/source:{args.limit}")

    # Build shared components.
    cleaner = DocumentCleaner()
    chunker = FixedSizeChunker(settings)

    pipeline: IndexingPipeline | None = None
    if not args.dry_run:
        embedder = BGEEmbedder(model_name=args.embedding_model)
        store = QdrantStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=collection,
        )
        pipeline = IndexingPipeline(
            embedder=embedder,
            vector_store=store,
            collection_name=collection,
            auto_create_collection=True,
        )

    sources = _PRIORITY_ORDER if args.source == "all" else [args.source]
    summaries = []
    for source_name in sources:
        summary = _run_source(source_name, args, pipeline, cleaner, chunker)
        summaries.append(summary)

    # ── Final report ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"  INGESTION COMPLETE")
    print(f"{'═' * 60}")
    total_docs = sum(s["documents"] for s in summaries)
    total_chunks = sum(s["chunks"] for s in summaries)
    total_indexed = sum(s["indexed"] for s in summaries)
    print(f"  Total documents fetched: {total_docs:>8,}")
    print(f"  Total chunks produced:   {total_chunks:>8,}")
    print(f"  Total chunks indexed:    {total_indexed:>8,}")
    print(f"  Collection:              {collection}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
