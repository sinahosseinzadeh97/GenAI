#!/usr/bin/env python3
"""CLI entrypoint for the RAGForge Phase 2 indexing pipeline.

Loads PDFs from a directory (Phase 1), cleans and chunks them, embeds all
chunks, then upserts them into a Qdrant collection.  Prints a formatted
:class:`~src.indexing.pipeline.IndexingReport` on completion.

Usage::

    python scripts/index_documents.py \\
        --input data/sample_docs/ \\
        --collection ragforge_docs \\
        --strategy recursive \\
        --embedder bge

Options
-------
--input       Directory containing PDF files to index.
--collection  Qdrant collection name (overrides QDRANT_COLLECTION_NAME env var).
--strategy    Chunking strategy: ``fixed``, ``recursive`` (default), or ``semantic``.
--embedder    Embedding provider: ``bge`` (default) or ``openai``.
--batch-size  Embedding mini-batch size (default: 32).
--host        Qdrant host (default: localhost).
--port        Qdrant REST port (default: 6333).
--no-normalize Skip L2 normalisation of embeddings.
--dry-run     Run through ingestion/chunking/embedding but skip upsert.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── Make sure the project root is on sys.path ─────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config.settings import get_settings
from src.embedding.bge_embedder import BGEEmbedder
from src.embedding.openai_embedder import OpenAIEmbedder
from src.indexing.pipeline import IndexingPipeline
from src.ingestion.chunker import (
    Chunk,
    FixedSizeChunker,
    RecursiveChunker,
    SemanticChunker,
)
from src.ingestion.cleaner import DocumentCleaner
from src.ingestion.loader import DocumentLoader
from src.utils.logger import get_logger
from src.vectorstore.qdrant_store import QdrantStore

log = get_logger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="index_documents",
        description="RAGForge Enterprise – PDF ingestion and indexing pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/sample_docs"),
        metavar="DIR",
        help="Directory containing PDF files to index.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name.  Falls back to QDRANT_COLLECTION_NAME env var.",
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "recursive", "semantic"],
        default="recursive",
        help="Chunking strategy.",
    )
    parser.add_argument(
        "--embedder",
        choices=["bge", "openai"],
        default="bge",
        help="Embedding provider.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Embedding mini-batch size.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Qdrant server hostname.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Qdrant REST API port.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalisation of embeddings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run ingestion, cleaning, chunking, and embedding but skip the Qdrant upsert.",
    )
    return parser


def load_and_chunk(input_dir: Path, strategy: str) -> list[Chunk]:
    """Load PDFs from *input_dir*, clean them, and chunk with *strategy*.

    Args:
        input_dir: Directory to scan for PDF files.
        strategy:  One of ``"fixed"``, ``"recursive"``, or ``"semantic"``.

    Returns:
        Flat list of :class:`~src.ingestion.chunker.Chunk` objects.
    """
    cfg = get_settings()
    loader = DocumentLoader()
    cleaner = DocumentCleaner()

    pdf_files = sorted(input_dir.glob("**/*.pdf"))
    if not pdf_files:
        log.warning("No PDF files found", extra={"directory": str(input_dir)})
        print(f"[WARN] No PDF files found in: {input_dir}")
        return []

    print(f"\n📂  Found {len(pdf_files)} PDF file(s) in {input_dir}")

    all_docs = []
    for pdf_path in pdf_files:
        print(f"  ↳ Loading: {pdf_path.name}")
        try:
            docs = loader.load_file(pdf_path)
            cleaned = cleaner.clean_batch(docs)
            all_docs.extend(cleaned)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] Failed to load {pdf_path.name}: {exc}")

    print(f"\n📄  Loaded {len(all_docs)} document page(s)")

    chunker_map = {
        "fixed": FixedSizeChunker(cfg),
        "recursive": RecursiveChunker(cfg),
        "semantic": SemanticChunker(cfg),
    }
    chunker = chunker_map[strategy]
    print(f"✂️   Chunking with strategy: {strategy!r}")

    chunks = chunker.chunk_documents(all_docs)
    print(f"🧩  Produced {len(chunks)} chunk(s)\n")
    return chunks


def main() -> None:
    """Run the full indexing pipeline from the command line."""
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = get_settings()

    # ── Resolve configuration ─────────────────────────────────────────────────
    collection_name: str = args.collection or cfg.qdrant_collection_name
    qdrant_host: str = args.host or cfg.qdrant_host
    qdrant_port: int = args.port or cfg.qdrant_port
    batch_size: int = args.batch_size or cfg.embedding_batch_size
    normalize: bool = not args.no_normalize

    print("=" * 60)
    print("  RAGForge Enterprise – Indexing Pipeline")
    print("=" * 60)
    print(f"  Input directory : {args.input}")
    print(f"  Collection      : {collection_name}")
    print(f"  Strategy        : {args.strategy}")
    print(f"  Embedder        : {args.embedder}")
    print(f"  Qdrant          : {qdrant_host}:{qdrant_port}")
    print(f"  Dry run         : {args.dry_run}")
    print("=" * 60)

    # ── Step 1–3: Load, clean, chunk ──────────────────────────────────────────
    t0 = time.perf_counter()
    chunks = load_and_chunk(args.input, args.strategy)
    if not chunks:
        sys.exit(0)

    # ── Step 4: Select embedder ───────────────────────────────────────────────
    print(f"🔢  Initialising embedder: {args.embedder!r}")
    if args.embedder == "bge":
        embedder = BGEEmbedder(
            model_name=cfg.embedding_model,
            normalize=normalize,
            batch_size=batch_size,
        )
    else:
        embedder = OpenAIEmbedder(
            normalize=normalize,
            batch_size=batch_size,
        )

    # ── Dry run: embed only ───────────────────────────────────────────────────
    if args.dry_run:
        print("🔬  Dry-run mode: embedding only (no Qdrant upsert)")
        t_embed = time.perf_counter()
        texts = [c.content for c in chunks]
        vectors = embedder.embed_batch(texts, batch_size=batch_size)
        elapsed = time.perf_counter() - t_embed
        throughput = len(chunks) / elapsed if elapsed > 0 else float("inf")
        print(f"\n✅  Dry-run complete")
        print(f"   Embedded {len(vectors)} chunks in {elapsed:.3f}s ({throughput:.1f} chunks/s)")
        sys.exit(0)

    # ── Step 5: Connect to Qdrant ─────────────────────────────────────────────
    print(f"\n🔌  Connecting to Qdrant @ {qdrant_host}:{qdrant_port} …")
    vector_store = QdrantStore(
        host=qdrant_host,
        port=qdrant_port,
        collection_name=collection_name,
    )
    if not vector_store.health_check():
        print(
            f"[ERROR] Cannot reach Qdrant at {qdrant_host}:{qdrant_port}.\n"
            "        Run: docker compose up -d"
        )
        sys.exit(1)
    print("    ✓ Qdrant is healthy")

    # ── Step 6: Run full pipeline ─────────────────────────────────────────────
    print("\n⚙️   Running indexing pipeline …")
    pipeline = IndexingPipeline(
        embedder=embedder,
        vector_store=vector_store,
        collection_name=collection_name,
        embedding_batch_size=batch_size,
        auto_create_collection=True,
    )

    report = pipeline.run(chunks)
    total_elapsed = time.perf_counter() - t0

    # ── Print report ──────────────────────────────────────────────────────────
    print(report)
    print(f"⏱   Total wall-clock time: {total_elapsed:.2f}s")

    # ── Collection stats ──────────────────────────────────────────────────────
    try:
        info = vector_store.collection_info(collection_name)
        print("\n📊  Qdrant Collection Stats")
        print(f"    vectors_count : {info.get('vectors_count', 'N/A')}")
        print(f"    status        : {info.get('status', 'N/A')}")
        print(f"    vector_size   : {info.get('vector_size', 'N/A')}")
        print(f"    distance      : {info.get('distance', 'N/A')}")
    except Exception:  # noqa: BLE001
        pass

    if report.total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
