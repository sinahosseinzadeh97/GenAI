"""Qdrant vector store implementation.

Connects to a Qdrant instance (local Docker or cloud) using the official
``qdrant-client`` library and implements the full :class:`BaseVectorStore`
interface, including:

- HNSW-configured collection creation with payload indexing.
- Batched upsert of :class:`~src.vectorstore.schema.IndexedChunk` objects.
- Metadata-filtered similarity search.
- Collection management utilities.
- Health check.

Typical usage::

    from src.vectorstore.qdrant_store import QdrantStore
    from src.vectorstore.schema import IndexedChunk

    store = QdrantStore()
    store.create_collection("ragforge_docs", vector_size=384)
    stats = store.upsert_chunks(indexed_chunks)
    results = store.search(query_vector, top_k=5)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from src.config.settings import get_settings
from src.utils.logger import get_logger, log_exception
from src.vectorstore.base import BaseVectorStore, UpsertStats, VectorStoreError
from src.vectorstore.schema import IndexedChunk, SearchResult

_settings = get_settings()
log = get_logger(__name__, level=_settings.log_level)


class QdrantStore(BaseVectorStore):
    """Qdrant-backed vector store.

    Wraps :mod:`qdrant_client` to provide a clean, typed interface that
    matches the :class:`~src.vectorstore.base.BaseVectorStore` contract.

    Args:
        host:            Qdrant server hostname (default from settings).
        port:            REST API port (default from settings).
        use_grpc:        If ``True``, use the gRPC transport (faster for bulk
                         operations).  Defaults to settings value.
        collection_name: Default collection used by convenience methods.
                         Individual method calls can always override this.
        hnsw_m:          HNSW ``m`` graph parameter (edges per node).
        hnsw_ef_construct: HNSW ``ef_construct`` (build-time quality knob).
        prefer_grpc:     Passed through to ``QdrantClient``.

    Raises:
        VectorStoreError: If the client cannot be instantiated.

    Example::

        store = QdrantStore(host="localhost", port=6333)
        ok = store.health_check()
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        use_grpc: bool | None = None,
        collection_name: str | None = None,
        hnsw_m: int | None = None,
        hnsw_ef_construct: int | None = None,
    ) -> None:
        self._host: str = host or _settings.qdrant_host
        self._port: int = port or _settings.qdrant_port
        self._use_grpc: bool = use_grpc if use_grpc is not None else _settings.qdrant_use_grpc
        self._collection_name: str = collection_name or _settings.qdrant_collection_name
        self._hnsw_m: int = hnsw_m or _settings.hnsw_m
        self._hnsw_ef_construct: int = hnsw_ef_construct or _settings.hnsw_ef_construct

        self._client: Any = self._build_client()

    # ── Client factory ────────────────────────────────────────────────────────

    def _build_client(self) -> Any:
        """Instantiate and return the underlying ``QdrantClient``.

        Returns:
            A connected ``qdrant_client.QdrantClient`` instance.

        Raises:
            VectorStoreError: If ``qdrant-client`` is not installed or the
                connection cannot be established.
        """
        try:
            from qdrant_client import QdrantClient  # type: ignore[import-untyped]

            log.info(
                "Connecting to Qdrant",
                extra={
                    "host": self._host,
                    "port": self._port,
                    "grpc": self._use_grpc,
                },
            )
            client = QdrantClient(
                host=self._host,
                port=self._port,
                prefer_grpc=self._use_grpc,
            )
            return client
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client is not installed. Run: pip install qdrant-client"
            ) from exc
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to connect to Qdrant at {self._host}:{self._port}: {exc}"
            ) from exc

    # ── Collection management ─────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        vector_size: int,
        distance: str = "COSINE",
    ) -> None:
        """Create a Qdrant collection with HNSW indexing and payload indices.

        Idempotent: if the collection already exists with the same vector size,
        this method returns silently.

        Args:
            name:        Collection name.
            vector_size: Embedding dimension.
            distance:    Distance metric – ``"COSINE"`` (default), ``"DOT"``,
                         or ``"EUCLID"``.

        Raises:
            VectorStoreError: On Qdrant API failure.
        """
        try:
            from qdrant_client.http import models as qmodels  # type: ignore[import-untyped]

            # Check if already exists.
            existing = self._client.get_collections().collections
            existing_names = {c.name for c in existing}
            if name in existing_names:
                info = self._client.get_collection(name)
                existing_size = info.config.params.vectors.size
                if existing_size != vector_size:
                    raise VectorStoreError(
                        f"Collection '{name}' exists with size={existing_size} "
                        f"but vector_size={vector_size} requested."
                    )
                log.info(
                    "Collection already exists, skipping creation",
                    extra={"collection": name, "vector_size": vector_size},
                )
                return

            distance_map = {
                "COSINE": qmodels.Distance.COSINE,
                "DOT": qmodels.Distance.DOT,
                "EUCLID": qmodels.Distance.EUCLID,
            }
            qdrant_distance = distance_map.get(distance.upper(), qmodels.Distance.COSINE)

            self._client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qdrant_distance,
                ),
                hnsw_config=qmodels.HnswConfigDiff(
                    m=self._hnsw_m,
                    ef_construct=self._hnsw_ef_construct,
                ),
            )

            # Create payload indices for fast filtered search.
            for field_name in ("source_path", "strategy_used", "page_number"):
                try:
                    self._client.create_payload_index(
                        collection_name=name,
                        field_name=field_name,
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                        if field_name != "page_number"
                        else qmodels.PayloadSchemaType.INTEGER,
                    )
                    log.debug(
                        "Payload index created",
                        extra={"collection": name, "field": field_name},
                    )
                except Exception:
                    # Payload index creation is best-effort.
                    log.warning(
                        "Could not create payload index",
                        extra={"collection": name, "field": field_name},
                    )

            log.info(
                "Collection created",
                extra={
                    "collection": name,
                    "vector_size": vector_size,
                    "distance": distance,
                    "hnsw_m": self._hnsw_m,
                    "hnsw_ef_construct": self._hnsw_ef_construct,
                },
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            log_exception(log, "create_collection failed", exc)
            raise VectorStoreError(f"Failed to create collection '{name}': {exc}") from exc

    def delete_collection(self, name: str) -> None:
        """Delete a Qdrant collection.

        Args:
            name: Collection name to delete.

        Raises:
            VectorStoreError: On Qdrant API failure.
        """
        try:
            self._client.delete_collection(name)
            log.info("Collection deleted", extra={"collection": name})
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete collection '{name}': {exc}") from exc

    def collection_info(self, name: str) -> dict[str, Any]:
        """Return metadata for *name*.

        Args:
            name: Collection name.

        Returns:
            Dictionary with ``vectors_count``, ``status``, ``vector_size``
            keys (plus additional Qdrant metadata).

        Raises:
            VectorStoreError: When the collection does not exist.
        """
        try:
            info = self._client.get_collection(name)
            # qdrant-client ≥ 1.9: vectors_count moved to points_count.
            vectors_count: int = (
                getattr(info, "points_count", None)
                or getattr(info, "vectors_count", None)
                or 0
            )
            indexed_count: int = (
                getattr(info, "indexed_vectors_count", None)
                or 0
            )
            return {
                "name": name,
                "vectors_count": vectors_count,
                "indexed_vectors_count": indexed_count,
                "status": str(info.status),
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
                "hnsw_m": info.config.hnsw_config.m,
                "hnsw_ef_construct": info.config.hnsw_config.ef_construct,
            }
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to get info for collection '{name}': {exc}"
            ) from exc

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_chunks(
        self,
        chunks: list[IndexedChunk],
        batch_size: int = 100,
    ) -> UpsertStats:
        """Upsert :class:`~src.vectorstore.schema.IndexedChunk` objects into Qdrant.

        Uses the collection name set at construction time (``self._collection_name``).

        Args:
            chunks:     List of indexed chunks to upsert.
            batch_size: Number of points per Qdrant upsert request.

        Returns:
            :class:`~src.vectorstore.base.UpsertStats` with insert/update counts
            and wall-clock duration.

        Raises:
            VectorStoreError: On Qdrant API failure.
        """
        if not chunks:
            return UpsertStats(total_inserted=0, total_updated=0, duration_seconds=0.0)

        try:
            from qdrant_client.http import models as qmodels  # type: ignore[import-untyped]
        except ImportError as exc:
            raise VectorStoreError("qdrant-client not installed") from exc

        log.info(
            "Upserting chunks to Qdrant",
            extra={
                "collection": self._collection_name,
                "total_chunks": len(chunks),
                "batch_size": batch_size,
            },
        )

        t_start = time.perf_counter()
        total_upserted = 0

        try:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                points: list[qmodels.PointStruct] = []

                for chunk in batch:
                    payload: dict[str, Any] = {
                        "content": chunk.content,
                        "indexed_at": chunk.indexed_at.isoformat(),
                        **chunk.metadata,
                    }
                    # Use chunk_id as the Qdrant point ID (convert to UUID).
                    try:
                        point_id = str(uuid.UUID(chunk.chunk_id))
                    except ValueError:
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

                    points.append(
                        qmodels.PointStruct(
                            id=point_id,
                            vector=chunk.embedding,
                            payload=payload,
                        )
                    )

                self._client.upsert(
                    collection_name=self._collection_name,
                    points=points,
                    wait=True,
                )
                total_upserted += len(points)

                log.debug(
                    "Batch upserted",
                    extra={
                        "batch": i // batch_size + 1,
                        "total_batches": (len(chunks) + batch_size - 1) // batch_size,
                        "points_in_batch": len(points),
                        "cumulative_upserted": total_upserted,
                    },
                )

            elapsed = time.perf_counter() - t_start
            log.info(
                "Upsert complete",
                extra={
                    "collection": self._collection_name,
                    "total_upserted": total_upserted,
                    "duration_seconds": round(elapsed, 4),
                },
            )
            # All upserted counts are treated as inserts; Qdrant does not
            # distinguish insert vs. update at the client level.
            return UpsertStats(
                total_inserted=total_upserted,
                total_updated=0,
                duration_seconds=round(elapsed, 4),
            )

        except VectorStoreError:
            raise
        except Exception as exc:
            log_exception(log, "upsert_chunks failed", exc)
            raise VectorStoreError(f"Upsert failed: {exc}") from exc

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the *top_k* most similar vectors in the collection.

        Args:
            query_vector: Dense query vector (must match collection dimension).
            top_k:        Maximum number of results to return.
            filters:      Optional filter dict.  Keys are payload field names,
                          values are exact-match strings or ints.  Example::

                              {"source_path": "data/sample.pdf", "page_number": 3}

        Returns:
            List of :class:`~src.vectorstore.schema.SearchResult` ordered by
            descending similarity score.

        Raises:
            VectorStoreError: On Qdrant API failure.
        """
        try:
            from qdrant_client.http import models as qmodels  # type: ignore[import-untyped]

            qdrant_filter: qmodels.Filter | None = None
            if filters:
                must_conditions: list[qmodels.FieldCondition] = []
                for key, value in filters.items():
                    if isinstance(value, int):
                        must_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=value),
                            )
                        )
                    else:
                        must_conditions.append(
                            qmodels.FieldCondition(
                                key=key,
                                match=qmodels.MatchValue(value=str(value)),
                            )
                        )
                qdrant_filter = qmodels.Filter(must=must_conditions)

            # qdrant-client ≥ 1.7 recommends query_points(); fall back to
            # legacy search() for older client versions.
            raw_hits: list[Any]
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self._collection_name,
                    query=query_vector,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                raw_hits = response.points
            else:
                raw_hits = self._client.search(  # type: ignore[attr-defined]
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )

            results: list[SearchResult] = []
            for rank, hit in enumerate(raw_hits, start=1):
                payload: dict[str, Any] = dict(hit.payload or {})
                content: str = payload.pop("content", "")
                # Some payload fields are stored redundantly; remove indexed_at
                # so that metadata is clean.
                payload.pop("indexed_at", None)

                results.append(
                    SearchResult(
                        chunk_id=str(hit.id),
                        content=content,
                        score=float(hit.score),
                        metadata=payload,
                        rank=rank,
                    )
                )

            log.info(
                "Search complete",
                extra={
                    "collection": self._collection_name,
                    "top_k": top_k,
                    "results_returned": len(results),
                    "filters": filters,
                },
            )
            return results

        except VectorStoreError:
            raise
        except Exception as exc:
            log_exception(log, "search failed", exc)
            raise VectorStoreError(f"Search failed: {exc}") from exc

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Verify connectivity to the Qdrant instance.

        Returns:
            ``True`` if the server is reachable and healthy.
        """
        try:
            self._client.get_collections()
            log.debug("Qdrant health check passed")
            return True
        except Exception as exc:
            log.warning("Qdrant health check failed", extra={"error": str(exc)})
            return False

    # ── Delete by source ──────────────────────────────────────────────────────

    def delete_chunks_by_source(self, collection_name: str, source_path: str) -> int:
        """Delete all points whose payload ``source_path`` equals *source_path*.

        The count of deleted points is determined by querying Qdrant's
        ``count()`` with the same filter **before** the delete is issued so
        that the exact number is always available even though Qdrant's delete
        response does not return a removed-points count.

        Args:
            collection_name: Collection to delete from.
            source_path:     Value of the ``source_path`` payload field.

        Returns:
            Number of points that were deleted.

        Raises:
            VectorStoreError: On Qdrant API failure.
        """
        try:
            from qdrant_client.http import models as qmodels  # type: ignore[import-untyped]

            source_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_path",
                        match=qmodels.MatchValue(value=source_path),
                    )
                ]
            )

            # Count before deletion so we can return a meaningful number.
            count_result = self._client.count(
                collection_name=collection_name,
                count_filter=source_filter,
                exact=True,
            )
            deleted_count: int = count_result.count

            if deleted_count > 0:
                self._client.delete(
                    collection_name=collection_name,
                    points_selector=qmodels.FilterSelector(filter=source_filter),
                    wait=True,
                )

            log.info(
                "Deleted chunks by source",
                extra={
                    "collection": collection_name,
                    "source_path": source_path,
                    "deleted_count": deleted_count,
                },
            )
            return deleted_count

        except VectorStoreError:
            raise
        except Exception as exc:
            log_exception(log, "delete_chunks_by_source failed", exc)
            raise VectorStoreError(
                f"Failed to delete chunks for source '{source_path}': {exc}"
            ) from exc

    # ── Source existence check ─────────────────────────────────────────────────

    def source_exists(self, collection_name: str, source_path: str) -> bool:
        """Return ``True`` if at least one chunk with *source_path* exists.

        Calls Qdrant's ``count()`` with an exact-match filter on the
        ``source_path`` payload field.  The operation is O(1) — it does not
        fetch any vectors or payloads — so it adds negligible overhead before
        each index request.

        If the collection does not exist yet (first-time upload), the underlying
        Qdrant call raises an exception that is caught here and treated as
        ``False`` rather than propagated as an error.

        Args:
            collection_name: Collection to inspect.
            source_path:     Value to look up in the ``source_path`` payload
                             field.

        Returns:
            ``True`` when count > 0, ``False`` otherwise.

        Raises:
            VectorStoreError: On unexpected Qdrant API failure (not collection-
                              not-found, which is silently treated as ``False``).
        """
        try:
            from qdrant_client.http import models as qmodels  # type: ignore[import-untyped]

            source_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_path",
                        match=qmodels.MatchValue(value=source_path),
                    )
                ]
            )
            count_result = self._client.count(
                collection_name=collection_name,
                count_filter=source_filter,
                exact=True,
            )
            exists: bool = count_result.count > 0
            log.debug(
                "source_exists check",
                extra={
                    "collection": collection_name,
                    "source_path": source_path,
                    "count": count_result.count,
                    "exists": exists,
                },
            )
            return exists

        except Exception as exc:
            # Collection not found → treat as "not yet indexed" so the caller
            # proceeds with normal ingestion.
            err_str = str(exc).lower()
            if "not found" in err_str or "doesn't exist" in err_str:
                log.debug(
                    "source_exists: collection not found, treating as False",
                    extra={"collection": collection_name, "source_path": source_path},
                )
                return False
            log_exception(log, "source_exists failed", exc)
            raise VectorStoreError(
                f"Failed to check existence for source '{source_path}': {exc}"
            ) from exc

    # ── Bulk fetch (sparse index warm-up) ────────────────────────────────────
    # Long-term: migrate to Qdrant native sparse vectors (BM42) to eliminate
    # this RAM index and the need for a warm-up scroll at startup.

    def get_all_chunks(
        self,
        collection_name: str,
        batch_size: int = 100,
    ) -> list[IndexedChunk]:
        """Page through every point in *collection_name* via Qdrant ``scroll()``
        and reconstruct :class:`~src.vectorstore.schema.IndexedChunk` objects.

        Used during API lifespan startup to pre-populate the in-memory
        :class:`~src.retrieval.sparse_retriever.SparseRetriever` corpus so the
        BM25 index is not empty after every restart.

        Args:
            collection_name: Qdrant collection to read from.
            batch_size:      Number of points fetched per scroll page (controls
                             memory pressure vs. round-trip count).

        Returns:
            List of :class:`IndexedChunk` objects reconstructed from stored
            payloads.  Returns an empty list if the collection does not exist
            or is empty — never raises for a missing collection.

        Raises:
            VectorStoreError: On unexpected Qdrant API failure.
        """
        try:
            t_start = time.perf_counter()

            # Verify the collection exists before scrolling; missing collection
            # is a normal state on first boot and should not be an error.
            existing = {c.name for c in self._client.get_collections().collections}
            if collection_name not in existing:
                log.info(
                    "get_all_chunks: collection not found — returning empty list",
                    extra={"collection": collection_name},
                )
                return []

            chunks: list[IndexedChunk] = []
            offset: str | None = None  # Qdrant scroll cursor (None = start)

            while True:
                scroll_result = self._client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # vectors not needed for BM25
                )
                points, next_offset = scroll_result

                for point in points:
                    payload: dict[str, Any] = dict(point.payload or {})
                    content: str = payload.pop("content", "")
                    indexed_at_raw: str = payload.pop("indexed_at", "")

                    try:
                        from datetime import datetime, timezone
                        indexed_at = (
                            datetime.fromisoformat(indexed_at_raw)
                            if indexed_at_raw
                            else datetime.now(tz=timezone.utc)
                        )
                    except ValueError:
                        from datetime import datetime, timezone
                        indexed_at = datetime.now(tz=timezone.utc)

                    chunks.append(
                        IndexedChunk(
                            chunk_id=str(point.id),
                            content=content,
                            embedding=[],  # not fetched — BM25 needs text only
                            metadata=payload,
                            indexed_at=indexed_at,
                        )
                    )

                if next_offset is None:
                    break
                offset = next_offset

            elapsed = time.perf_counter() - t_start
            log.info(
                "get_all_chunks complete",
                extra={
                    "collection": collection_name,
                    "total_chunks": len(chunks),
                    "duration_seconds": round(elapsed, 4),
                },
            )
            return chunks

        except VectorStoreError:
            raise
        except Exception as exc:
            log_exception(log, "get_all_chunks failed", exc)
            raise VectorStoreError(
                f"Failed to fetch all chunks from '{collection_name}': {exc}"
            ) from exc
