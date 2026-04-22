"""Immutable audit log for EU AI Act compliance (Annex III — high-risk AI).

Every query submitted to ``/lexreview/query`` must produce an audit record in
``logs/audit.jsonl``.  The file is opened in **append mode** (``"a"``) and
never truncated by this module, giving an immutable append-only audit trail.

Privacy constraints (GDPR Art. 5 — data minimisation):
* The **raw query text is never written**.  Only a SHA-256 hash is stored so
  that records can be correlated with application logs without re-exposing PII.
* All other fields (request ID, collection, model, confidence) are
  non-personal operational metadata.

EU AI Act obligations implemented here:
* Audit trail per query (Art. 12 — record-keeping for high-risk systems).
* Confidence score logged to support post-hoc human review triggers.

Usage::

    from src.utils.audit_log import AuditLogger

    audit = AuditLogger()                      # writes to logs/audit.jsonl
    audit.record(
        request_id="abc-123",
        query="...",                           # hashed internally
        collection_name="ragforge_italia",
        model="gpt-4o-mini",
        confidence=0.87,
    )
"""

from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import IO


class AuditLogger:
    """Thread-safe append-only audit log writer.

    Records are written as newline-delimited JSON (NDJSON) to the file
    specified by *log_path*.  The file handle is kept open for the lifetime
    of the instance; call :meth:`close` (or use as a context manager) to
    flush and release it.

    Args:
        log_path: Path to the audit JSONL file.  Parent directories are
                  created automatically.  Defaults to ``logs/audit.jsonl``
                  relative to the current working directory.

    Example::

        with AuditLogger() as audit:
            audit.record(
                request_id="req-1",
                query="What is Art. 1 CC?",
                collection_name="italia",
                model="gpt-4o-mini",
                confidence=0.91,
            )
    """

    def __init__(self, log_path: str | Path = "logs/audit.jsonl") -> None:
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode — never truncate.
        self._fh: IO[str] = self._path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def record(
        self,
        *,
        request_id: str,
        query: str,
        collection_name: str,
        model: str,
        confidence: float,
    ) -> None:
        """Write one immutable audit record.

        The raw *query* is **never** written.  Only its SHA-256 hex digest is
        stored so that records can be linked to application logs without
        re-exposing potential PII.

        Args:
            request_id:      Trace identifier from ``X-Request-ID`` header.
            query:           The raw user query (hashed before storage).
            collection_name: Qdrant collection the query was executed against.
            model:           LLM model name used for generation.
            confidence:      ``AgentResponse.confidence`` score in [0, 1].
        """
        entry: dict = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "request_id": request_id,
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "collection_name": collection_name,
            "model": model,
            "confidence": round(confidence, 6),
            # Required by EU AI Act Art. 13 (transparency) — always present.
            "system": "ragforge-italia-legal-ai",
            "requires_human_review": True,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "AuditLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ── Module-level singleton ────────────────────────────────────────────────────
# Imported once at startup by the FastAPI lifespan and stored on app.state.
# This ensures a single file handle is shared across all requests in a worker,
# avoiding the overhead of opening/closing the file on every query.

_default_audit_logger: AuditLogger | None = None


def get_audit_logger(log_path: str | Path = "logs/audit.jsonl") -> AuditLogger:
    """Return (or lazily create) the process-wide :class:`AuditLogger` singleton.

    Args:
        log_path: Passed to :class:`AuditLogger` only on first call.

    Returns:
        The shared :class:`AuditLogger` instance.
    """
    global _default_audit_logger
    if _default_audit_logger is None:
        _default_audit_logger = AuditLogger(log_path=log_path)
    return _default_audit_logger
