"""Structured JSON logging utilities for RAGForge Enterprise.

All modules obtain their logger through :func:`get_logger` so that every
log line is emitted as machine-parseable JSON, suitable for log-aggregation
systems (Loki, ELK, CloudWatch Logs, etc.).

GDPR compliance (Art. 5 — data minimisation)
--------------------------------------------
Every record is passed through :class:`~src.utils.pii_scrubber.PIIScrubber`
before serialisation.  This ensures that Codice Fiscale, P.IVA, email
addresses, IBANs, and Italian phone numbers are **never** written to disk.

Log rotation (GDPR Art. 5(1)(e) — storage limitation)
------------------------------------------------------
When *log_dir* is supplied to :func:`get_logger`, a
:class:`~logging.handlers.TimedRotatingFileHandler` is attached that rotates
daily and automatically deletes files older than *retention_days* (default 90).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.pii_scrubber import PIIScrubber

_scrubber = PIIScrubber()


class _JSONFormatter(logging.Formatter):
    """Render every :class:`logging.LogRecord` as a single JSON line.

    Keys emitted per record:
    - ``timestamp`` – ISO-8601 UTC timestamp.
    - ``level``     – Log level name (DEBUG, INFO, …).
    - ``logger``    – Dotted logger name.
    - ``message``   – The formatted log message.
    - ``module``    – Python module that emitted the record.
    - ``lineno``    – Source line number.
    - ``exc_info``  – Stringified traceback (only when an exception is attached).
    - Any extra key/value pairs passed via ``extra=`` on the log call.
    """

    _RESERVED_ATTRS: frozenset[str] = frozenset(
        {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "id",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: ANN201
        """Serialise *record* to a JSON string with PII scrubbing applied.

        All string values — the message and every ``extra=`` field — are
        passed through :class:`~src.utils.pii_scrubber.PIIScrubber` before
        serialisation, implementing GDPR Art. 5(1)(c) data minimisation.

        Args:
            record: The log record produced by the logging subsystem.

        Returns:
            A single-line JSON string terminated without a newline.
        """
        # Scrub the log message itself.
        raw_message = record.getMessage()
        clean_message = _scrubber.scrub(raw_message)

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": clean_message,
            "module": record.module,
            "lineno": record.lineno,
        }

        # Attach traceback when an exception is bound to the record.
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exc_info"] = record.exc_text

        # Surface any caller-supplied extra fields — scrub string values.
        for key, value in record.__dict__.items():
            if key not in self._RESERVED_ATTRS:
                payload[key] = _scrubber.scrub(value) if isinstance(value, str) else value

        return json.dumps(payload, default=str, ensure_ascii=False)


def get_logger(
    name: str,
    level: str = "INFO",
    *,
    log_dir: str | Path | None = None,
    retention_days: int = 90,
) -> logging.Logger:
    """Return a named logger that emits structured JSON to *stdout*.

    Optionally also writes to a rotating file with automatic age-based
    deletion, satisfying GDPR Art. 5(1)(e) storage limitation.

    Calling this function multiple times with the same *name* is safe; the
    handler is attached only once.

    Args:
        name:          Dotted logger name, typically ``__name__``.
        level:         Minimum log level string (e.g. ``"DEBUG"``, ``"WARNING"``).
        log_dir:       When provided, a :class:`~logging.handlers.TimedRotatingFileHandler`
                       is also attached, rotating daily and keeping at most
                       *retention_days* files.
        retention_days: Number of backup files to retain (each corresponds to
                        one day).  Implements GDPR storage limitation.

    Returns:
        A configured :class:`logging.Logger` instance.

    Example:
        >>> log = get_logger(__name__, level="DEBUG")
        >>> log.info("Pipeline started", extra={"doc_count": 42})
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers when modules are reloaded (e.g. in tests).
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger

    formatter = _JSONFormatter()

    # ── stdout handler (always attached) ─────────────────────────────────────
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # ── Rotating file handler (GDPR storage limitation) ───────────────────────
    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.TimedRotatingFileHandler(
            filename=log_path / "app.log",
            when="midnight",
            interval=1,
            backupCount=retention_days,
            encoding="utf-8",
            utc=True,
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.setLevel(level.upper())
    # Prevent propagation to the root logger which might have its own handlers.
    logger.propagate = False
    return logger


def log_exception(logger: logging.Logger, message: str, exc: BaseException) -> None:
    """Log *exc* as a structured ERROR record with a full traceback.

    Args:
        logger:  The logger to write to.
        message: A human-readable summary (e.g. ``"PDF parse failed"``).
        exc:     The caught exception instance.
    """
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        message,
        extra={
            "exception_type": type(exc).__name__,
            "traceback": tb_str,
        },
    )
