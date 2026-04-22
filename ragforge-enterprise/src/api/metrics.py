"""Prometheus metrics definitions and exposition endpoint.

This module defines the application-level Prometheus metrics and exposes a
helper :func:`get_metrics_response` that serialises the current registry into
the Prometheus text exposition format, suitable for scraping by a Prometheus
server or a compatible agent (e.g. Grafana Agent, VictoriaMetrics).

Metrics defined
---------------
``ragforge_requests_total``
    Counter — total HTTP requests processed, labelled by ``method``, ``path``,
    and ``status_code``.  Incremented by the metrics-aware middleware (or by
    the ``/metrics`` endpoint handler via helper functions exported here).

``ragforge_request_latency_seconds``
    Histogram — per-request latency in seconds, labelled by ``method`` and
    ``path``.  Default Prometheus buckets are used.

``ragforge_llm_tokens_used_total``
    Counter — total LLM tokens consumed, labelled by ``provider``.  This
    counter is **incremented externally** (e.g. from the LLM client layer);
    this module only defines and exports it so that the definition is
    centralised and available to the ``/metrics`` scrape endpoint.

Usage
-----
Import the counter/histogram objects in other modules and call their ``labels``
method to obtain a child metric before incrementing::

    from src.api.metrics import REQUESTS_TOTAL, REQUEST_LATENCY, LLM_TOKENS_TOTAL

    REQUESTS_TOTAL.labels(method="POST", path="/lexreview/query", status_code="200").inc()
    LLM_TOKENS_TOTAL.labels(provider="openai").inc(512)
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.responses import PlainTextResponse

# ── Metric definitions ────────────────────────────────────────────────────────

REQUESTS_TOTAL: Counter = Counter(
    name="ragforge_requests_total",
    documentation="Total HTTP requests processed by RAGForge Enterprise.",
    labelnames=["method", "path", "status_code"],
)

REQUEST_LATENCY: Histogram = Histogram(
    name="ragforge_request_latency_seconds",
    documentation="Per-request latency in seconds.",
    labelnames=["method", "path"],
)

LLM_TOKENS_TOTAL: Counter = Counter(
    name="ragforge_llm_tokens_used_total",
    documentation=(
        "Total LLM tokens consumed by RAGForge Enterprise, labelled by provider. "
        "Incremented externally by the LLM client layer."
    ),
    labelnames=["provider"],
)


# ── Scrape helper ─────────────────────────────────────────────────────────────


def get_metrics_response() -> PlainTextResponse:
    """Return a :class:`~starlette.responses.PlainTextResponse` in Prometheus exposition format.

    The response body is produced by :func:`prometheus_client.generate_latest`
    using the default (global) registry, which includes all metrics defined in
    this module plus any process/platform metrics collected by the
    ``prometheus_client`` library.

    Returns:
        A ``PlainTextResponse`` with ``Content-Type`` set to the value required
        by the Prometheus scrape protocol
        (``text/plain; version=0.0.4; charset=utf-8``).
    """
    return PlainTextResponse(
        content=generate_latest().decode("utf-8"),
        media_type=CONTENT_TYPE_LATEST,
    )
