import logging
import json
import time
import os
from typing import Optional


def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )


class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def log_query(
        self,
        query: str,
        sql: str,
        rows: int,
        latency_ms: float,
        tokens_used: Optional[int] = None,
        cached: bool = False
    ):
        self.logger.info(json.dumps({
            "event": "sql_query",
            "query_preview": query[:100],
            "sql_length": len(sql),
            "rows_returned": rows,
            "latency_ms": round(latency_ms, 2),
            "tokens_used": tokens_used,
            "cached": cached
        }))

    def log_rag_search(self, query: str, results_count: int, latency_ms: float):
        self.logger.info(json.dumps({
            "event": "rag_search",
            "query_preview": query[:100],
            "results_count": results_count,
            "latency_ms": round(latency_ms, 2)
        }))

    def log_agent(self, session_id: str, tools_used: list, latency_ms: float):
        self.logger.info(json.dumps({
            "event": "agent_invocation",
            "session_id": session_id,
            "tools_used": tools_used,
            "latency_ms": round(latency_ms, 2)
        }))

    def log_pdf_ingest(self, filename: str, pages: int, latency_ms: float):
        self.logger.info(json.dumps({
            "event": "pdf_ingest",
            "filename": filename,
            "pages": pages,
            "latency_ms": round(latency_ms, 2)
        }))

    def log_error(self, event: str, error: str, context: dict = {}):
        self.logger.error(json.dumps({
            "event": event,
            "error": str(error)[:500],
            **context
        }))
