"""Corte Costituzionale connector — cortecostituzionale.it.

Fetches Italian Constitutional Court decisions (sentenze, ordinanze)
from the official search portal at cortecostituzionale.it.

Authentication: None required.
API:            JSON search endpoint at /servizi/pronunce.
Pagination:     Page-based (``pagina`` parameter).

Each decision includes:
  - ``numero``      : decision number within the year
  - ``anno``        : year
  - ``tipo``        : ``"Sentenza"`` | ``"Ordinanza"``
  - ``data``        : filing date
  - ``argomento``   : thematic classification
  - ``testo``       : full text of the decision
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any
from urllib.parse import urljoin

from src.ingestion.loader import Document
from src.italia.connectors.base import BaseConnector
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)

_BASE_URL = "https://www.cortecostituzionale.it"
_SEARCH_API = f"{_BASE_URL}/servizi/pronunce"
_DETAIL_URL = f"{_BASE_URL}/actionPronuncia.do"


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    from datetime import datetime  # noqa: PLC0415
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


class CorteCostituzionaleConnector(BaseConnector):
    """Fetches Constitutional Court decisions from cortecostituzionale.it.

    Args:
        rate_limit_rps: Max req/s (default 0.5).
        timeout:        HTTP timeout in seconds.

    Example::

        conn = CorteCostituzionaleConnector()
        docs = conn.fetch(anno=2023, limit=20)
        # or
        docs = conn.fetch(argomento="privacy", limit=10)
    """

    source_name = "corte_costituzionale"

    def __init__(self, rate_limit_rps: float = 0.5, timeout: int = 45) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)

    def fetch(  # type: ignore[override]
        self,
        anno: int | None = None,
        argomento: str | None = None,
        query: str = "",
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 30,
        **_: Any,
    ) -> list[Document]:
        """Fetch Constitutional Court decisions.

        Args:
            anno:      Filter by year (e.g. ``2023``).
            argomento: Thematic area (e.g. ``"privacy"``, ``"lavoro"``).
            query:     Full-text search query.
            from_date: Start date filter.
            to_date:   End date filter.
            limit:     Max decisions to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        log.info(
            "Fetching from Corte Costituzionale",
            extra={"anno": anno, "argomento": argomento, "limit": limit},
        )

        items = list(self._search(anno, argomento, query, from_date, to_date, limit))
        documents: list[Document] = []

        for item in items[:limit]:
            try:
                doc = self._build_document(item)
                if doc:
                    documents.append(doc)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to build Corte Cost. document",
                    extra={"item": str(item)[:100], "error": str(exc)},
                )

        log.info("Corte Costituzionale fetch complete", extra={"count": len(documents)})
        return documents

    # ── Private helpers ───────────────────────────────────────────────────────

    def _search(
        self,
        anno: int | None,
        argomento: str | None,
        query: str,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> Any:
        params: dict[str, Any] = {
            "rows": min(limit, 20),
            "formato": "json",
        }
        if anno:
            params["anno"] = anno
        if argomento:
            params["argomento"] = argomento
        if query:
            params["testo"] = query
        if from_date:
            params["dataDecisioneDal"] = from_date.strftime("%d/%m/%Y")
        if to_date:
            params["dataDecisioneAl"] = to_date.strftime("%d/%m/%Y")

        for item in self._paginate(
            url=_SEARCH_API,
            params=params,
            page_key="pagina",
            size_key="rows",
            page_size=min(limit, 20),
            max_pages=(limit // 20) + 2,
            results_key="pronunce",
        ):
            yield item

    def _build_document(self, item: dict[str, Any]) -> Document | None:
        """Convert a raw JSON item to a Document."""
        numero = str(item.get("numero") or item.get("num") or "")
        anno_val = item.get("anno")
        tipo_str: str = item.get("tipo") or item.get("tipoProvvedimento") or "Sentenza"
        argomento_val = item.get("argomento") or item.get("materia") or ""
        raw_date = item.get("dataDecisione") or item.get("data") or ""

        # Try to fetch full text; fall back to available summary/abstract.
        testo = item.get("testo") or item.get("testoPronuncia") or ""
        if not testo and numero and anno_val:
            testo = self._fetch_full_text(numero, str(anno_val))

        if not testo:
            return None

        try:
            anno = int(anno_val) if anno_val else None
        except (ValueError, TypeError):
            anno = None

        dep_date = _parse_date(raw_date)

        numero_sentenza: str | None = None
        if numero:
            anno_suffix = f"/{anno}" if anno else ""
            tipo_abbrev = "Ord." if "ORDIN" in tipo_str.upper() else "Sent."
            numero_sentenza = f"C. Cost. {tipo_abbrev} n. {numero}{anno_suffix}"

        materia: list[str] = (
            [argomento_val] if isinstance(argomento_val, str) and argomento_val
            else (argomento_val if isinstance(argomento_val, list) else [])
        )

        meta = ItalianLegalMetadata(
            fonte=self.source_name,
            tipo_documento=TipoDocumento.SENTENZA_COSTITUZIONALE,
            anno=anno,
            numero_sentenza=numero_sentenza,
            data_deposito=dep_date,
            materia=materia,
        )

        source_uri = (
            item.get("url")
            or f"{_DETAIL_URL}?anno={anno}&numero={numero}"
        )
        return self._make_document(
            content=testo,
            italian_meta=meta,
            source_uri=source_uri,
            filename=f"ccost_{numero}_{anno}.txt",
        )

    def _fetch_full_text(self, numero: str, anno: str) -> str:
        """Attempt to fetch the full decision text from the detail endpoint."""
        try:
            resp = self._get(
                _DETAIL_URL,
                params={"anno": anno, "numero": numero},
            )
            # Try to extract text from HTML.
            try:
                from bs4 import BeautifulSoup  # noqa: PLC0415
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup.select("nav, header, footer, script, style"):
                    tag.decompose()
                main = soup.select_one("main, article, .pronuncia, #testo")
                return (main or soup).get_text(separator="\n", strip=True)
            except ImportError:
                return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", resp.text)).strip()
        except Exception as exc:  # noqa: BLE001
            log.debug("Could not fetch full text", extra={"error": str(exc)})
            return ""
