"""AGCM connector — Antitrust Authority provvedimenti (agcm.it).

Fetches competition and antitrust decisions (provvedimenti) from the
Italian Antitrust Authority (Autorità Garante della Concorrenza e del
Mercato) via its public search API and HTML pages.

Authentication: None required.
Primary source: https://www.agcm.it/dotcmsplugin/servlet/RicercaDocumentiServlet
Format:         JSON + HTML detail pages.

Each provvedimento includes:
  - Procedure number (``numeroProvvedimento``)
  - Date (``dataProvvedimento``)
  - Subject companies and practices (``descrizione``)
  - Full decision text from the detail page
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

_BASE_URL = "https://www.agcm.it"
_SEARCH_API = f"{_BASE_URL}/dotcmsplugin/servlet/RicercaDocumentiServlet"


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    from datetime import datetime  # noqa: PLC0415
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


class AGCMConnector(BaseConnector):
    """Fetches AGCM antitrust authority provvedimenti.

    Args:
        rate_limit_rps: Max req/s (default 0.5).
        timeout:        HTTP timeout in seconds.

    Example::

        conn = AGCMConnector()
        docs = conn.fetch(query="pratiche commerciali scorrette", limit=20)
    """

    source_name = "agcm"

    def __init__(self, rate_limit_rps: float = 0.5, timeout: int = 45) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)

    def fetch(  # type: ignore[override]
        self,
        query: str = "",
        tipo: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 30,
        **_: Any,
    ) -> list[Document]:
        """Fetch AGCM provvedimenti.

        Args:
            query:     Full-text query (Italian).
            tipo:      Procedure type filter (e.g. ``"A"`` = antitrust, ``"PS"`` = consumer).
            from_date: Start date filter.
            to_date:   End date filter.
            limit:     Max provvedimenti to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        log.info(
            "Fetching from AGCM",
            extra={"query": query, "tipo": tipo, "limit": limit},
        )

        results = self._search(query, tipo, from_date, to_date, limit)
        documents: list[Document] = []

        for item in results[:limit]:
            try:
                doc = self._fetch_provvedimento(item)
                if doc:
                    documents.append(doc)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to fetch AGCM provvedimento",
                    extra={"item": str(item)[:100], "error": str(exc)},
                )

        log.info("AGCM fetch complete", extra={"count": len(documents)})
        return documents

    # ── Private helpers ───────────────────────────────────────────────────────

    def _search(
        self,
        query: str,
        tipo: str | None,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "q": query or "provvedimento",
            "tipoDocumento": tipo or "",
            "start": 0,
            "rows": min(limit, 20),
            "format": "json",
        }
        if from_date:
            params["dataDal"] = from_date.strftime("%d/%m/%Y")
        if to_date:
            params["dataAl"] = to_date.strftime("%d/%m/%Y")

        try:
            resp = self._get(_SEARCH_API, params=params)
            data = resp.json()
            return (
                data.get("documenti", [])
                or data.get("results", [])
                or data.get("items", [])
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("AGCM search API failed", extra={"error": str(exc)})
            return []

    def _fetch_provvedimento(self, item: dict[str, Any]) -> Document | None:
        """Fetch and parse a single AGCM provvedimento."""
        url = item.get("url") or item.get("link") or item.get("urlDettaglio") or ""
        if url and not url.startswith("http"):
            url = urljoin(_BASE_URL, url)

        # Build metadata from search result fields first.
        numero = str(item.get("numeroProvvedimento") or item.get("numero") or "")
        raw_date = item.get("dataProvvedimento") or item.get("data") or ""
        descrizione = item.get("oggetto") or item.get("descrizione") or item.get("titolo") or ""

        dep_date = _parse_date(raw_date)
        anno = dep_date.year if dep_date else None

        # Fetch full text from detail URL if available.
        full_text = item.get("testo") or ""
        if not full_text and url:
            try:
                resp = self._get(url)
                full_text = self._extract_text(resp.text)
            except Exception:  # noqa: BLE001
                full_text = descrizione

        if not full_text:
            full_text = descrizione
        if not full_text:
            return None

        numero_sentenza: str | None = f"AGCM Prov. {numero}" if numero else None

        meta = ItalianLegalMetadata(
            fonte=self.source_name,
            tipo_documento=TipoDocumento.PROVVEDIMENTO,
            anno=anno,
            numero_sentenza=numero_sentenza,
            data_deposito=dep_date,
            materia=["diritto della concorrenza", "antitrust"],
            parole_chiave=["AGCM", "antitrust", "concorrenza"],
        )
        source_uri = url or f"agcm://provvedimento/{numero}"
        return self._make_document(
            content=full_text,
            italian_meta=meta,
            source_uri=source_uri,
            filename=f"agcm_{numero}_{anno}.txt",
        )

    @staticmethod
    def _extract_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.select("nav, header, footer, script, style"):
                tag.decompose()
            main = soup.select_one("main, article, .content, #contenuto")
            return (main or soup).get_text(separator="\n", strip=True)
        except ImportError:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
