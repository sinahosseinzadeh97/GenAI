"""TAR/Consiglio di Stato connector — giustizia-amministrativa.it.

Fetches Italian administrative justice decisions (sentenze TAR and
Consiglio di Stato) from the official portal giustizia-amministrativa.it.

Authentication: None required for public access.
Format:         HTML scraping via BeautifulSoup.
Pagination:     Offset-based search results.

The connector targets the public search endpoint at:
    https://www.giustizia-amministrativa.it/web/guest/dcsnprr

Each sentenza is fetched as a standalone HTML page and converted to text.
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

_BASE_URL = "https://www.giustizia-amministrativa.it"
_SEARCH_URL = f"{_BASE_URL}/web/guest/ricerca-fulltext"
_SEARCH_API = f"{_BASE_URL}/home/ricerca.json"


def _guess_tipo(organ: str) -> TipoDocumento:
    organ_up = organ.upper()
    if "CONSIGLIO" in organ_up and "STATO" in organ_up:
        # Consiglio di Stato is the appellate court — still SENTENZA_TAR here
        # (we don't have a separate enum; can be differentiated via sezione).
        return TipoDocumento.SENTENZA_TAR
    return TipoDocumento.SENTENZA_TAR


class TARConnector(BaseConnector):
    """Fetches TAR and Consiglio di Stato sentenze.

    Args:
        rate_limit_rps: Max req/s (default 0.5).
        timeout:        HTTP timeout in seconds.

    Example::

        conn = TARConnector()
        docs = conn.fetch(
            query="appalti pubblici",
            from_date=date(2023, 6, 1),
            limit=25,
        )
    """

    source_name = "tar"

    def __init__(self, rate_limit_rps: float = 0.5, timeout: int = 45) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)

    def fetch(  # type: ignore[override]
        self,
        query: str = "",
        organo: str | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 30,
        **_: Any,
    ) -> list[Document]:
        """Fetch TAR/Consiglio di Stato sentenze.

        Args:
            query:    Full-text query (Italian).
            organo:   Organo di giustizia filter (e.g. ``"TAR Lazio"``).
            from_date: Start date filter.
            to_date:   End date filter.
            limit:     Max sentenze to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        log.info(
            "Fetching from giustizia-amministrativa.it",
            extra={"query": query, "organo": organo, "limit": limit},
        )

        results = self._search(query, organo, from_date, to_date, limit)
        documents: list[Document] = []

        for item in results[:limit]:
            try:
                doc = self._fetch_sentenza(item)
                if doc:
                    documents.append(doc)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to fetch TAR sentenza",
                    extra={"item": str(item)[:100], "error": str(exc)},
                )

        log.info("TAR fetch complete", extra={"count": len(documents)})
        return documents

    # ── Private helpers ───────────────────────────────────────────────────────

    def _search(
        self,
        query: str,
        organo: str | None,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Query the search API and return raw result metadata."""
        params: dict[str, Any] = {
            "q": query or "sentenza",
            "start": 0,
            "rows": min(limit, 20),
        }
        if organo:
            params["organo"] = organo
        if from_date:
            params["dataDal"] = from_date.strftime("%Y-%m-%d")
        if to_date:
            params["dataAl"] = to_date.strftime("%Y-%m-%d")

        try:
            resp = self._get(_SEARCH_API, params=params)
            data = resp.json()
            return data.get("response", {}).get("docs", []) or data.get("docs", [])
        except Exception as exc:  # noqa: BLE001
            log.warning("TAR search API failed", extra={"error": str(exc)})
            return self._fallback_search(query, limit)

    def _fallback_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """HTML scraping fallback for the search results page."""
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
        except ImportError:
            log.warning("beautifulsoup4 not installed; cannot scrape TAR")
            return []

        params = {"q": query or "sentenza", "start": 0}
        try:
            resp = self._get(_SEARCH_URL, params=params)
            soup = BeautifulSoup(resp.text, "html.parser")
            items = []
            for link in soup.select("a.risultato-link, a[href*='sentenza']")[:limit]:
                href = link.get("href", "")
                if href:
                    items.append({"url": urljoin(_BASE_URL, href), "title": link.get_text()})
            return items
        except Exception as exc:  # noqa: BLE001
            log.warning("TAR HTML fallback failed", extra={"error": str(exc)})
            return []

    def _fetch_sentenza(self, item: dict[str, Any]) -> Document | None:
        """Fetch and parse a single sentenza page."""
        url = item.get("url") or item.get("link") or item.get("urlDettaglio") or ""
        if url and not url.startswith("http"):
            url = urljoin(_BASE_URL, url)

        if not url:
            # Construct from fields if possible
            doc_id = item.get("id") or item.get("idDocumento")
            if doc_id:
                url = f"{_BASE_URL}/web/guest/dcsnprr?doc={doc_id}"
            else:
                return None

        resp = self._get(url)
        text = self._extract_text(resp.text)
        if not text.strip():
            return None

        title: str = item.get("title") or item.get("titolo") or ""
        organo: str = item.get("organo") or item.get("organoDecidente") or ""
        numero: str = item.get("numero") or item.get("numeroSentenza") or ""
        raw_date: str = item.get("data") or item.get("dataDeposizione") or ""

        dep_date: date | None = None
        if raw_date:
            for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
                try:
                    from datetime import datetime  # noqa: PLC0415
                    dep_date = datetime.strptime(raw_date[:10], fmt).date()
                    break
                except ValueError:
                    continue

        anno = dep_date.year if dep_date else None
        tipo = _guess_tipo(organo)
        numero_sentenza: str | None = None
        if numero:
            anno_suffix = f"/{anno}" if anno else ""
            short_organo = (
                "C.d.S." if "CONSIGLIO" in organo.upper() else "TAR"
            )
            numero_sentenza = f"{short_organo} n. {numero}{anno_suffix}"

        meta = ItalianLegalMetadata(
            fonte=self.source_name,
            tipo_documento=tipo,
            anno=anno,
            numero_sentenza=numero_sentenza,
            sezione=organo or None,
            data_deposito=dep_date,
        )
        return self._make_document(
            content=text,
            italian_meta=meta,
            source_uri=url,
            filename=f"tar_{numero or 'unknown'}_{anno}.html",
        )

    @staticmethod
    def _extract_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415

            soup = BeautifulSoup(html, "html.parser")
            # Remove nav, header, footer noise.
            for tag in soup.select("nav, header, footer, script, style, .navbar, .breadcrumb"):
                tag.decompose()
            main = soup.select_one("main, article, .sentenza-text, #contenuto, .portlet-body")
            return (main or soup).get_text(separator="\n", strip=True)
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()
