"""DeJure (Giuffrè) connector — legal doctrine and massime.

DeJure is a **paid subscription** service operated by Giuffrè Francis Lefebvre
that provides access to Italian legal doctrine, comprehensive jurisprudence
(massime), and annotated codes.

**Access model**: Full API access requires a paid institutional licence from
Giuffrè.  Without a valid ``DEJURE_API_KEY``, this connector raises
:class:`DeJureAccessError` with clear guidance on how to obtain access.

Fallback
--------
When ``fallback_public=True`` (the default), the connector falls back to
scraping *publicly accessible* massime from ``dejure.it`` — the free web
interface — which exposes headnotes for referenced cases without requiring
authentication.  This subset is legally in the public domain as judicial
decisions, but coverage is significantly reduced compared to the full API.

Authentication
--------------
Set the ``DEJURE_API_KEY`` environment variable (or ``italia_dejure_api_key``
in ``.env``) to enable full API access.  Contact Giuffrè at:
  https://www.giuffre.it/riviste-e-banche-dati/dejure

API base URL (when authenticated):
  https://api.dejure.it/v1
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any
from urllib.parse import urljoin, quote

from src.ingestion.loader import Document
from src.italia.connectors.base import BaseConnector, ConnectorError
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)

_DEJURE_PUBLIC_BASE = "https://dejure.it"
_DEJURE_SEARCH_URL = f"{_DEJURE_PUBLIC_BASE}/ricerca"
_DEJURE_API_BASE = "https://api.dejure.it/v1"


class DeJureAccessError(ConnectorError):
    """Raised when DeJure API key is missing and public fallback is disabled."""


class DeJureConnector(BaseConnector):
    """Fetches Italian legal doctrine and massime from DeJure / Giuffrè.

    Full API mode requires a valid ``api_key``.  Without it, only publicly
    accessible massime from dejure.it are fetched (``fallback_public=True``).

    Args:
        api_key:        DeJure API key (from ``DEJURE_API_KEY`` env var or
                        direct injection).  ``None`` → fallback or error.
        fallback_public: When ``True`` (default), use public massime scraping
                         when ``api_key`` is absent.
        rate_limit_rps: Max req/s (default 0.3 — conservative for DeJure).
        timeout:        HTTP timeout in seconds.

    Raises:
        :class:`DeJureAccessError`: When ``api_key`` is ``None`` and
            ``fallback_public=False``.

    Example::

        # Public massime only (no key required):
        conn = DeJureConnector()
        docs = conn.fetch(query="responsabilità medica", limit=10)

        # Full API (key required):
        import os
        conn = DeJureConnector(api_key=os.getenv("DEJURE_API_KEY"))
        docs = conn.fetch(query="diritto del lavoro subordinato", limit=50)
    """

    source_name = "dejure"

    def __init__(
        self,
        api_key: str | None = None,
        fallback_public: bool = True,
        rate_limit_rps: float = 0.3,
        timeout: int = 45,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout, headers=headers)
        self._api_key = api_key
        self._fallback_public = fallback_public

        if not api_key and not fallback_public:
            raise DeJureAccessError(
                "DeJure API key is required (set DEJURE_API_KEY env var). "
                "Obtain access at https://www.giuffre.it/riviste-e-banche-dati/dejure\n"
                "Set fallback_public=True to use public massime as a fallback."
            )

    def fetch(  # type: ignore[override]
        self,
        query: str = "",
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 30,
        **_: Any,
    ) -> list[Document]:
        """Fetch DeJure massime and doctrine.

        Args:
            query:     Full-text query (Italian legal terms).
            from_date: Start date filter.
            to_date:   End date filter.
            limit:     Max documents to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.

        Raises:
            :class:`DeJureAccessError`: When API key is missing and public
                fallback is disabled.
        """
        if self._api_key:
            log.info(
                "Fetching from DeJure API",
                extra={"query": query, "limit": limit, "mode": "api"},
            )
            return self._fetch_api(query, from_date, to_date, limit)

        log.info(
            "Fetching from DeJure (public massime fallback)",
            extra={"query": query, "limit": limit, "mode": "public"},
        )
        return self._fetch_public(query, from_date, to_date, limit)

    # ── Full API path (requires api_key) ──────────────────────────────────────

    def _fetch_api(
        self,
        query: str,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> list[Document]:
        """Query the DeJure full REST API."""
        params: dict[str, Any] = {
            "q": query,
            "size": min(limit, 50),
            "page": 0,
        }
        if from_date:
            params["from"] = from_date.isoformat()
        if to_date:
            params["to"] = to_date.isoformat()

        documents: list[Document] = []
        for item in self._paginate(
            url=f"{_DEJURE_API_BASE}/search",
            params=params,
            page_key="page",
            size_key="size",
            page_size=min(limit, 50),
            max_pages=(limit // 50) + 2,
            results_key="hits",
        ):
            if len(documents) >= limit:
                break
            doc = self._api_item_to_document(item)
            if doc:
                documents.append(doc)

        return documents

    def _api_item_to_document(self, item: dict[str, Any]) -> Document | None:
        """Convert a DeJure API result to a Document."""
        testo = item.get("testo") or item.get("massima") or item.get("abstract") or ""
        if not testo:
            return None

        numero = item.get("numeroSentenza") or item.get("numero") or ""
        anno_str = item.get("anno") or ""
        massima = item.get("massima") or None
        sezione = item.get("sezione") or None
        tipo_str = item.get("tipo") or "sentenza"
        source_url = item.get("url") or f"{_DEJURE_API_BASE}/doc/{item.get('id', '')}"

        try:
            anno = int(anno_str) if anno_str else None
        except ValueError:
            anno = None

        tipo = TipoDocumento.SENTENZA_CASSAZIONE
        if "DOTTRINA" in tipo_str.upper():
            tipo = TipoDocumento.DOTTRINA

        meta = ItalianLegalMetadata(
            fonte=self.source_name,
            tipo_documento=tipo,
            anno=anno,
            numero_sentenza=f"Cass. n. {numero}/{anno}" if numero else None,
            sezione=sezione,
            massima=massima[:1000] if massima else None,
        )
        return self._make_document(
            content=testo,
            italian_meta=meta,
            source_uri=source_url,
            filename=f"dejure_{numero}_{anno}.txt",
        )

    # ── Public fallback (no api_key required) ─────────────────────────────────

    def _fetch_public(
        self,
        query: str,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> list[Document]:
        """Scrape publicly accessible massime from dejure.it."""
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
        except ImportError:
            log.warning(
                "beautifulsoup4 not installed; cannot scrape DeJure public massime"
            )
            return []

        search_url = f"{_DEJURE_PUBLIC_BASE}/ricerca/{quote(query or 'responsabilita')}"
        documents: list[Document] = []

        try:
            resp = self._get(search_url)
            soup = BeautifulSoup(resp.text, "html.parser")

            # dejure.it renders massime inside .massima or .risultato elements.
            for block in soup.select(".massima, .risultato-massima, .doc-massima")[:limit]:
                text = block.get_text(separator="\n", strip=True)
                if not text or len(text) < 30:
                    continue

                # Try to extract citation from nearby heading.
                heading = block.find_previous(["h2", "h3", "h4"])
                numero_sentenza = heading.get_text(strip=True) if heading else None

                meta = ItalianLegalMetadata(
                    fonte=self.source_name,
                    tipo_documento=TipoDocumento.SENTENZA_CASSAZIONE,
                    massima=text[:1000],
                    numero_sentenza=numero_sentenza,
                )
                link_tag = block.find("a", href=True)
                doc_url = urljoin(_DEJURE_PUBLIC_BASE, link_tag["href"]) if link_tag else search_url
                documents.append(
                    self._make_document(
                        content=text,
                        italian_meta=meta,
                        source_uri=doc_url,
                        filename=f"dejure_public_{hash(text) % 99999}.txt",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "DeJure public scrape failed",
                extra={"error": str(exc), "url": search_url},
            )

        return documents
