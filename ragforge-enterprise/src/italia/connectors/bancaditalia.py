"""Banca d'Italia / IVASS connector — financial regulation (bancaditalia.it).

Fetches regulatory documents, circolari, provvedimenti, and supervisory
guidance from:
  - Banca d'Italia:  https://www.bancaditalia.it
  - IVASS (insurance regulator, subsidiary of BdI): https://www.ivass.it

Authentication: None required for public documents.
Format:         JSON search API + HTML detail pages.

Categories indexed:
  - Circolari (supervisory instructions)
  - Provvedimenti (regulatory decisions)
  - Comunicazioni (supervisory communications)
  - Regolamenti IVASS
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

_BDI_BASE = "https://www.bancaditalia.it"
_BDI_SEARCH = f"{_BDI_BASE}/footer/ricerca/index.html"
_BDI_API = f"{_BDI_BASE}/pubblicazioni/bollettino-vigilanza/ricerca/json"

_IVASS_BASE = "https://www.ivass.it"
_IVASS_SEARCH = f"{_IVASS_BASE}/normativa/regolamenti/ricerca"

_TIPO_MAP: dict[str, TipoDocumento] = {
    "circolare": TipoDocumento.CIRCOLARE,
    "regolamento": TipoDocumento.REGOLAMENTO,
    "provvedimento": TipoDocumento.PROVVEDIMENTO,
    "decreto": TipoDocumento.DECRETO_LEGISLATIVO,
}


def _guess_tipo(title: str) -> TipoDocumento:
    lower = title.lower()
    for key, tipo in _TIPO_MAP.items():
        if key in lower:
            return tipo
    return TipoDocumento.CIRCOLARE


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


class BancaItaliaConnector(BaseConnector):
    """Fetches Banca d'Italia and IVASS regulatory documents.

    Args:
        include_ivass:  Also fetch IVASS regulations (default True).
        rate_limit_rps: Max req/s (default 0.5).
        timeout:        HTTP timeout in seconds.

    Example::

        conn = BancaItaliaConnector()
        docs = conn.fetch(query="antiriciclaggio", limit=20)
    """

    source_name = "bancaditalia"

    def __init__(
        self,
        include_ivass: bool = True,
        rate_limit_rps: float = 0.5,
        timeout: int = 45,
    ) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)
        self._include_ivass = include_ivass

    def fetch(  # type: ignore[override]
        self,
        query: str = "",
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 30,
        **_: Any,
    ) -> list[Document]:
        """Fetch Banca d'Italia and IVASS regulatory documents.

        Args:
            query:     Full-text query (Italian — e.g. ``"antiriciclaggio"``).
            from_date: Start date filter.
            to_date:   End date filter.
            limit:     Max documents to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        log.info(
            "Fetching from Banca d'Italia",
            extra={"query": query, "limit": limit, "ivass": self._include_ivass},
        )

        bdi_docs = self._fetch_bdi(query, from_date, to_date, limit)
        ivass_docs: list[Document] = []
        if self._include_ivass:
            remaining = limit - len(bdi_docs)
            if remaining > 0:
                ivass_docs = self._fetch_ivass(query, from_date, to_date, remaining)

        documents = bdi_docs + ivass_docs
        log.info("BdI/IVASS fetch complete", extra={"count": len(documents)})
        return documents[:limit]

    # ── BdI helpers ───────────────────────────────────────────────────────────

    def _fetch_bdi(
        self,
        query: str,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> list[Document]:
        params: dict[str, Any] = {
            "q": query or "circolare vigilanza",
            "start": 0,
            "rows": min(limit, 20),
        }
        if from_date:
            params["dataDal"] = from_date.strftime("%d/%m/%Y")
        if to_date:
            params["dataAl"] = to_date.strftime("%d/%m/%Y")

        documents: list[Document] = []
        try:
            resp = self._get(_BDI_API, params=params)
            items = resp.json().get("items", []) or resp.json().get("results", [])
            for item in items[:limit]:
                try:
                    doc = self._item_to_document(item, source="bancaditalia")
                    if doc:
                        documents.append(doc)
                except Exception as exc:  # noqa: BLE001
                    log.warning("BdI item failed", extra={"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.warning("BdI API failed", extra={"error": str(exc)})

        return documents

    # ── IVASS helpers ─────────────────────────────────────────────────────────

    def _fetch_ivass(
        self,
        query: str,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> list[Document]:
        try:
            from bs4 import BeautifulSoup  # noqa: PLC0415
        except ImportError:
            log.warning("beautifulsoup4 not installed; skipping IVASS scraping")
            return []

        params: dict[str, Any] = {"q": query or "regolamento"}
        documents: list[Document] = []
        try:
            resp = self._get(_IVASS_SEARCH, params=params)
            soup = BeautifulSoup(resp.text, "html.parser")
            for link in soup.select("a[href*='regolamento'], a[href*='circolare']")[:limit]:
                href = link.get("href", "")
                if not href:
                    continue
                full_url = urljoin(_IVASS_BASE, href)
                try:
                    doc_resp = self._get(full_url)
                    text = self._extract_text(doc_resp.text)
                    if not text.strip():
                        continue
                    title = link.get_text(strip=True)
                    tipo = _guess_tipo(title)
                    meta = ItalianLegalMetadata(
                        fonte="ivass",
                        tipo_documento=tipo,
                        numero_atto=title[:200],
                        materia=["assicurazioni", "IVASS"],
                        parole_chiave=["IVASS", "assicurazioni"],
                    )
                    documents.append(
                        self._make_document(text, meta, full_url, filename=f"ivass_{href.split('/')[-1]}")
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("IVASS doc failed", extra={"url": full_url, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            log.warning("IVASS scrape failed", extra={"error": str(exc)})

        return documents

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _item_to_document(
        self, item: dict[str, Any], source: str
    ) -> Document | None:
        url = item.get("url") or item.get("link") or ""
        title = item.get("titolo") or item.get("title") or ""
        raw_date = item.get("data") or item.get("dataPubblicazione") or ""
        body = item.get("testo") or item.get("abstract") or ""

        if not body and url:
            if url and not url.startswith("http"):
                url = urljoin(_BDI_BASE, url)
            try:
                resp = self._get(url)
                body = self._extract_text(resp.text)
            except Exception:  # noqa: BLE001
                body = title

        if not body:
            return None

        dep_date = _parse_date(raw_date)
        anno = dep_date.year if dep_date else None
        tipo = _guess_tipo(title)
        numero_atto = title[:200] if title else None

        meta = ItalianLegalMetadata(
            fonte=source,
            tipo_documento=tipo,
            numero_atto=numero_atto,
            anno=anno,
            data_vigenza=dep_date,
            materia=["normativa finanziaria", "banca", "vigilanza"],
            parole_chiave=["Banca d'Italia", "IVASS", "vigilanza"],
        )
        return self._make_document(
            content=body,
            italian_meta=meta,
            source_uri=url or f"{source}://{tipo.value}/{anno}",
            filename=f"{source}_{anno}_{hash(title) % 99999}.txt",
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
