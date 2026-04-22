"""Gazzetta Ufficiale connector — gazzettaufficiale.it.

Fetches Italian legislative acts (leggi, decreti, regolamenti) published
in the Official Gazette via the gazzettaufficiale.it search endpoint.

Authentication: None required for public acts.
Format: HTML → cleaned plain text via BeautifulSoup + html2text.
Pagination: Offset-based (``start`` param).

Notes
-----
- The GU website is not an API — this connector does screen-scraping.
  It uses polite headers and a conservative rate limit (0.5 req/s).
- Each GU act is published as a standalone HTML page with a stable URL
  pattern:  ``/eli/id/{YYYY}/{MM}/{DD}/{n}/sg``
- The search endpoint returns JSON with ``total``, ``lista`` array.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from src.ingestion.loader import Document
from src.italia.connectors.base import BaseConnector, ConnectorParseError
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)

_SEARCH_URL = "https://www.gazzettaufficiale.it/ricerca/json/ricerca"
_ACT_BASE_URL = "https://www.gazzettaufficiale.it"

_TIPO_MAP: dict[str, TipoDocumento] = {
    "LEGGE": TipoDocumento.LEGGE,
    "DECRETO LEGISLATIVO": TipoDocumento.DECRETO_LEGISLATIVO,
    "DECRETO-LEGGE": TipoDocumento.DECRETO_LEGGE,
    "DECRETO LEGGE": TipoDocumento.DECRETO_LEGGE,
    "DECRETO DEL PRESIDENTE DELLA REPUBBLICA": TipoDocumento.REGOLAMENTO,
    "REGOLAMENTO": TipoDocumento.REGOLAMENTO,
}


def _guess_tipo(title: str) -> TipoDocumento:
    upper = title.upper()
    for key, tipo in _TIPO_MAP.items():
        if key in upper:
            return tipo
    return TipoDocumento.LEGGE


def _extract_numero_anno(title: str) -> tuple[str | None, int | None]:
    """Extract act number and year from title string."""
    m = re.search(r"n[.°]?\s*(\d+)\s*(?:del\s*)?(?:,\s*)?(\d{4})", title, re.IGNORECASE)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


class GazzettaUfficialeConnector(BaseConnector):
    """Fetches Italian acts published in the Gazzetta Ufficiale.

    Args:
        rate_limit_rps: Max req/s (default 0.5 — conservative scraping).
        timeout:        HTTP timeout in seconds.

    Example::

        conn = GazzettaUfficialeConnector()
        docs = conn.fetch(
            from_date=date(2024, 1, 1),
            to_date=date(2024, 3, 31),
            limit=20,
        )
    """

    source_name = "gazzetta_ufficiale"

    def __init__(self, rate_limit_rps: float = 0.5, timeout: int = 30) -> None:
        super().__init__(rate_limit_rps=rate_limit_rps, timeout=timeout)

    def fetch(  # type: ignore[override]
        self,
        query: str = "",
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 50,
        **_: Any,
    ) -> list[Document]:
        """Search and fetch acts from the Gazzetta Ufficiale.

        Args:
            query:     Free-text search query (Italian keywords).
            from_date: Start date filter (inclusive).
            to_date:   End date filter (inclusive).
            limit:     Maximum number of acts to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        log.info(
            "Fetching from Gazzetta Ufficiale",
            extra={
                "query": query,
                "from_date": str(from_date),
                "to_date": str(to_date),
                "limit": limit,
            },
        )

        acts = list(self._iter_search_results(query, from_date, to_date, limit))
        documents: list[Document] = []

        for act in acts:
            try:
                doc = self._fetch_act(act)
                if doc:
                    documents.append(doc)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to fetch GU act",
                    extra={"act": act.get("id"), "error": str(exc)},
                )
                continue

        log.info(
            "Gazzetta Ufficiale fetch complete",
            extra={"fetched": len(documents)},
        )
        return documents

    # ── Private helpers ───────────────────────────────────────────────────────

    def _iter_search_results(
        self,
        query: str,
        from_date: date | None,
        to_date: date | None,
        limit: int,
    ) -> Any:
        """Yield raw act metadata dicts from the search endpoint."""
        page_size = min(limit, 20)
        params: dict[str, Any] = {
            "testo": query or "decreto legge legge regolamento",
            "redaz": "",
            "tipoSerie": "GO",  # Gazzetta Ordinaria
        }
        if from_date:
            params["dataPubblicazioneFrom"] = from_date.strftime("%d/%m/%Y")
        if to_date:
            params["dataPubblicazioneTo"] = to_date.strftime("%d/%m/%Y")

        fetched = 0
        for item in self._paginate(
            url=_SEARCH_URL,
            params=params,
            page_key="start",
            size_key="length",
            page_size=page_size,
            max_pages=(limit // page_size) + 1,
            results_key="lista",
        ):
            if fetched >= limit:
                break
            yield item
            fetched += 1

    def _fetch_act(self, act_meta: dict[str, Any]) -> Document | None:
        """Fetch the full text of a single GU act and return a Document."""
        url_path = act_meta.get("urlDettaglio") or act_meta.get("linkDettaglio", "")
        if not url_path:
            return None

        full_url = _ACT_BASE_URL + url_path if url_path.startswith("/") else url_path
        resp = self._get(full_url)

        text = self._html_to_text(resp.text)
        if not text.strip():
            return None

        title: str = act_meta.get("titoloAtt", "") or act_meta.get("titolo", "")
        pub_date_str: str = act_meta.get("dataPubblicazione", "")
        pub_date: date | None = None
        if pub_date_str:
            for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
                try:
                    pub_date = datetime.strptime(pub_date_str, fmt).date()
                    break
                except ValueError:
                    continue

        numero, anno = _extract_numero_anno(title)
        tipo = _guess_tipo(title)

        meta = ItalianLegalMetadata(
            fonte=self.source_name,
            tipo_documento=tipo,
            numero_atto=title[:200] if title else None,
            anno=anno or (pub_date.year if pub_date else None),
            data_vigenza=pub_date,
        )
        return self._make_document(text, meta, full_url, filename=f"gu_{url_path.split('/')[-1]}.html")

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Convert HTML to clean plain text."""
        try:
            import html2text  # noqa: PLC0415

            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.ignore_tables = False
            h.body_width = 0
            return h.handle(html)
        except ImportError:
            # Fallback: strip tags with regex
            text = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL)
            text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"&[a-z]+;", " ", text)
            return re.sub(r"\s+", " ", text).strip()
