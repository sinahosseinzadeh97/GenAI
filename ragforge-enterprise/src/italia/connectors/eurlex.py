"""EUR-Lex connector — Italian-language EU law (eur-lex.europa.eu).

Fetches EU Directives and Regulations in Italian via the EUR-Lex REST/SPARQL
API. Uses the official EUR-Lex data.europa.eu SPARQL endpoint as the
primary discovery mechanism, then fetches full-text via the ELI REST API.

Authentication: None required (public open-data endpoint).
Primary API:    https://data.europa.eu/euodp/sparqlproxy (SPARQL 1.1)
Full-text URL:  https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=<CELEX>

Pagination: SPARQL OFFSET/LIMIT pattern.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from src.ingestion.loader import Document
from src.italia.connectors.base import BaseConnector
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento
from src.utils.logger import get_logger

log = get_logger(__name__)

_SPARQL_URL = "https://publications.europa.eu/webapi/rdf/sparql"
_EURLEX_TXT_URL = "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/"

# SPARQL query template — fetches Italian-language EU acts.
_SPARQL_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>

SELECT DISTINCT ?work ?celex ?title ?date ?type
WHERE {{
  ?work cdm:work_has_resource-type ?type .
  ?work cdm:work_date_document ?date .
  ?work cdm:resource_legal_id_celex ?celex .
  ?work cdm:work_is_about_concept_directory-code ?dc .
  ?expr cdm:expression_belongs_to_work ?work .
  ?expr cdm:expression_uses_language <http://publications.europa.eu/resource/authority/language/ITA> .
  ?expr cdm:expression_title ?title .
  FILTER(?date >= "{from_date}"^^xsd:date && ?date <= "{to_date}"^^xsd:date)
  FILTER(?type IN (
    <http://publications.europa.eu/resource/authority/resource-type/REG>,
    <http://publications.europa.eu/resource/authority/resource-type/DIR>
  ))
}}
ORDER BY DESC(?date)
LIMIT {limit}
OFFSET {offset}
"""


def _tipo_from_celex(celex: str) -> TipoDocumento:
    if celex.startswith("3") and "L" in celex[1:5]:
        return TipoDocumento.DIRETTIVA_EU
    return TipoDocumento.REGOLAMENTO


class EurLexConnector(BaseConnector):
    """Fetches EU Directives and Regulations in Italian from EUR-Lex.

    Args:
        rate_limit_rps: Max req/s (default 0.5 — EUR-Lex is rate-sensitive).
        timeout:        HTTP timeout in seconds.

    Example::

        conn = EurLexConnector()
        docs = conn.fetch(
            from_date=date(2020, 1, 1),
            to_date=date(2024, 12, 31),
            limit=20,
        )
    """

    source_name = "eurlex"

    def __init__(self, rate_limit_rps: float = 0.5, timeout: int = 60) -> None:
        super().__init__(
            rate_limit_rps=rate_limit_rps,
            timeout=timeout,
            headers={"Accept": "application/sparql-results+json"},
        )

    def fetch(  # type: ignore[override]
        self,
        from_date: date | None = None,
        to_date: date | None = None,
        limit: int = 30,
        **_: Any,
    ) -> list[Document]:
        """Fetch Italian-language EU acts from EUR-Lex.

        Args:
            from_date: Start date (inclusive). Defaults to 2020-01-01.
            to_date:   End date (inclusive). Defaults to today.
            limit:     Maximum acts to return.

        Returns:
            List of :class:`~src.ingestion.loader.Document` objects.
        """
        fd = from_date or date(2020, 1, 1)
        td = to_date or date.today()

        log.info(
            "Fetching from EUR-Lex",
            extra={"from_date": str(fd), "to_date": str(td), "limit": limit},
        )

        acts = self._sparql_search(fd, td, limit)
        documents: list[Document] = []
        for act in acts:
            try:
                doc = self._fetch_fulltext(act)
                if doc:
                    documents.append(doc)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Failed to fetch EUR-Lex act",
                    extra={"celex": act.get("celex"), "error": str(exc)},
                )

        log.info("EUR-Lex fetch complete", extra={"count": len(documents)})
        return documents

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sparql_search(
        self, from_date: date, to_date: date, limit: int
    ) -> list[dict[str, str]]:
        """Run the SPARQL query and return a list of act descriptor dicts."""
        query = _SPARQL_QUERY.format(
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
            limit=limit,
            offset=0,
        )
        try:
            resp = self._get(
                _SPARQL_URL,
                params={"query": query, "format": "application/sparql-results+json"},
            )
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            log.warning("SPARQL query failed", extra={"error": str(exc)})
            return []

        results = []
        for binding in data.get("results", {}).get("bindings", []):
            results.append({
                "uri": binding.get("work", {}).get("value", ""),
                "celex": binding.get("celex", {}).get("value", ""),
                "title": binding.get("title", {}).get("value", ""),
                "date": binding.get("date", {}).get("value", ""),
            })
        return results

    def _fetch_fulltext(self, act: dict[str, str]) -> Document | None:
        """Fetch the Italian HTML full text for a CELEX identifier."""
        celex = act.get("celex", "")
        if not celex:
            return None

        full_url = _EURLEX_TXT_URL
        resp = self._get(full_url, params={"uri": f"CELEX:{celex}"})

        text = self._strip_html(resp.text)
        if len(text.strip()) < 100:
            return None

        title = act.get("title", f"EU Act {celex}")
        raw_date = act.get("date", "")
        act_date: date | None = None
        if raw_date:
            try:
                act_date = date.fromisoformat(raw_date[:10])
            except ValueError:
                pass

        anno = act_date.year if act_date else None
        tipo = _tipo_from_celex(celex)

        meta = ItalianLegalMetadata(
            fonte=self.source_name,
            tipo_documento=tipo,
            urn_nir=f"urn:lex:eu:{celex.lower()}",
            numero_atto=title[:200],
            anno=anno,
            data_vigenza=act_date,
            parole_chiave=["EU", celex],
        )
        return self._make_document(
            content=text,
            italian_meta=meta,
            source_uri=f"{_EURLEX_TXT_URL}?uri=CELEX:{celex}",
            filename=f"eurlex_{celex}.html",
        )

    @staticmethod
    def _strip_html(html: str) -> str:
        try:
            import html2text  # noqa: PLC0415

            h = html2text.HTML2Text()
            h.ignore_links = True
            h.ignore_images = True
            h.body_width = 0
            return h.handle(html)
        except ImportError:
            text = re.sub(r"<[^>]+>", " ", html)
            return re.sub(r"\s+", " ", text).strip()
