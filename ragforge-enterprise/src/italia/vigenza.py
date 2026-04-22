"""Vigenza check service for Italian legal norms.

Determines whether a cited Italian norm (e.g. "Art. 18 L. 300/1970") was
in force (vigente) on a given reference date by querying the Normattiva
knowledge base stored in Qdrant and/or performing a lightweight LLM-based
determination when the structured metadata is unavailable.

Architecture
------------
1. Parse the *norma* string into structured components (tipo, numero, anno).
2. Search the ``ragforge_italia`` Qdrant collection for the matching act using
   the existing :class:`~src.retrieval.hybrid_retriever.HybridRetriever`.
3. If a matching chunk is found with ``it_data_vigenza`` / ``it_data_abrogazione``
   payload fields, derive the answer deterministically from the metadata.
4. Otherwise, fall back to an LLM call that returns a structured JSON response.
5. Return a :class:`VigenzaResult` dataclass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from src.lexreview.agent.llm_client import LLMClient
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Norm-string parser ────────────────────────────────────────────────────────

_NORMA_RE = re.compile(
    r"""
    # Optional article specification: "Art. 18", "artt. 3-5", "comma 2"
    (?:(?:artt?\.|articol[oi])\s*[\d\-]+[\s,]*)?
    # Abbreviated act type
    (?P<tipo>
        D\.Lgs\.|D\.L\.|D\.P\.R\.|D\.P\.C\.M\.|
        L\.|Legge|Regio\s+Decreto|R\.D\.|
        Codice\s+Civile|c\.c\.|Codice\s+Penale|c\.p\.|
        Costituzione|Cost\.
    )?
    \s*
    # Optional number e.g. "300" or "300bis"
    (?P<numero>\d+\s*(?:bis|ter|quater|quinquies)?)?\s*
    /?
    # Year e.g. "1970"
    (?P<anno>\d{4})?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_norma(norma: str) -> dict[str, str | None]:
    """Extract tipo, numero, anno from a norm citation string.

    Args:
        norma: Free-text norm citation, e.g. ``"Art. 18 L. 300/1970"``.

    Returns:
        Dict with keys ``tipo``, ``numero``, ``anno`` (all may be ``None``).
    """
    m = _NORMA_RE.search(norma)
    if not m:
        return {"tipo": None, "numero": None, "anno": None}
    return {
        "tipo": (m.group("tipo") or "").strip() or None,
        "numero": (m.group("numero") or "").strip() or None,
        "anno": (m.group("anno") or "").strip() or None,
    }


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class VigenzaResult:
    """Result of a vigenza (norm validity) check.

    Attributes:
        vigente:              ``True`` when the norm was in force on
                              *data_riferimento*.
        data_entrata_vigore:  Date the norm entered into force (``None``
                              when unknown).
        data_abrogazione:     Date the norm was repealed (``None`` when still
                              in force or unknown).
        modificata_da:        List of later acts that modified this norm.
        testo_vigente:        The in-force text of the norm at the reference
                              date (``None`` when unavailable).
        fonte:                Source of the determination: ``"metadata"`` or
                              ``"llm_inference"``.
    """

    vigente: bool
    data_entrata_vigore: date | None
    data_abrogazione: date | None
    modificata_da: list[str]
    testo_vigente: str | None
    fonte: str = "llm_inference"


# ── Service ───────────────────────────────────────────────────────────────────

_VIGENZA_SYSTEM = """\
Sei un esperto di diritto italiano e stai operando come assistente per la \
verifica della vigenza delle norme. Rispondi SEMPRE e SOLO con un oggetto JSON \
valido, senza testo aggiuntivo, con i seguenti campi:

{
  "vigente": <true | false>,
  "data_entrata_vigore": "<YYYY-MM-DD> | null",
  "data_abrogazione": "<YYYY-MM-DD> | null",
  "modificata_da": ["<atto1>", "<atto2>"],
  "testo_vigente": "<testo breve della norma vigente alla data indicata, null se sconosciuto>"
}

REGOLE FONDAMENTALI:
- Verifica se la norma era in vigore alla data di riferimento indicata.
- Se la norma è stata abrogata PRIMA della data di riferimento → vigente: false.
- Se la norma è stata abrogata DOPO la data di riferimento → vigente: true.
- Indica tutte le fonti di modifica o abrogazione note (D.Lgs., L., D.L. ecc.).
- Il campo testo_vigente deve contenere il testo della norma vigente alla data \
  di riferimento, oppure null se non lo conosci con certezza.
"""


def check_vigenza(
    norma: str,
    data_riferimento: date,
    llm: LLMClient,
    context_chunks: list[dict[str, Any]] | None = None,
) -> VigenzaResult:
    """Check whether *norma* was in force on *data_riferimento*.

    The function first attempts a deterministic check using structured
    ``it_data_vigenza`` / ``it_data_abrogazione`` payload fields found in
    *context_chunks*.  When these fields are not available it falls back to an
    LLM inference call.

    Args:
        norma:            Free-text norm citation, e.g. ``"Art. 18 L. 300/1970"``.
        data_riferimento: Reference date for the vigency check.
        llm:              :class:`~src.lexreview.agent.llm_client.LLMClient`
                          used for the fallback LLM inference.
        context_chunks:   Optional list of Qdrant payload dicts retrieved for
                          **norma**; used to extract structured metadata before
                          falling back to LLM.

    Returns:
        :class:`VigenzaResult` populated from metadata or LLM output.
    """
    parsed = _parse_norma(norma)
    log.info(
        "vigenza.check_vigenza",
        extra={"norma": norma, "parsed": parsed, "data_riferimento": str(data_riferimento)},
    )

    # ── 1. Try deterministic metadata path ───────────────────────────────────
    if context_chunks:
        for chunk in context_chunks:
            meta = chunk.get("metadata", {})
            raw_vigenza = meta.get("it_data_vigenza")
            raw_abrogazione = meta.get("it_data_abrogazione")
            if raw_vigenza:
                try:
                    dv = date.fromisoformat(raw_vigenza)
                    da: date | None = (
                        date.fromisoformat(raw_abrogazione) if raw_abrogazione else None
                    )
                    if dv <= data_riferimento:
                        vigente = da is None or da > data_riferimento
                        return VigenzaResult(
                            vigente=vigente,
                            data_entrata_vigore=dv,
                            data_abrogazione=da,
                            modificata_da=[],
                            testo_vigente=chunk.get("content"),
                            fonte="metadata",
                        )
                except ValueError:
                    pass

    # ── 2. LLM inference fallback ─────────────────────────────────────────────
    context_text = ""
    if context_chunks:
        snippets = [c.get("content", "")[:400] for c in context_chunks[:3]]
        context_text = "\n\n---\n\n".join(snippets)

    user_msg = (
        f"Norma da verificare: {norma}\n"
        f"Data di riferimento: {data_riferimento.isoformat()}\n"
    )
    if context_text:
        user_msg += f"\nContesto recuperato dalla knowledge base:\n{context_text}\n"

    messages = [
        {"role": "system", "content": _VIGENZA_SYSTEM},
        {"role": "user", "content": user_msg},
    ]

    raw = llm.complete(messages)

    # Parse JSON from LLM response
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("` ")
        data = json.loads(clean)

        def _try_date(v: str | None) -> date | None:
            if not v or v == "null":
                return None
            try:
                return date.fromisoformat(v)
            except (ValueError, TypeError):
                return None

        return VigenzaResult(
            vigente=bool(data.get("vigente", False)),
            data_entrata_vigore=_try_date(data.get("data_entrata_vigore")),
            data_abrogazione=_try_date(data.get("data_abrogazione")),
            modificata_da=data.get("modificata_da") or [],
            testo_vigente=data.get("testo_vigente"),
            fonte="llm_inference",
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning(
            "vigenza: failed to parse LLM JSON response",
            extra={"error": str(exc), "raw": raw[:200]},
        )
        return VigenzaResult(
            vigente=False,
            data_entrata_vigore=None,
            data_abrogazione=None,
            modificata_da=[],
            testo_vigente=None,
            fonte="llm_inference",
        )
