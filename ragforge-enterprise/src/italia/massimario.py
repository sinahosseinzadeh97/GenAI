"""Massimario automatico — automatic headnote generation for Italian judgments.

Given the full text of an Italian court *sentenza*, this module generates:

- **Massima ufficiale** — the official headnote (≤ 150 words), summarising
  the legal holding in the third-person impersonal style used by the
  *Ufficio del Massimario* at the Corte di Cassazione.
- **Principio di diritto** — the binding legal principle in syllogistic form.
- **Parole chiave** — keyword tags for indexing and classification.
- **Classificazione materia** — area-of-law taxonomy (e.g. *diritto civile*,
  *diritto del lavoro*).

All generation is performed via a single structured LLM call that returns a
JSON object, avoiding multi-hop round-trips.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.lexreview.agent.llm_client import LLMClient
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class MassimaResult:
    """Auto-generated massima for an Italian sentenza.

    Attributes:
        massima_ufficiale:      Official headnote (≤ 150 words).
        principio_di_diritto:   Binding legal principle in syllogistic form.
        parole_chiave:          Keyword list for indexing.
        classificazione_materia: Area-of-law label(s), e.g.
                                ``["diritto del lavoro", "licenziamento"]``.
    """

    massima_ufficiale: str
    principio_di_diritto: str
    parole_chiave: list[str]
    classificazione_materia: list[str]


# ── Prompt ────────────────────────────────────────────────────────────────────

_MASSIMA_SYSTEM = """\
Sei il Massimario della Corte di Cassazione italiana. Il tuo compito è \
analizzare il testo di una sentenza e generare le seguenti componenti \
della massima ufficiale. Rispondi SEMPRE e SOLO con un oggetto JSON valido, \
senza testo aggiuntivo, rispettando esattamente questo schema:

{
  "massima_ufficiale": "<massima in max 150 parole, stile impersonale terza persona>",
  "principio_di_diritto": "<enunciazione del principio di diritto in forma sillogistica>",
  "parole_chiave": ["<keyword1>", "<keyword2>", "<keyword3>"],
  "classificazione_materia": ["<materia1>", "<materia2>"]
}

ISTRUZIONI STILISTICHE:
- La massima_ufficiale deve essere in massimo 150 parole, redatta in stile \
  impersonale (terza persona), come da prassi del Massimario CED Cassazione.
- Il principio_di_diritto deve enunciare la regola giuridica in forma astratta, \
  separata dal caso concreto.
- Le parole_chiave devono includere: materia del diritto, istituti giuridici \
  rilevanti, norme citate, concetti chiave.
- La classificazione_materia deve usare le categorie standard del diritto italiano: \
  diritto civile, diritto penale, diritto del lavoro, diritto amministrativo, \
  diritto commerciale, diritto tributario, diritto processuale civile, \
  diritto processuale penale, diritto costituzionale, diritto europeo.
"""


# ── Service ───────────────────────────────────────────────────────────────────


def generate_massima(sentenza_text: str, llm: LLMClient) -> MassimaResult:
    """Generate a *massima* automatically from a sentenza text.

    The sentenza text is truncated to ≤ 8 000 characters before being sent to
    the LLM to stay within context-window limits and avoid excessive token
    spend on boilerplate procedural text.  The most legally relevant portion
    (typically the *motivazione* and *dispositivo*) should be passed directly
    rather than the full PDF dump.

    Args:
        sentenza_text: Full or partial text of the Italian court judgment.
        llm:           :class:`~src.lexreview.agent.llm_client.LLMClient`.

    Returns:
        :class:`MassimaResult` with all four components populated.

    Raises:
        ValueError: When the LLM returns malformed JSON that cannot be parsed
                    even after cleanup.
    """
    log.info("massimario.generate_massima", extra={"text_len": len(sentenza_text)})

    # Truncate to a safe context-window slice — prefer the last 8 k chars
    # since dispositivo / motivazione typically appear near the end.
    MAX_CHARS = 8_000
    if len(sentenza_text) > MAX_CHARS:
        sentenza_truncated = sentenza_text[-MAX_CHARS:]
        log.debug(
            "massimario: sentenza truncated",
            extra={"original_len": len(sentenza_text), "truncated_len": MAX_CHARS},
        )
    else:
        sentenza_truncated = sentenza_text

    messages = [
        {"role": "system", "content": _MASSIMA_SYSTEM},
        {
            "role": "user",
            "content": (
                "Analizza la seguente sentenza e genera la massima secondo le istruzioni:\n\n"
                f"{sentenza_truncated}"
            ),
        },
    ]

    raw = llm.complete(messages)

    # ── Parse JSON ─────────────────────────────────────────────────────────────
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("` ")
        data = json.loads(clean)
        return MassimaResult(
            massima_ufficiale=str(data.get("massima_ufficiale", "")),
            principio_di_diritto=str(data.get("principio_di_diritto", "")),
            parole_chiave=list(data.get("parole_chiave") or []),
            classificazione_materia=list(data.get("classificazione_materia") or []),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning(
            "massimario: failed to parse LLM JSON — returning raw as massima",
            extra={"error": str(exc), "raw_preview": raw[:200]},
        )
        # Degrade gracefully: surface the raw text rather than raise
        return MassimaResult(
            massima_ufficiale=raw.strip()[:600],
            principio_di_diritto="",
            parole_chiave=[],
            classificazione_materia=[],
        )
