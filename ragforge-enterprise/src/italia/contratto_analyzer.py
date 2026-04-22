"""Contratto Analyzer — Italian consumer/civil contract analysis.

Analyses contract text against two statutory frameworks:

1. **Codice del Consumo** (D.Lgs. 206/2005, Art. 33–38) — *Clausole vessatorie*
   (unfair/abusive terms in B2C contracts) and *Clausole nulle di protezione*
   (per se void clauses).

2. **Codice Civile** (Art. 1418 c.c.) — *Nullità contrattuale* (clauses void
   under general civil law: contra legem, lacking essential elements, etc.).

For each identified clause the analyzer assigns a **risk score** (🔴 HIGH /
🟠 MEDIUM-HIGH / 🟡 MEDIUM / 🟢 LOW) and a concrete **suggested correction**
in Italian.

Architecture
-----------
The analysis is performed entirely via a single structured LLM call. No
additional retrieval is performed inside this module — callers may optionally
provide retrieved context chunks from the Qdrant knowledge base to ground the
LLM's output in the indexed Codice del Consumo provisions.

The LLM is prompted to return a structured JSON response matching
:class:`ContrattoAnalysisResult`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

from src.lexreview.agent.llm_client import LLMClient
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Risk-level enum ───────────────────────────────────────────────────────────


class RiskLevel(str, Enum):
    """Risk level for a contract clause.

    The emoji prefix is intentional: it appears in API responses and helps
    legal professionals scan risk reports at a glance.
    """

    ALTO = "🔴"
    MEDIO_ALTO = "🟠"
    MEDIO = "🟡"
    BASSO = "🟢"


# ── Clause result dataclass ───────────────────────────────────────────────────


@dataclass
class ClausolaAnalisi:
    """Analysis result for a single contract clause.

    Attributes:
        testo_clausola:      Verbatim or summarised clause text.
        tipo:                ``"vessatoria"`` | ``"nulla_cc"`` | ``"irregolare"``.
        riferimento_normativo: Applicable statutory reference,
                               e.g. ``"Art. 33 co. 2 lett. f) D.Lgs. 206/2005"``.
        motivazione:         Explanation of why the clause is problematic.
        risk_score:          :class:`RiskLevel` emoji indicator.
        correzione_suggerita: Suggested corrective wording in Italian.
    """

    testo_clausola: str
    tipo: str  # "vessatoria" | "nulla_cc" | "irregolare"
    riferimento_normativo: str
    motivazione: str
    risk_score: RiskLevel
    correzione_suggerita: str


@dataclass
class ContrattoAnalysisResult:
    """Full analysis result for a contract document.

    Attributes:
        clausole_vessatorie:  Clauses potentially abusive under Art. 33-38
                              Codice del Consumo.
        clausole_nulle:       Clauses void under Art. 1418 c.c.
        risk_score_globale:   Overall contract risk level.
        sommario:             Free-text executive summary in Italian.
    """

    clausole_vessatorie: list[ClausolaAnalisi] = field(default_factory=list)
    clausole_nulle: list[ClausolaAnalisi] = field(default_factory=list)
    risk_score_globale: RiskLevel = RiskLevel.BASSO
    sommario: str = ""


# ── Prompt ────────────────────────────────────────────────────────────────────

_CONTRATTO_SYSTEM = """\
Sei un avvocato esperto di diritto dei contratti italiano con specializzazione \
in tutela del consumatore. Il tuo compito è analizzare il contratto fornito e \
identificare:

(A) CLAUSOLE VESSATORIE ai sensi degli artt. 33-38 del Codice del Consumo \
    (D.Lgs. 206/2005), in particolare le clausole di cui all'art. 33 co. 2 \
    (lista grigia) e le clausole nulle ai sensi dell'art. 36 (lista nera).

(B) CLAUSOLE NULLE ai sensi dell'art. 1418 c.c. (nullità per contrarietà a \
    norme imperative, mancanza di causa, oggetto o forma prescritti).

Per ogni clausola problematica fornisci: testo, tipo, riferimento normativo, \
motivazione, risk_score e correzione suggerita.

SCALE RISK SCORE:
- 🔴 ALTO: clausola nulla di diritto o gravemente vessatoria (art. 36 cod. cons.)
- 🟠 MEDIO-ALTO: clausola presumibilmente vessatoria (art. 33 co. 2 cod. cons.)
- 🟡 MEDIO: clausola irregolare o migliorabile per chiarezza
- 🟢 BASSO: clausola conforme ma con suggerimenti stilistici

Rispondi SEMPRE e SOLO con un oggetto JSON valido, senza testo aggiuntivo:

{
  "clausole_vessatorie": [
    {
      "testo_clausola": "<testo>",
      "tipo": "vessatoria",
      "riferimento_normativo": "<art. e comma>",
      "motivazione": "<spiegazione>",
      "risk_score": "🔴|🟠|🟡|🟢",
      "correzione_suggerita": "<testo corretto>"
    }
  ],
  "clausole_nulle": [
    {
      "testo_clausola": "<testo>",
      "tipo": "nulla_cc",
      "riferimento_normativo": "<art. e comma>",
      "motivazione": "<spiegazione>",
      "risk_score": "🔴|🟠|🟡|🟢",
      "correzione_suggerita": "<testo corretto>"
    }
  ],
  "risk_score_globale": "🔴|🟠|🟡|🟢",
  "sommario": "<sommario esecutivo in italiano>"
}
"""

_RISK_EMOJI_MAP: dict[str, RiskLevel] = {
    "🔴": RiskLevel.ALTO,
    "🟠": RiskLevel.MEDIO_ALTO,
    "🟡": RiskLevel.MEDIO,
    "🟢": RiskLevel.BASSO,
    # Textual fallbacks for models that skip emoji
    "alto": RiskLevel.ALTO,
    "medio-alto": RiskLevel.MEDIO_ALTO,
    "medio": RiskLevel.MEDIO,
    "basso": RiskLevel.BASSO,
}


def _parse_risk(raw: str) -> RiskLevel:
    """Map a raw risk string to :class:`RiskLevel`.

    Args:
        raw: Raw risk string from LLM output.

    Returns:
        Corresponding :class:`RiskLevel`, defaulting to
        :attr:`RiskLevel.MEDIO` when unrecognised.
    """
    stripped = (raw or "").strip()
    for key, val in _RISK_EMOJI_MAP.items():
        if key in stripped:
            return val
    return RiskLevel.MEDIO


def _parse_clausole(items: list[dict]) -> list[ClausolaAnalisi]:
    """Parse a list of clause dicts from the LLM JSON into typed objects.

    Args:
        items: Raw list of dicts from the LLM JSON response.

    Returns:
        List of :class:`ClausolaAnalisi` objects.
    """
    results: list[ClausolaAnalisi] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            ClausolaAnalisi(
                testo_clausola=str(item.get("testo_clausola", "")),
                tipo=str(item.get("tipo", "irregolare")),
                riferimento_normativo=str(item.get("riferimento_normativo", "")),
                motivazione=str(item.get("motivazione", "")),
                risk_score=_parse_risk(str(item.get("risk_score", "🟡"))),
                correzione_suggerita=str(item.get("correzione_suggerita", "")),
            )
        )
    return results


# ── Service ───────────────────────────────────────────────────────────────────

MAX_CONTRACT_CHARS = 12_000  # ~3 000 tokens; most standard contracts fit


def analyze_contract(contract_text: str, llm: LLMClient) -> ContrattoAnalysisResult:
    """Analyse *contract_text* for abusive and void clauses.

    Args:
        contract_text: Plain-text content of the contract (extracted from PDF
                       by the calling endpoint).
        llm:           :class:`~src.lexreview.agent.llm_client.LLMClient`.

    Returns:
        :class:`ContrattoAnalysisResult` with classified clauses, risk scores,
        corrections, and an executive summary.
    """
    log.info("contratto.analyze_contract", extra={"text_len": len(contract_text)})

    if len(contract_text) > MAX_CONTRACT_CHARS:
        log.debug(
            "contratto: text truncated",
            extra={"original_len": len(contract_text), "max_chars": MAX_CONTRACT_CHARS},
        )
        contract_text = contract_text[:MAX_CONTRACT_CHARS]

    messages = [
        {"role": "system", "content": _CONTRATTO_SYSTEM},
        {
            "role": "user",
            "content": (
                "Analizza il seguente contratto secondo le istruzioni:\n\n"
                f"{contract_text}"
            ),
        },
    ]

    raw = llm.complete(messages)

    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("` ")
        data = json.loads(clean)

        return ContrattoAnalysisResult(
            clausole_vessatorie=_parse_clausole(data.get("clausole_vessatorie") or []),
            clausole_nulle=_parse_clausole(data.get("clausole_nulle") or []),
            risk_score_globale=_parse_risk(str(data.get("risk_score_globale", "🟢"))),
            sommario=str(data.get("sommario", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning(
            "contratto: failed to parse LLM JSON",
            extra={"error": str(exc), "raw_preview": raw[:200]},
        )
        return ContrattoAnalysisResult(
            sommario=f"Analisi non disponibile: errore parsing LLM ({exc})",
        )
