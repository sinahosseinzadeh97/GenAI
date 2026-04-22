"""231 Compliance Module — D.Lgs. 231/2001 corporate criminal liability.

D.Lgs. 231/2001 introduced vicarious corporate liability in Italy for a
catalogue of *reati presupposto* (predicate offences).  This module provides
a risk-assessment service that, given a company's sector (*settore*) and
business activity description, returns:

- The applicable *reati presupposto* (predicate offences from the Annexes to
  D.Lgs. 231/2001) for the described sector/activities.
- *ODV recommendations* (Organismo di Vigilanza — supervisory body) aligned
  with Confindustria and Assonime best-practice guidelines.
- A **risk score** (0.0 – 1.0) reflecting the breadth and severity of
  applicable predicate offences.
- Statutory references (*riferimenti normativi*) for each applicable offence.

The assessment uses the LLM as a structured knowledge engine, grounded by a
comprehensive system prompt encoding the D.Lgs. 231/2001 predicate offence
catalogue through the latest legislative amendments.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from src.lexreview.agent.llm_client import LLMClient
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class Compliance231Result:
    """Result of a D.Lgs. 231/2001 risk assessment.

    Attributes:
        reati_presupposto:      List of applicable predicate offences
                                (with statutory reference), e.g.
                                ``["Corruzione art. 25 D.Lgs. 231/2001"]``.
        odv_raccomandazioni:    Practical recommendations for the Organismo di
                                Vigilanza (supervisory body).
        risk_score:             Numeric risk score in [0.0, 1.0].
        riferimenti_normativi:  Statutory citations for the applicable articles
                                of D.Lgs. 231/2001 and related legislation.
        sintesi:                Free-text Italian executive summary.
    """

    reati_presupposto: list[str] = field(default_factory=list)
    odv_raccomandazioni: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    riferimenti_normativi: list[str] = field(default_factory=list)
    sintesi: str = ""


# ── Prompt ────────────────────────────────────────────────────────────────────

_D231_SYSTEM = """\
Sei un avvocato penalista specializzato in responsabilità degli enti ai sensi \
del D.Lgs. 231/2001. Il tuo compito è effettuare una valutazione del rischio \
231 per una società, identificando i reati presupposto applicabili e \
formulando raccomandazioni per l'Organismo di Vigilanza (ODV).

CATALOGO REATI PRESUPPOSTO (D.Lgs. 231/2001 e successive modifiche):
- Reati contro la PA: corruzione (art. 25), concussione, malversazione (art. 24)
- Reati informatici: accesso abusivo a sistema informatico (art. 24-bis)
- Delitti di criminalità organizzata: associazione mafiosa (art. 24-ter)
- Reati societari: false comunicazioni sociali, market abuse (art. 25-ter)
- Reati in materia di salute e sicurezza: omicidio colposo, lesioni gravi (art. 25-septies)
- Reati ambientali: inquinamento ambientale (art. 25-undecies)
- Reati tributari: dichiarazione fraudolenta, emissione fatture false (art. 25-quinquiesdecies)
- Reati di riciclaggio e autoriciclaggio (art. 25-octies)
- Reati contro la personalità individuale: sfruttamento lavorativo (art. 25-quinquies)
- Impiego di cittadini di paesi terzi in condizioni di sfruttamento (art. 25-duodecies)
- Reati colposi di omicidio e lesioni gravi contro la sicurezza sul lavoro (art. 25-septies)
- Reati di market abuse (art. 25-sexies)
- Reati transnazionali (L. 146/2006)
- Cybercrime e reati informatici

LINEE GUIDA ODV (allineate a Confindustria e Assonime):
- Mappatura delle aree di rischio specifiche per settore
- Protocolli di prevenzione per ciascun reato presupposto identificato
- Sistemi di controllo interno e flussi informativi verso l'ODV
- Formazione del personale esposto al rischio
- Meccanismi di whistleblowing (D.Lgs. 24/2023)
- Clausole 231 nei contratti con fornitori/partner

RISK SCORE (0.0 – 1.0):
- 0.0–0.3: Rischio basso (settori con pochi reati presupposto applicabili)
- 0.3–0.6: Rischio medio (settori con esposizione moderata)
- 0.6–0.8: Rischio alto (settori ad alta esposizione: PA, banche, sanità, costruzioni)
- 0.8–1.0: Rischio molto alto (settori con esposizione critica multipla)

Rispondi SEMPRE e SOLO con un oggetto JSON valido, senza testo aggiuntivo:

{
  "reati_presupposto": [
    "<nome reato + riferimento art. D.Lgs. 231/2001>",
    ...
  ],
  "odv_raccomandazioni": [
    "<raccomandazione concreta e actionable>",
    ...
  ],
  "risk_score": <float 0.0-1.0>,
  "riferimenti_normativi": [
    "<art. X D.Lgs. 231/2001>",
    "<D.Lgs. Y/AAAA>",
    ...
  ],
  "sintesi": "<sintesi esecutiva in italiano>"
}
"""


# ── Service ───────────────────────────────────────────────────────────────────


def assess_231_risk(
    settore: str,
    descrizione_attivita: str,
    llm: LLMClient,
) -> Compliance231Result:
    """Perform a D.Lgs. 231/2001 risk assessment for the given entity.

    Args:
        settore:               Industry/sector of the company,
                               e.g. ``"edilizia"``, ``"bancario"``,
                               ``"farmaceutico"``, ``"pubblica amministrazione"``.
        descrizione_attivita:  Description of business activities (free text,
                               max 2 000 chars recommended).
        llm:                   :class:`~src.lexreview.agent.llm_client.LLMClient`.

    Returns:
        :class:`Compliance231Result` with predicate offences, ODV
        recommendations, risk score, and statutory references.
    """
    log.info(
        "d231.assess_231_risk",
        extra={"settore": settore, "descrizione_len": len(descrizione_attivita)},
    )

    # Truncate description to avoid excessive token spend
    MAX_DESC_CHARS = 2_000
    if len(descrizione_attivita) > MAX_DESC_CHARS:
        descrizione_attivita = descrizione_attivita[:MAX_DESC_CHARS]
        log.debug("d231: descrizione_attivita truncated")

    messages = [
        {"role": "system", "content": _D231_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Settore di attività: {settore}\n\n"
                f"Descrizione delle attività svolte:\n{descrizione_attivita}\n\n"
                "Effettua la valutazione del rischio D.Lgs. 231/2001 per questa società."
            ),
        },
    ]

    raw = llm.complete(messages)

    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip().strip("` ")
        data = json.loads(clean)

        raw_score = data.get("risk_score", 0.0)
        try:
            risk_score = max(0.0, min(1.0, float(raw_score)))
        except (TypeError, ValueError):
            risk_score = 0.0

        return Compliance231Result(
            reati_presupposto=list(data.get("reati_presupposto") or []),
            odv_raccomandazioni=list(data.get("odv_raccomandazioni") or []),
            risk_score=round(risk_score, 4),
            riferimenti_normativi=list(data.get("riferimenti_normativi") or []),
            sintesi=str(data.get("sintesi", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        log.warning(
            "d231: failed to parse LLM JSON",
            extra={"error": str(exc), "raw_preview": raw[:200]},
        )
        return Compliance231Result(
            sintesi=f"Valutazione non disponibile: errore parsing LLM ({exc})",
        )
