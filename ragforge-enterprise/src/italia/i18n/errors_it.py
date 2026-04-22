"""Italian error message catalogue for RAGForge Italia API.

All error messages that the API returns when the client sends
``Accept-Language: it`` or ``Accept-Language: it-IT`` are defined here.

Design principles
-----------------
- Messages are **formal** (Italian "lei" register) — suitable for professional
  legal practitioners (avvocati, notai, magistrati).
- Placeholders use ``{key}`` syntax; call :func:`get_error_message_it` to
  render them safely.
- Every key maps to an HTTP status code and a human-readable Italian sentence.

Usage::

    from src.italia.i18n.errors_it import get_error_message_it

    msg = get_error_message_it("RATE_LIMIT_EXCEEDED", retry_after=60)
    # → "Limite di richieste superato. Riprovare tra 60 secondi."
"""

from __future__ import annotations

from typing import Any

# ── Catalogue ─────────────────────────────────────────────────────────────────
# Keys match FastAPI/Starlette exception types and custom RAGForge error codes.
# HTTP status code is embedded as the first value in each tuple for routing.

ERROR_MESSAGES_IT: dict[str, tuple[int, str]] = {
    # 4xx — Client errors
    "NOT_FOUND": (
        404,
        "Risorsa non trovata.",
    ),
    "INVALID_REQUEST": (
        422,
        "Richiesta non valida: {detail}",
    ),
    "VALIDATION_ERROR": (
        422,
        "Errore di validazione: {detail}",
    ),
    "AUTHENTICATION_REQUIRED": (
        401,
        "Autenticazione richiesta. Verificare l'header X-API-Key.",
    ),
    "FORBIDDEN": (
        403,
        "Accesso negato. L'utente non dispone dei permessi necessari per questa operazione.",
    ),
    "RATE_LIMIT_EXCEEDED": (
        429,
        "Limite di richieste superato. Riprovare tra {retry_after} secondi.",
    ),
    "METHOD_NOT_ALLOWED": (
        405,
        "Metodo HTTP non consentito per questo endpoint.",
    ),
    "REQUEST_TOO_LARGE": (
        413,
        "Richiesta troppo grande. La dimensione massima consentita è {max_size} MB.",
    ),
    "UNSUPPORTED_MEDIA_TYPE": (
        415,
        "Tipo di contenuto non supportato. Utilizzare 'application/json'.",
    ),
    "CONFLICT": (
        409,
        "Conflitto: la risorsa esiste già. Utilizzare force_reindex=true per sovrascriverla.",
    ),
    # GDPR-specific 4xx
    "GDPR_RIGHT_TO_ERASURE": (
        200,
        (
            "Richiesta di cancellazione ai sensi dell'art. 17 GDPR ricevuta. "
            "Il documento e i relativi dati personali saranno eliminati entro 72 ore."
        ),
    ),
    "GDPR_DATA_RESIDENCY_VIOLATION": (
        403,
        (
            "Operazione non consentita: i dati devono rimanere nel territorio italiano "
            "(regione AWS eu-south-1 / Azure Italy North) ai sensi del segreto professionale."
        ),
    ),
    # EU AI Act — High-risk system warnings
    "AI_ACT_LOW_CONFIDENCE": (
        200,
        (
            "Avviso ai sensi del Regolamento UE 2024/1689 (AI Act): "
            "l'output generato ha un punteggio di fiducia inferiore alla soglia ({confidence:.0%}). "
            "Si raccomanda la supervisione umana prima dell'utilizzo in contesti giuridici."
        ),
    ),
    "AI_ACT_HUMAN_OVERSIGHT_REQUIRED": (
        200,
        (
            "Sistema ad alto rischio (AI Act, All. III): "
            "questo output richiede revisione da parte di un professionista legale qualificato."
        ),
    ),
    # 5xx — Server errors
    "INTERNAL_ERROR": (
        500,
        "Errore interno del server. Riprovare più tardi o contattare il supporto.",
    ),
    "SERVICE_UNAVAILABLE": (
        503,
        (
            "Servizio temporaneamente non disponibile. "
            "Manutenzione programmata in corso. Riprovare tra {retry_after} minuti."
        ),
    ),
    "GATEWAY_TIMEOUT": (
        504,
        "Timeout del gateway. La fonte esterna non ha risposto nei tempi previsti.",
    ),
    "VECTOR_STORE_ERROR": (
        503,
        "Errore nel database vettoriale (Qdrant). Riprovare o contattare l'amministratore.",
    ),
    "LLM_ERROR": (
        503,
        "Errore nel modello linguistico. Riprovare più tardi.",
    ),
    # Connector-specific errors
    "CONNECTOR_HTTP_ERROR": (
        502,
        "Errore HTTP {status_code} dalla fonte {source}: {message}",
    ),
    "CONNECTOR_PARSE_ERROR": (
        502,
        "Impossibile analizzare la risposta dalla fonte {source}. Contattare il supporto.",
    ),
    "CONNECTOR_RATE_LIMIT": (
        429,
        "Limite di richieste superato per la fonte {source}. Riprovare tra {retry_after} secondi.",
    ),
    "NOTARTEL_UNAVAILABLE": (
        503,
        (
            "Il servizio Notartel non è raggiungibile al momento. "
            "Verificare le credenziali NOTARTEL_TOKEN o riprovare."
        ),
    ),
    "SIECIC_READONLY_VIOLATION": (
        403,
        (
            "Il connettore SIECIC/SICID è di sola lettura (art. 2 D.Lgs. n. 82/2005 — CAD). "
            "Non sono consentite operazioni di scrittura sui sistemi del Ministero della Giustizia."
        ),
    ),
    "FILENET_AUTH_ERROR": (
        401,
        "Autenticazione FileNet/Documentum fallita. Verificare FILENET_USERNAME e FILENET_PASSWORD.",
    ),
    "LEXISNEXIS_TOKEN_ERROR": (
        401,
        "Impossibile ottenere il token OAuth 2.0 da LexisNexis Italia. Verificare le credenziali.",
    ),
    # Webhook errors
    "WEBHOOK_SIGNATURE_INVALID": (
        401,
        (
            "Firma del webhook non valida. "
            "Verificare il segreto HMAC-SHA256 configurato per l'integrazione."
        ),
    ),
    "WEBHOOK_PAYLOAD_INVALID": (
        400,
        "Payload del webhook non valido o mancante. Verificare il formato dell'evento.",
    ),
}


def get_error_message_it(key: str, **kwargs: Any) -> str:
    """Return the Italian error message for *key*, with placeholders filled.

    Args:
        key:      Error code key from :data:`ERROR_MESSAGES_IT`.
        **kwargs: Placeholder values interpolated into the message template.

    Returns:
        Formatted Italian string.  Falls back to a generic message if *key*
        is unknown.

    Example::

        >>> get_error_message_it("RATE_LIMIT_EXCEEDED", retry_after=30)
        'Limite di richieste superato. Riprovare tra 30 secondi.'
    """
    if key not in ERROR_MESSAGES_IT:
        return f"Errore sconosciuto ({key}). Contattare il supporto."
    _, template = ERROR_MESSAGES_IT[key]
    try:
        return template.format(**kwargs) if kwargs else template
    except KeyError:
        # Return the raw template if caller forgot a placeholder.
        return template


def get_http_status_for_key(key: str) -> int:
    """Return the HTTP status code associated with *key*.

    Args:
        key: Error code key from :data:`ERROR_MESSAGES_IT`.

    Returns:
        Integer HTTP status code; 500 for unknown keys.
    """
    entry = ERROR_MESSAGES_IT.get(key)
    return entry[0] if entry else 500
