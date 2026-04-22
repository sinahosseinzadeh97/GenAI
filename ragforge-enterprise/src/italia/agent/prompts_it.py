"""Prompt templates italiani per l'ItalianLegalRAGAgent.

Il formato CoT adottato è a sette sezioni::

    [COMPRENSIONE]      — Riassunto della questione giuridica
    [NORME_APPLICABILI] — Norme di riferimento (codici, leggi, regolamenti)
    [GIURISPRUDENZA]    — Sentenze rilevanti dal corpus recuperato
    [RAGIONAMENTO]      — Ragionamento sillogistico: norma → fatto → effetto
    [RISPOSTA]          — Conclusione giuridica chiara e motivata
    [FONTI]             — Lista completa delle fonti citate
    [AVVERTENZE]        — Limitazioni e invito a consultare un professionista

Functions
---------
build_prompt_it
    Assembla messaggi in formato OpenAI/Anthropic da query + contesti.
"""

from __future__ import annotations

from src.vectorstore.schema import SearchResult

# ── System prompt ─────────────────────────────────────────────────────────────

ITALIAN_LEGAL_SYSTEM_PROMPT = """\
Sei un assistente giuridico esperto del diritto italiano ed europeo.
Rispondi sempre in italiano, con linguaggio tecnico-giuridico preciso.

REGOLE FONDAMENTALI:
1. Cita sempre la fonte normativa esatta (es. "Art. 2043 c.c.", "Art. 7 GDPR")
2. Distingui tra norma vigente e abrogata — segnala sempre la data di vigenza
3. Per le sentenze, cita: tribunale, sezione, numero e anno
4. Non fornire pareri legali vincolanti — indica sempre che si tratta di
   informazioni giuridiche generali
5. Se la risposta richiede aggiornamenti normativi recenti, segnalalo esplicitamente
6. Segui il ragionamento sillogistico: norma → fatto → conseguenza giuridica

AREE DI COMPETENZA PRIORITARIA:
- Diritto civile (contratti, responsabilità, proprietà)
- Diritto commerciale e societario
- Diritto del lavoro
- Diritto penale dell'economia (D.Lgs. 231/2001)
- GDPR e normativa privacy italiana (D.Lgs. 196/2003)
- Diritto amministrativo e appalti pubblici (D.Lgs. 36/2023)
- Diritto tributario
- Diritto europeo applicato in Italia
"""

# ── CoT template italiano ─────────────────────────────────────────────────────

_COT_TEMPLATE_IT = """\
## Questione Giuridica
{query}

## Estratti di Riferimento Recuperati
{context_block}

## Istruzioni per il Ragionamento
Svolgi i seguenti passi in sequenza, quindi fornisci la risposta definitiva.

[COMPRENSIONE]
Riassumi con precisione la questione giuridica posta. Identifica i concetti
chiave, le parti in causa e l'obiettivo dell'utente.

[NORME_APPLICABILI]
Identifica e cita le norme di riferimento pertinenti (articoli di codice,
disposizioni di legge, regolamenti, direttive europee). Indica sempre:
- il testo normativo (es. c.c., c.p., D.Lgs. 196/2003)
- l'articolo esatto (es. Art. 2043 c.c.)
- lo stato di vigenza (vigente / abrogato / modificato — con data)

[GIURISPRUDENZA]
Cita le sentenze rilevanti presenti negli estratti recuperati. Per ciascuna
sentenza indica: organo giudicante, sezione, numero e anno
(es. "Cass. civ., Sez. I, n. 12345/2023").
Se nessuna sentenza è presente nel corpus, indica esplicitamente l'assenza.

[RAGIONAMENTO]
Applica le norme ai fatti con ragionamento sillogistico:
  Premessa maggiore (norma) → Premessa minore (fatto) → Conclusione (effetto giuridico).
Segnala eventuali orientamenti giurisprudenziali contrastanti o lacune normative.

[RISPOSTA]
Fornisci la conclusione giuridica chiara, motivata e fondata sugli estratti
recuperati e sulle norme citate. Usa il formato:
  "Alla luce di quanto sopra, [conclusione]."

[FONTI]
Elenca in modo strutturato tutte le fonti citate, compresi:
- Riferimenti normativi (es. Art. 2043 c.c.)
- Sentenze (es. Cass. civ., Sez. I, n. 12345/2023)
- Documenti del corpus [CHUNK: <chunk_id>]

[AVVERTENZE]
Indica le limitazioni della risposta:
- Aggiornamento normativo: specificare se potrebbero esserci novità successive
- Ambito di applicazione: la risposta si basa sugli estratti disponibili
- Invito esplicito: "Si raccomanda di consultare un avvocato o notaio per
  una consulenza personalizzata sul caso specifico."

Inizia la risposta:
"""


def _format_context_it(results: list[SearchResult]) -> str:
    """Formatta i chunk recuperati in un blocco di contesto numerato.

    Args:
        results: Risultati di ricerca riordinati dal reranker.

    Returns:
        Stringa multi-riga con ogni chunk prefissato da ID, fonte e punteggio.
    """
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        source = r.metadata.get("source", "sconosciuta")
        tipo = r.metadata.get("tipo_documento", "")
        riferimento = r.metadata.get("riferimento_normativo", "")
        header_parts = [
            f"[{i}] CHUNK_ID={r.chunk_id}",
            f"FONTE={source}",
            f"SCORE={r.score:.3f}",
        ]
        if tipo:
            header_parts.append(f"TIPO={tipo}")
        if riferimento:
            header_parts.append(f"RIFERIMENTO={riferimento}")
        lines.append(" | ".join(header_parts) + f"\n{r.content.strip()}\n")
    return "\n---\n".join(lines)


def build_prompt_it(
    query: str,
    contexts: list[SearchResult],
) -> list[dict[str, str]]:
    """Assembla messaggi in formato OpenAI/Anthropic per l'ItalianLegalRAGAgent.

    I messaggi prodotti sono compatibili con l'API Anthropic (``messages.create``)
    e con l'API OpenAI Chat Completions — il ``system`` prompt viene estratto
    dal primo elemento con ``role == "system"`` dall'``LLMClient``.

    Args:
        query:    Domanda giuridica dell'utente.
        contexts: Risultati di ricerca riordinati da includere come contesto.

    Returns:
        Lista di ``{"role": ..., "content": ...}`` pronta per ``LLMClient.complete()``.

    Example::

        messages = build_prompt_it(
            "Quali sono i requisiti dell'art. 2043 c.c.?", results
        )
        response = llm.complete(messages)
    """
    context_block = (
        _format_context_it(contexts) if contexts else "(Nessun estratto recuperato)"
    )
    user_content = _COT_TEMPLATE_IT.format(
        query=query,
        context_block=context_block,
    )
    return [
        {"role": "system", "content": ITALIAN_LEGAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
