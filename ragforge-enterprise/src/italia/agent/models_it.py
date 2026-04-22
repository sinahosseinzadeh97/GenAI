"""Modelli Pydantic v2 per l'ItalianLegalRAGAgent.

Classes
-------
ItalianCitation
    Singola fonte citata nella risposta dell'agente.
ItalianAgentResponse
    Risposta strutturata completa restituita da ItalianLegalRAGAgent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ItalianCitation(BaseModel):
    """Fonte citata nella risposta dell'agente giuridico italiano.

    Attributes:
        chunk_id:            Identificatore univoco del chunk Qdrant.
        content:             Testo grezzo del passo citato.
        score:               Punteggio di rilevanza del reranker.
        source:              URL o percorso del documento originale.
        riferimento_normativo: Riferimento normativo estratto dai metadati
                               (es. "Art. 2043 c.c.").
        tipo_documento:      Tipo di documento legale (es. "sentenza", "legge").
    """

    chunk_id: str = Field(..., description="Identificatore univoco del chunk.")
    content: str = Field(..., description="Testo grezzo del passo citato.")
    score: float = Field(..., description="Punteggio di rilevanza del reranker.")
    source: str | None = Field(
        default=None, description="URL o percorso del documento originale."
    )
    riferimento_normativo: str | None = Field(
        default=None,
        description="Riferimento normativo del chunk (es. 'Art. 2043 c.c.').",
    )
    tipo_documento: str | None = Field(
        default=None,
        description="Tipo di documento (es. 'sentenza', 'legge', 'regolamento').",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "chunk_id": "norm-2043-cc",
                "content": "Qualunque fatto doloso o colposo che cagioni ad altri...",
                "score": 0.97,
                "source": "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1942-03-16;262",
                "riferimento_normativo": "Art. 2043 c.c.",
                "tipo_documento": "codice",
            }
        }
    }


class ItalianAgentResponse(BaseModel):
    """Risposta strutturata completa dell'ItalianLegalRAGAgent.

    Le sezioni seguono fedelmente il formato CoT sette-blocchi::

        comprensione        → [COMPRENSIONE]
        norme_applicabili   → [NORME_APPLICABILI]
        giurisprudenza      → [GIURISPRUDENZA]
        ragionamento        → [RAGIONAMENTO]
        risposta            → [RISPOSTA]
        fonti               → [FONTI]
        avvertenze          → [AVVERTENZE]

    Attributes:
        risposta:           Conclusione giuridica principale.
        citations:          Elenco ordinato delle fonti citate.
        confidence:         Punteggio euristico di confidenza in [0.0, 1.0].
        comprensione:       Riassunto della questione giuridica.
        norme_applicabili:  Norme di riferimento identificate.
        giurisprudenza:     Sentenze rilevanti citate.
        ragionamento:       Ragionamento sillogistico applicato.
        fonti_raw:          Lista grezza delle fonti dalla sezione [FONTI].
        avvertenze:         Limitazioni e avvertenze professionali.
        latency_ms:         Latenza totale della pipeline in millisecondi.
    """

    risposta: str = Field(..., description="Conclusione giuridica sintetizzata.")
    citations: list[ItalianCitation] = Field(
        default_factory=list,
        description="Fonti su cui si basa la risposta.",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Punteggio euristico di confidenza in [0, 1] derivato dai logit "
            "del reranker. Non è una probabilità calibrata."
        ),
    )
    comprensione: str = Field(
        default="",
        description="Riassunto della questione giuridica [COMPRENSIONE].",
    )
    norme_applicabili: str = Field(
        default="",
        description="Norme di riferimento identificate [NORME_APPLICABILI].",
    )
    giurisprudenza: str = Field(
        default="",
        description="Sentenze rilevanti citate [GIURISPRUDENZA].",
    )
    ragionamento: str = Field(
        default="",
        description="Ragionamento sillogistico applicato [RAGIONAMENTO].",
    )
    fonti_raw: str = Field(
        default="",
        description="Elenco grezzo delle fonti dalla sezione [FONTI].",
    )
    avvertenze: str = Field(
        default="",
        description="Limitazioni e invito a consultare un professionista [AVVERTENZE].",
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="Latenza totale della pipeline in ms."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "risposta": "Alla luce di quanto sopra, il debitore è responsabile ai sensi dell'art. 2043 c.c.",
                "citations": [],
                "confidence": 0.91,
                "comprensione": "L'utente chiede i presupposti della responsabilità aquiliana.",
                "norme_applicabili": "Art. 2043 c.c. (vigente)",
                "giurisprudenza": "Cass. civ., Sez. III, n. 5678/2022",
                "ragionamento": "Art. 2043 c.c. → fatto illecito → obbligo risarcitorio.",
                "fonti_raw": "- Art. 2043 c.c.\n- [CHUNK: norm-2043-cc]",
                "avvertenze": "Si raccomanda di consultare un avvocato per una consulenza personalizzata.",
                "latency_ms": 843.2,
            }
        }
    }
