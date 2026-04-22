"""RAGForge Italia — Italian Legal Knowledge sub-package.

Provides:

- Ingestion connectors for nine authoritative Italian legal sources.
- A typed metadata schema (``ItalianLegalMetadata``) with Italian-specific
  provenance fields.
- **ItalianLegalRAGAgent** — production RAG pipeline powered by
  Claude 3.5 Sonnet (Anthropic) with a seven-section Italian CoT format.

Quick-start — connectors::

    from src.italia.connectors import CONNECTORS

    connector = CONNECTORS["normattiva"]()
    documents = connector.fetch(codice="civile")
    print(len(documents), "articles fetched")

Quick-start — RAG agent::

    from src.italia import ItalianLegalRAGAgent
    from src.lexreview.agent.llm_client import LLMClient

    llm = LLMClient(provider="anthropic", model="claude-sonnet-4-5", max_tokens=4096)
    agent = ItalianLegalRAGAgent(retriever=hybrid, reranker=reranker, llm=llm)
    resp = agent.answer("Quali sono i presupposti dell'art. 2043 c.c.?")
    print(resp.risposta)
"""

from src.italia.agent import (
    ItalianAgentResponse,
    ItalianCitation,
    ItalianLegalRAGAgent,
    build_prompt_it,
)
from src.italia.metadata import ItalianLegalMetadata, TipoDocumento

__all__ = [
    # Metadata
    "ItalianLegalMetadata",
    "TipoDocumento",
    # Agent
    "ItalianLegalRAGAgent",
    "ItalianAgentResponse",
    "ItalianCitation",
    "build_prompt_it",
]
