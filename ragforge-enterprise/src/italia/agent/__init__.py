"""RAGForge Italia — agent sub-package.

Exposes :class:`ItalianLegalRAGAgent` and its companion prompt/model helpers.

Quick-start::

    from src.italia.agent import ItalianLegalRAGAgent, build_prompt_it

    agent = ItalianLegalRAGAgent(
        retriever=hybrid,
        reranker=reranker,
        llm=LLMClient(provider="anthropic", model="claude-sonnet-4-5"),
    )
    response = agent.answer("Quali sono i presupposti della responsabilità ex art. 2043 c.c.?")
    print(response.risposta)
"""

from src.italia.agent.legal_rag_agent_it import ItalianLegalRAGAgent
from src.italia.agent.models_it import ItalianAgentResponse, ItalianCitation
from src.italia.agent.prompts_it import build_prompt_it

__all__ = [
    "ItalianLegalRAGAgent",
    "ItalianAgentResponse",
    "ItalianCitation",
    "build_prompt_it",
]
