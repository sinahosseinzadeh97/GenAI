"""agent sub-package — LegalRAGAgent, LLMClient, prompts, models."""

from src.lexreview.agent.legal_rag_agent import LegalRAGAgent
from src.lexreview.agent.llm_client import LLMClient, LLMError
from src.lexreview.agent.models import AgentResponse, Citation
from src.lexreview.agent.prompts import build_prompt

__all__ = [
    "LegalRAGAgent",
    "LLMClient",
    "LLMError",
    "AgentResponse",
    "Citation",
    "build_prompt",
]
