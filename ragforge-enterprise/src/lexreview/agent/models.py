"""Pydantic v2 models for the LexReview RAG agent.

Classes
-------
Citation
    A single source chunk cited in an agent answer.
AgentResponse
    Complete structured response returned by LegalRAGAgent.

EU AI Act compliance (Annex III — high-risk AI system)
------------------------------------------------------
``AgentResponse.requires_human_review`` is always ``True`` and cannot be
overridden.  This field signals to downstream systems (API clients, UI, audit
logs) that every AI-generated legal answer **must** be reviewed by a qualified
human professional before being acted upon, as required by the EU AI Act for
high-risk systems in the administration of justice.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """A source chunk cited in an agent answer.

    Attributes:
        chunk_id: Qdrant chunk identifier used to verify provenance.
        content:  Raw text of the cited passage.
        score:    Relevance score from the reranker (cross-encoder logit).
        source:   Optional document source path / URL from chunk metadata.
    """

    chunk_id: str = Field(..., description="Unique identifier of the retrieved chunk.")
    content: str = Field(..., description="Raw text of the cited passage.")
    score: float = Field(..., description="Relevance score from the reranker.")
    source: str | None = Field(default=None, description="Originating document path.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "chunk_id": "abc-123",
                "content": "The indemnifying party shall defend...",
                "score": 0.94,
                "source": "contracts/nda_2024.pdf",
            }
        }
    }


class AgentResponse(BaseModel):
    """Complete structured response from the LegalRAGAgent.

    Attributes:
        answer:               Final synthesised answer to the user's legal query.
        citations:            Ordered list of source chunks the answer draws from.
        confidence:           Agent's self-assessed confidence in [0.0, 1.0].
        reasoning_steps:      Chain-of-Thought trace extracted from the LLM response.
        latency_ms:           Total wall-clock time for the agent pipeline in ms.
        requires_human_review: Always ``True``.  Non-overridable EU AI Act Annex III
                              mandate — every legal AI output must be reviewed by a
                              qualified human professional before being relied upon.
    """

    answer: str = Field(..., description="Synthesised answer to the legal query.")
    citations: list[Citation] = Field(
        default_factory=list, description="Source passages the answer is grounded in."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Heuristic confidence score in [0, 1] derived from cross-encoder logits. "
            "Not a calibrated probability; accuracy depends on the reranker model used."
        ),
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Chain-of-Thought steps extracted from LLM response.",
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="Total pipeline latency in milliseconds."
    )
    requires_human_review: bool = Field(
        default=True,
        description=(
            "EU AI Act Annex III — high-risk AI system flag. "
            "Always True: every AI-generated legal answer must be reviewed by a "
            "qualified human professional before being relied upon. "
            "This field cannot be set to False."
        ),
    )

    @field_validator("requires_human_review", mode="before")
    @classmethod
    def _enforce_human_review(cls, v: object) -> bool:
        """Ensure requires_human_review is always True regardless of input.

        This validator is non-negotiable: the EU AI Act (Annex III) classifies
        legal AI systems as high-risk and mandates human oversight.  No caller
        may disable this flag programmatically.

        Args:
            v: Input value (ignored).

        Returns:
            Always ``True``.
        """
        return True  # Non-overridable: EU AI Act Annex III human oversight mandate.

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "The governing law is the State of Delaware.",
                "citations": [],
                "confidence": 0.88,
                "reasoning_steps": [
                    "Identified jurisdiction clause in chunk abc-123.",
                    "Confirmed 'State of Delaware' appears twice.",
                ],
                "latency_ms": 312.4,
                "requires_human_review": True,
            }
        }
    }
