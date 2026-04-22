"""Pydantic v2 data-models for legal extraction.

Classes
-------
Clause
    A single detected legal clause with type, text, span, and confidence.
LegalEntities
    Structured container for all named entities extracted from a legal document.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Clause(BaseModel):
    """A detected legal clause inside a document.

    Attributes:
        type:       Semantic category of the clause (e.g. ``"indemnification"``).
        text:       Raw clause text as it appears in the source document.
        span:       Character-offset tuple ``(start, end)`` in the source string.
        confidence: Classifier confidence in [0.0, 1.0].
    """

    type: str = Field(..., description="Legal clause category label.")
    text: str = Field(..., description="Raw clause text from the source document.")
    span: tuple[int, int] = Field(
        ..., description="Character offsets (start, end) in the source document."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Clause detection confidence score."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "indemnification",
                "text": "Party A shall indemnify Party B against all losses...",
                "span": [0, 52],
                "confidence": 0.91,
            }
        }
    }


class LegalEntities(BaseModel):
    """Container for all named entities extracted from a legal document.

    Attributes:
        parties:       List of party names (organizations, individuals).
        dates:         List of date strings found in the document.
        amounts:       List of monetary amounts (with currency).
        jurisdictions: List of governing law / jurisdiction references.
    """

    parties: list[str] = Field(
        default_factory=list, description="Contracting party names."
    )
    dates: list[str] = Field(
        default_factory=list, description="Date strings found in the document."
    )
    amounts: list[str] = Field(
        default_factory=list, description="Monetary amounts with currency symbols."
    )
    jurisdictions: list[str] = Field(
        default_factory=list, description="Governing law / jurisdiction references."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "parties": ["Acme Corp.", "Beta LLC"],
                "dates": ["January 1, 2024", "2024-12-31"],
                "amounts": ["$50,000.00", "USD 1,000,000"],
                "jurisdictions": ["State of Delaware", "New York"],
            }
        }
    }
