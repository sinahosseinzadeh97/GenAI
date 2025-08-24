from __future__ import annotations
from typing import Optional, Any
from datetime import datetime
from sqlmodel import SQLModel, Field, Relationship

class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    content_type: str
    path: str
    text: str
    title: Optional[str] = None
    tags: Optional[str] = None  # comma-separated
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Workflow(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    type: str  # document_analysis | query | draft
    status: str = "pending"  # pending|running|completed|failed
    payload: Optional[str] = None  # JSON string
    result: Optional[str] = None   # JSON string
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)