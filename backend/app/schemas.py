from pydantic import BaseModel
from typing import List, Optional, Any

class DocumentOut(BaseModel):
    id: int
    filename: str
    title: Optional[str] = None
    tags: Optional[str] = None

class UploadResponse(BaseModel):
    document_id: int
    workflow_id: int

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    scope_doc_ids: Optional[List[int]] = None
    action: Optional[str] = None  # "contract" | "case_summary" | "client_letter"

class Source(BaseModel):
    document_id: int
    title: Optional[str] = None
    chunk: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    laws: List[Source]
    workflow_id: Optional[int] = None

class WorkflowOut(BaseModel):
    id: int
    type: str
    status: str
    payload: Optional[str]
    result: Optional[str]