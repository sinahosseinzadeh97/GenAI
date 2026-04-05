from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import List

from querymind.rag.ingestion.pdf_parser import parse_pdf
from querymind.rag.ingestion.embedder import get_embedding
from querymind.rag.store.vector_store import save_chunks
from querymind.rag.retrieval.search import search_chunks
from querymind.rag.generation.llm_client import generate_answer

router = APIRouter(prefix="/rag", tags=["RAG"])

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class ExtractRequest(BaseModel):
    query: str
    field: str

class CompareRequest(BaseModel):
    query: str
    filenames: List[str]

@router.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename:
        return {"error": "Filename missing."}
    
    content = await file.read()
    
    # 1. Parse PDF
    parsed_chunks = parse_pdf(content, file.filename)
    
    # 2. Embed
    for chunk in parsed_chunks:
        chunk["embedding"] = get_embedding(chunk["text"])
        
    # 3. Store
    await save_chunks(parsed_chunks)
    
    return {"message": f"Successfully ingested {len(parsed_chunks)} chunks from {file.filename}"}

@router.post("/search")
async def search_documents(req: SearchRequest):
    results = await search_chunks(req.query, req.top_k)
    return {"results": results}

@router.post("/extract")
async def extract_field(req: ExtractRequest):
    results = await search_chunks(req.query + f" {req.field}", top_k=5)
    answer = generate_answer(f"Extract the {req.field} based on the query: {req.query}", results)
    
    return {"extracted": answer, "sources": [{"filename": r["filename"], "page": r["page_number"]} for r in results]}

@router.post("/compare")
async def compare_documents(req: CompareRequest):
    results = await search_chunks(req.query, top_k=10)
    
    # filter by filenames if specified
    if req.filenames:
        results = [r for r in results if r["filename"] in req.filenames]
        
    answer = generate_answer(req.query, results)
    
    return {"comparison": answer, "sources": [{"filename": r["filename"], "page": r["page_number"]} for r in results]}
