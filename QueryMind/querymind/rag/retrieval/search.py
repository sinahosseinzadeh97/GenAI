from querymind.rag.ingestion.embedder import get_embedding
from querymind.rag.store.vector_store import similarity_search as v_search

async def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    emb = get_embedding(query)
    results = await v_search(emb, top_k)
    return results
