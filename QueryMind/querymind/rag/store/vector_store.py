import asyncpg
from pgvector.asyncpg import register_vector
from querymind.config import settings

_pool = None

async def init_vector_store():
    global _pool
    if _pool is None:
        if not settings.postgres_dsn:
            raise RuntimeError("POSTGRES_DSN not configured")
        _pool = await asyncpg.create_pool(
            settings.postgres_dsn,
            min_size=2,
            max_size=10
        )
    
    async with _pool.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                page_number INT,
                content TEXT,
                embedding vector(1536)
            )
        """)

async def save_chunks(chunks: list[dict]):
    if _pool is None:
        await init_vector_store()
        
    async with _pool.acquire() as conn:
        await register_vector(conn)
        stmt = "INSERT INTO rag_chunks (filename, page_number, content, embedding) VALUES ($1, $2, $3, $4)"
        
        for c in chunks:
            await conn.execute(
                stmt,
                c["filename"],
                c["page_number"],
                c.get("content") or c.get("text"),
                c["embedding"]
            )

async def similarity_search(embedding: list[float], top_k: int = 5) -> list[dict]:
    if _pool is None:
        await init_vector_store()
        
    async with _pool.acquire() as conn:
        await register_vector(conn)
        
        # Using cosine distance <=> (<=>)
        rows = await conn.fetch("""
            SELECT id, filename, page_number, content
            FROM rag_chunks
            ORDER BY embedding <=> $1
            LIMIT $2
        """, embedding, top_k)
        
        return [dict(r) for r in rows]
