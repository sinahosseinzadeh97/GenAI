"""
PostgreSQL database engine using asyncpg.
"""

import asyncpg
from typing import Any
from querymind.config import settings
from querymind.database.engine import QueryResultData

_pool: asyncpg.Pool | None = None

async def _get_pool() -> asyncpg.Pool:
    """Return existing pool or create new one."""
    global _pool
    if _pool is None:
        if not settings.postgres_dsn:
            raise RuntimeError("POSTGRES_DSN not configured")
        _pool = await asyncpg.create_pool(
            settings.postgres_dsn,
            min_size=2,
            max_size=10
        )
    return _pool

def _convert_placeholders(query: str) -> str:
    """Convert SQLite ? placeholders to PostgreSQL $1, $2..."""
    count = 0
    result = []
    for char in query:
        if char == "?":
            count += 1
            result.append(f"${count}")
        else:
            result.append(char)
    return "".join(result)

async def execute_query(
    query: str,
    parameters: list[Any] | None = None
) -> QueryResultData:
    """Execute a query against PostgreSQL."""
    pool = await _get_pool()
    pg_query = _convert_placeholders(query)
    
    async with pool.acquire() as conn:
        if parameters:
            rows = await conn.fetch(pg_query, *parameters)
        else:
            rows = await conn.fetch(pg_query)
        
        if not rows:
            return QueryResultData(columns=[], rows=[])
        
        columns = list(rows[0].keys())
        return QueryResultData(
            columns=columns,
            rows=[dict(row) for row in rows]
        )

async def close_pool() -> None:
    """Call on shutdown to release connections."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
