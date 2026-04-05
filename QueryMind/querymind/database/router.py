"""
Database execution router.
"""

from querymind.config import settings
from querymind.database.engine import execute_query as sqlite_execute
from querymind.database.pg_engine import execute_query as pg_execute
from querymind.database.engine import QueryResultData
from typing import Any

async def execute_query(
    query: str,
    parameters: list[Any] | None = None
) -> QueryResultData:
    """Route query to correct database engine based on settings."""
    if settings.database_type == "postgresql":
        return await pg_execute(query, parameters)
    return await sqlite_execute(query, parameters)
