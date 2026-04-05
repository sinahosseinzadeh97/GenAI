"""
Database engine and connection management.
"""

from dataclasses import dataclass
from typing import Any, AsyncGenerator
from contextlib import asynccontextmanager

import aiosqlite

from querymind.config import settings


@dataclass
class QueryResultData:
    """Raw result structure from the database engine."""
    columns: list[str]
    rows: list[dict[str, Any]]


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[aiosqlite.Connection, None]:
    """
    Get an async connection to the SQLite database.
    
    Yields:
        aiosqlite.Connection: An async SQLite connection.
    """
    async with aiosqlite.connect(settings.db_path) as db:
        # Enable foreign key constraint checking
        await db.execute("PRAGMA foreign_keys = ON")
        # Use aiosqlite.Row factory for dictionary-like access
        db.row_factory = aiosqlite.Row
        yield db


async def execute_query(query: str, parameters: list[Any] | None = None) -> QueryResultData:
    """
    Execute a SQL query against the database and return the columns and rows.
    
    Args:
        query: The SQL query string to execute.
        parameters: Optional list of parameters for the query.
        
    Returns:
        QueryResultData containing column definitions and row dictionaries.
    """
    if parameters is None:
        parameters = []
        
    async with get_db_connection() as db:
        async with db.execute(query, parameters) as cursor:
            rows = await cursor.fetchall()
            
            columns = []
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                
            return QueryResultData(
                columns=columns,
                rows=[dict(row) for row in rows]
            )
