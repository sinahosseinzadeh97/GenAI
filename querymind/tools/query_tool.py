"""
MCP tool to handle natural language queries, translate to SQL, execute, and return results.
"""

import re
from typing import Any, Callable, Awaitable

from querymind.schemas.models import QueryRequest, QueryResult, QueryError, ColumnInfo, QueryInsight
from querymind.database.router import execute_query
from querymind.database.readonly_engine import ReadOnlyEngine
from querymind.database.engine import QueryResultData
from querymind.config import settings
from querymind.tools.schema_tool import get_db_schema
from querymind.prompts.sql_generator import get_sql_generation_prompt, build_user_prompt
from querymind.memory.conversation import ConversationMemory
from querymind.cache.query_cache import QueryCache
from querymind.prompts.insight_generator import (
    INSIGHT_SYSTEM_PROMPT,
    build_insight_prompt
)


def sanitize_sql(raw: str) -> str:
    """Clean the generated SQL string before execution."""
    cleaned = raw.strip()
    cleaned = cleaned.removeprefix("```sql").removeprefix("```")
    cleaned = cleaned.removesuffix("```")
    return cleaned.strip()


def strip_sql_comments(sql: str) -> str:
    # Remove multi-line comments
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # Remove single-line comments
    sql = re.sub(r'--[^\n]*', '', sql)
    return sql


def validate_sql(sql: str) -> bool:
    """
    Basic safety check: ensure the query is a SELECT statement and
    does not contain harmful modification keywords.
    """
    sql = strip_sql_comments(sql)
    upper_sql = sql.upper().strip()
    if not upper_sql.startswith("SELECT"):
        return False
        
    forbidden_tokens = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE", "TRUNCATE", "GRANT", "REVOKE"}
    # Tokenize the SQL string nicely using regex to catch whole words only
    tokens = set(re.findall(r'\b[A-Z]+\b', upper_sql))
    
    if tokens.intersection(forbidden_tokens):
        return False
        
    return True


async def fetch_table_schema_info() -> dict[str, dict[str, Any]]:
    """
    Use PRAGMA table_info to build a mapping of table names and columns
    to their real types and nullability rules.
    """
    schema_map = {}
    
    # First get all tables
    tables_result = await execute_query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    
    known_tables = [tr["name"] for tr in tables_result.rows]
    
    for tr in tables_result.rows:
        table_name = tr["name"]
        
        # Validation step
        if table_name not in known_tables:
            continue
        
        # Check table
        pragma_result = await execute_query("PRAGMA table_info(?)", [table_name])
        col_info = {}
        for row in pragma_result.rows:
            col_name = row["name"]
            col_type = row.get("type", "TEXT")
            notnull = row.get("notnull", 0)
            col_info[col_name] = {
                "type": col_type,
                "nullable": notnull == 0
            }
        schema_map[table_name] = col_info
        
    return schema_map


async def execute_nl_query(
    request: QueryRequest,
    sampler_callback: Callable[[str, str], Awaitable[str]],
    memory: ConversationMemory | None = None,
    cache: QueryCache | None = None
) -> QueryResult | QueryError:
    """
    Translates the Request to SQL via the injected sampler, executes it, 
    and returns the strictly typed response.
    """
    try:
        # 1. Fetch schema DDL
        schema_ddl = await get_db_schema()

        # Cache check (before LLM call)
        if cache is not None:
            cached = cache.get(request.nl_query, schema_ddl)
            if cached is not None:
                cached.from_cache = True
                # Still update memory even on cache hit
                if memory:
                    await memory.add_turn("user", request.nl_query, sql="[cache hit]")
                    await memory.add_turn("assistant", f"{cached.row_count} rows returned (cached)", sql=None)
                return cached
        
        # 2. Build Prompts
        history = await memory.get_context() if memory else ""
        system_prompt = get_sql_generation_prompt(schema_ddl, request.max_rows, history)
        user_prompt = build_user_prompt(request.nl_query)
        
        # 3. Call LLM natively via MCP sampling logic
        raw_sql = await sampler_callback(system_prompt, user_prompt)
        
        # 4. Sanitize SQL
        sql = sanitize_sql(raw_sql)
        
        # 5. Validate SQL
        if not validate_sql(sql):
            if memory:
                await memory.add_turn("user", request.nl_query, sql=None)
            return QueryError(
                code="VALIDATION_ERROR",
                message="Invalid SQL query. Only safe SELECT queries are expected.",
                detail=sql
            )
            
        # Error directly from LLM
        if sql.strip() == "SELECT 'ERROR: CANNOT_ANSWER' AS error":
             if memory:
                 await memory.add_turn("user", request.nl_query, sql=None)
             return QueryError(
                 code="LLM_ERROR",
                 message="The LLM could not answer the question based on the schema.",
                 detail="Return statement triggered fail-safe."
             )
        
        # 6. Execute SQL
        import time
        from sqlalchemy.orm import Session
        from querymind.database.history import engine as history_engine
        from querymind.database.history_repo import save_query
        
        start_time = time.time()
        
        if settings.database_type == "postgresql":
            engine = ReadOnlyEngine()
            try:
                raw_rows = await engine.execute(sql)
                cols = list(raw_rows[0].keys()) if raw_rows else []
                result_data = QueryResultData(columns=cols, rows=raw_rows)
            finally:
                await engine.close()
        else:
            result_data = await execute_query(sql)
            
        elapsed = int((time.time() - start_time) * 1000)
        
        with Session(history_engine) as db_session:
            await save_query(
                db=db_session,
                session_id="default",
                user_question=request.nl_query,
                sql_generated=sql,
                result_data=result_data.rows,
                execution_time_ms=elapsed,
                status="success"
            )
        
        # 7. Build Typed QueryResult
        schema_map = await fetch_table_schema_info()
        
        # Flatten column infos across tables for quick lookup.
        all_cols_info = {}
        for table, cols in schema_map.items():
            for col_name, c_info in cols.items():
                if col_name not in all_cols_info:
                    all_cols_info[col_name] = c_info
                    
        column_infos = []
        for col in result_data.columns:
            meta = all_cols_info.get(col, {"type": "TEXT", "nullable": True})
            column_infos.append(
                ColumnInfo(name=col, type=meta["type"], nullable=meta["nullable"])
            )
            
        result = QueryResult(
            columns=column_infos,
            rows=result_data.rows,
            row_count=len(result_data.rows)
        )
        
        async def _generate_insight(res: QueryResult, execution_sql: str) -> QueryInsight | None:
            try:
                user_prompt = build_insight_prompt(
                    nl_query=request.nl_query,
                    sql=execution_sql,
                    row_count=res.row_count,
                    sample_rows=res.rows
                )
                raw = await sampler_callback(INSIGHT_SYSTEM_PROMPT, user_prompt)
                
                # Strip any accidental markdown fences
                clean = raw.strip()
                clean = clean.removeprefix("```json").removeprefix("```")
                clean = clean.removesuffix("```").strip()
                
                import json
                data = json.loads(clean)
                return QueryInsight(**data)
            except Exception:
                return None  # Never crash the main result

        result.insight = await _generate_insight(result, sql)
        
        if cache is not None:
            cache.set(request.nl_query, schema_ddl, result)
        
        if memory:
            await memory.add_turn("user", request.nl_query, sql=sql)
            await memory.add_turn("assistant", f"{result.row_count} rows returned", sql=None)
            
        return result
        
    except Exception as e:
        import time
        from sqlalchemy.orm import Session
        from querymind.database.history import engine as history_engine
        from querymind.database.history_repo import save_query
        
        elapsed = int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
        sql_to_save = sql if 'sql' in locals() else None
        
        try:
            with Session(history_engine) as db_session:
                await save_query(
                    db=db_session,
                    session_id="default",
                    user_question=request.nl_query,
                    sql_generated=sql_to_save,
                    result_data=None,
                    execution_time_ms=elapsed,
                    status="error",
                    error_message=str(e)
                )
        except Exception as db_err:
            print(f"Failed to save query history: {db_err}")

        if memory:
            await memory.add_turn("user", request.nl_query, sql=None)
        return QueryError(
            code="EXECUTION_ERROR",
            message=f"An error occurred while executing the query: {str(e)}",
            detail=None
        )
