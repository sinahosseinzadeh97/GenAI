"""
Pydantic schemas and models for QueryMind.
All models use strict Pydantic v2 configuration.
"""

from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    """Request from the LLM containing a natural language query."""
    nl_query: str = Field(..., description="The natural language question to translate to SQL.")
    max_rows: int = Field(default=50, description="Maximum number of rows to return.")
    
    model_config = ConfigDict(strict=True)


class SQLQuery(BaseModel):
    """The generated SQL query and its parameters."""
    sql: str = Field(..., description="The executable SQL query.")
    parameters: list[Any] = Field(default_factory=list, description="List of parameters for the SQL query.")
    
    model_config = ConfigDict(strict=True)


class ColumnInfo(BaseModel):
    """Metadata about a single output column."""
    name: str = Field(..., description="The name of the column.")
    type: str = Field(..., description="The data type of the column.")
    nullable: bool = Field(..., description="Whether the column can contain null values.")
    
    model_config = ConfigDict(strict=True)


class QueryInsight(BaseModel):
    """Business insight and explanation of the query results."""
    explanation: str = Field(..., description="A clear explanation of what the query returns.")
    insight: str = Field(..., description="A key takeaway or data point from the results.")
    suggestion: str = Field(..., description="A logical follow-up question for the user.")
    
    model_config = ConfigDict(strict=True)


class QueryResult(BaseModel):
    """The structured result of executing a SQL query."""
    columns: list[ColumnInfo] = Field(..., description="Schema information for the returned columns.")
    rows: list[dict[str, Any]] = Field(..., description="The actual row data as dictionaries.")
    row_count: int = Field(..., description="Total number of rows returned.")
    insight: QueryInsight | None = Field(default=None, description="Generated data insights, if available.")
    from_cache: bool = False
    
    model_config = ConfigDict(strict=True)


class QueryError(BaseModel):
    """Structured error information when a query fails."""
    code: str = Field(..., description="Error code (e.g., 'SQLITE_ERROR', 'VALIDATION_ERROR').")
    message: str = Field(..., description="Human-readable error message.")
    detail: str | None = Field(default=None, description="Detailed error trace or prompt context, if any.")
    
    model_config = ConfigDict(strict=True)


class ImportResult(BaseModel):
    """Result of importing a CSV file into the database."""
    table_name: str = Field(..., description="Name of the table created or modified.")
    rows_imported: int = Field(..., description="Number of rows successfully imported.")
    columns: list[str] = Field(..., description="List of inferred column types or column names.")
    
    model_config = ConfigDict(strict=True)


from datetime import datetime, timezone

class SchemaChangeEvent(BaseModel):
    """Event triggered when a database schema change is detected."""
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="When the change was detected.")
    previous_hash: str = Field(..., description="Hash of the schema before the change.")
    current_hash: str = Field(..., description="Hash of the schema after the change.")
    message: str = Field(default="Schema change detected", description="Message describing the event.")
    
    model_config = ConfigDict(strict=True)

class SchemaStatus(BaseModel):
    """Status of the database schema watcher."""
    current_hash: str = Field(..., description="Current hash of the database schema.")
    last_changed_at: datetime | None = Field(default=None, description="When the schema was last changed.")
    change_count: int = Field(default=0, description="Total number of schema changes detected.")
    is_stale: bool = Field(default=False, description="True if a change was detected and not yet acknowledged.")
    
    model_config = ConfigDict(strict=True)
