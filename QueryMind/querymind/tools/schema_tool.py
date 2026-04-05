"""
MCP tool to expose the database schema.
"""

from querymind.database.router import execute_query

async def get_db_schema() -> str:
    """
    Fetch the complete database schema as raw DDL statements.
    """
    query = "SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    try:
        result = await execute_query(query)
        schema_parts = [row["sql"] for row in result.rows if row.get("sql")]
        return "\n\n".join(schema_parts)
    except Exception as e:
        return f"Error retrieving schema: {str(e)}"
