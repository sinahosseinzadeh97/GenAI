import pytest
from unittest.mock import patch, AsyncMock
from querymind.config import settings
from querymind.database import router
from querymind.database.pg_engine import _convert_placeholders

@pytest.mark.asyncio
async def test_router_uses_sqlite_by_default():
    assert settings.database_type == "sqlite"
    # Execute a simple query
    result = await router.execute_query("SELECT 1 AS num")
    assert result.columns == ["num"]
    assert len(result.rows) == 1
    assert result.rows[0]["num"] == 1

@pytest.mark.asyncio
async def test_router_raises_without_postgres_dsn():
    original_type = settings.database_type
    original_dsn = settings.postgres_dsn
    
    settings.database_type = "postgresql"
    settings.postgres_dsn = None
    
    try:
        with pytest.raises(RuntimeError, match="POSTGRES_DSN"):
            await router.execute_query("SELECT 1")
    finally:
        settings.database_type = original_type
        settings.postgres_dsn = original_dsn

def test_placeholder_conversion():
    assert _convert_placeholders("SELECT * FROM t WHERE id=?") == "SELECT * FROM t WHERE id=$1"
    assert _convert_placeholders("WHERE a=? AND b=?") == "WHERE a=$1 AND b=$2"
    assert _convert_placeholders("SELECT 1") == "SELECT 1"
