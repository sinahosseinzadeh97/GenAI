import pytest
import asyncio
import os

from querymind.tools.query_tool import sanitize_sql, validate_sql, execute_nl_query
from querymind.schemas.models import QueryRequest, QueryResult, QueryError
from querymind.database.seed import seed_database
from querymind.config import settings

@pytest.fixture(autouse=True)
async def setup_db():
    # Point out settings to a temporary test db to avoid mutating the dev one
    old_db_path = settings.db_path
    settings.db_path = "test_querymind.db"
    
    # Ensure a fresh state
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
        
    await seed_database()
    
    yield
    
    # Cleanup after test finishes
    if os.path.exists(settings.db_path):
        os.remove(settings.db_path)
    settings.db_path = old_db_path

def test_sanitize_sql():
    # Test Markdown unwrapping
    assert sanitize_sql("```sql\nSELECT * FROM users;\n```") == "SELECT * FROM users;"
    assert sanitize_sql("```\nSELECT * FROM users\n```") == "SELECT * FROM users"
    
    # Test trailing space trimming
    assert sanitize_sql("  SELECT * FROM users;  ") == "SELECT * FROM users;"
    
    # Test plain SQL untouched
    assert sanitize_sql("SELECT name FROM products") == "SELECT name FROM products"

def test_validate_sql():
    # Valid
    assert validate_sql("SELECT * FROM users") is True
    assert validate_sql("SELECT id, name FROM products WHERE id = 1") is True
    
    # Invalid non-SELECT
    assert validate_sql("INSERT INTO users (name) VALUES ('Test')") is False
    assert validate_sql("UPDATE users SET name='Test'") is False
    assert validate_sql("DROP TABLE users") is False
    
    # Invalid injection token tests
    assert validate_sql("SELECT * FROM users; DROP TABLE users;") is False

@pytest.mark.asyncio
async def test_execute_nl_query_success():
    req = QueryRequest(nl_query="Get all users")
    
    # Mock Sampler properly returning a valid query
    async def mock_sampler(system: str, user: str) -> str:
        return "SELECT * FROM users;"
        
    res = await execute_nl_query(req, sampler_callback=mock_sampler)
    
    assert isinstance(res, QueryResult)
    assert res.row_count == 3
    assert len(res.rows) == 3
    
    # Check schema column presence
    col_names = [c.name for c in res.columns]
    assert "id" in col_names
    assert "name" in col_names

@pytest.mark.asyncio
async def test_execute_nl_query_failure_validation():
    req = QueryRequest(nl_query="Drop users table")
    
    # Mock Sampler properly returning a destructive query
    async def mock_sampler(system: str, user: str) -> str:
        return "DROP TABLE users;"
        
    res = await execute_nl_query(req, sampler_callback=mock_sampler)
    
    assert isinstance(res, QueryError)
    assert res.code == "VALIDATION_ERROR"
    
@pytest.mark.asyncio
async def test_execute_nl_query_llm_error():
    req = QueryRequest(nl_query="What's the weather?")
    
    # Mock Sampler returning fail-safe marker
    async def mock_sampler(system: str, user: str) -> str:
        return "SELECT 'ERROR: CANNOT_ANSWER' AS error"
        
    res = await execute_nl_query(req, sampler_callback=mock_sampler)
    
    assert isinstance(res, QueryError)
    assert res.code == "LLM_ERROR"
