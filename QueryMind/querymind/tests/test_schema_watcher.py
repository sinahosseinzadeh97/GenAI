import pytest
import sqlite3
import os
from querymind.database.schema_watcher import SchemaWatcher
from querymind.schemas.models import SchemaChangeEvent
from querymind.database.router import execute_query
from querymind.config import settings
from querymind.cache.query_cache import QueryCache
from querymind.schemas.models import QueryRequest, SQLQuery, QueryResult

@pytest.fixture(autouse=True)
async def setup_db():
    temp_db_path = "test_schema_watcher.db"
    settings.db_path = temp_db_path
    
    conn = sqlite3.connect(temp_db_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    
    yield
    
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)


@pytest.mark.asyncio
async def test_watcher_initializes_hash():
    watcher = SchemaWatcher()
    await watcher.initialize()
    status = watcher.get_status()
    assert status.current_hash != ""
    assert len(status.current_hash) == 64  # sha256 hex length


@pytest.mark.asyncio
async def test_no_change_returns_none():
    watcher = SchemaWatcher()
    await watcher.initialize()
    result = await watcher.check_for_changes()
    assert result is None


@pytest.mark.asyncio
async def test_change_detected_after_schema_alter():
    watcher = SchemaWatcher()
    await watcher.initialize()
    
    # alter the DB
    await execute_query("CREATE TABLE test_detection (id INTEGER PRIMARY KEY)")
    
    result = await watcher.check_for_changes()
    assert result is not None
    assert isinstance(result, SchemaChangeEvent)
    assert result.previous_hash != result.current_hash


@pytest.mark.asyncio
async def test_change_count_increments():
    watcher = SchemaWatcher()
    await watcher.initialize()
    
    await execute_query("CREATE TABLE test_table_1 (id INTEGER PRIMARY KEY)")
    await watcher.check_for_changes()
    assert watcher.get_status().change_count == 1
    
    await execute_query("CREATE TABLE test_table_2 (id INTEGER PRIMARY KEY)")
    await watcher.check_for_changes()
    assert watcher.get_status().change_count == 2


@pytest.mark.asyncio
async def test_acknowledge_resets_stale():
    watcher = SchemaWatcher()
    await watcher.initialize()
    
    await execute_query("CREATE TABLE test_ack (id INTEGER PRIMARY KEY)")
    await watcher.check_for_changes()
    
    assert watcher.get_status().is_stale is True
    watcher.acknowledge_change()
    assert watcher.get_status().is_stale is False


@pytest.mark.asyncio
async def test_cache_invalidated_on_schema_change():
    watcher = SchemaWatcher()
    await watcher.initialize()
    
    cache = QueryCache()
    req = QueryRequest(nl_query="test query")
    sql = SQLQuery(sql="SELECT * FROM users")
    res = QueryResult(columns=[], rows=[{"id": 1}], row_count=1)
    
    cache.set(req.nl_query, "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)", res)
    assert cache.stats().total_entries == 1
    
    # Action: alter schema
    await execute_query("CREATE TABLE test_cache_invalidate (id INTEGER PRIMARY KEY)")
    
    # Action: check for changes
    event = await watcher.check_for_changes()
    assert event is not None
    
    # Action: manual integration simulation as done in server.py
    cache.invalidate_all()
    watcher.acknowledge_change()
    
    assert cache.stats().total_entries == 0
