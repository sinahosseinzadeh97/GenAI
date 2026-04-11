import pytest
from pydantic import ValidationError
from querymind.prompts.insight_generator import build_insight_prompt
from querymind.schemas.models import QueryInsight, QueryResult, QueryRequest
from querymind.tools.query_tool import execute_nl_query
from querymind.cache.query_cache import QueryCache
import json

def test_build_insight_prompt_contains_query():
    nl_query = "What is the total revenue?"
    sql = "SELECT sum(revenue) FROM sales"
    row_count = 3
    sample_rows = [{"total": 100}, {"total": 200}, {"total": 300}]
    
    prompt = build_insight_prompt(nl_query, sql, row_count, sample_rows)
    
    assert nl_query in prompt
    assert sql in prompt
    assert "3" in prompt

def test_query_insight_model_valid():
    insight = QueryInsight(
        explanation="Shows total revenue",
        insight="Revenue is high",
        suggestion="What is revenue by region?"
    )
    assert insight.explanation == "Shows total revenue"
    assert insight.insight == "Revenue is high"
    assert insight.suggestion == "What is revenue by region?"

def test_query_insight_model_invalid():
    with pytest.raises(ValidationError):
        QueryInsight(explanation="x")

@pytest.mark.asyncio
async def test_execute_nl_query_returns_insight():
    request = QueryRequest(nl_query="test?", max_rows=10)
    
    call_count = 0
    async def mock_sampler(system_prompt: str, user_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "SELECT 1 as val"
        else:
            return '{"explanation":"x","insight":"y","suggestion":"z"}'
            
    result = await execute_nl_query(request, mock_sampler)
    
    assert isinstance(result, QueryResult)
    assert result.insight is not None
    assert result.insight.explanation == "x"
    assert result.insight.insight == "y"
    assert result.insight.suggestion == "z"

@pytest.mark.asyncio
async def test_execute_nl_query_insight_failure_safe():
    request = QueryRequest(nl_query="test?", max_rows=10)
    
    call_count = 0
    async def mock_sampler(system_prompt: str, user_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "SELECT 1 as val"
        else:
            raise ValueError("LLM fail")
            
    result = await execute_nl_query(request, mock_sampler)
    
    assert isinstance(result, QueryResult)
    assert result.insight is None

@pytest.mark.asyncio
async def test_execute_nl_query_no_insight_on_cache_hit():
    request = QueryRequest(nl_query="test_cache?", max_rows=10)
    
    call_count = 0
    async def mock_sampler(system_prompt: str, user_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "SELECT 1 as val"
        else:
            return '{"explanation":"x","insight":"y","suggestion":"z"}'
            
    cache = QueryCache(ttl_seconds=60)
    
    # First call
    result1 = await execute_nl_query(request, mock_sampler, cache=cache)
    assert result1.insight is not None
    assert call_count == 2
    
    # Second call (cache hit)
    result2 = await execute_nl_query(request, mock_sampler, cache=cache)
    assert result2.from_cache is True
    # Still 2, sampler was not called for insight again
    assert call_count == 2
