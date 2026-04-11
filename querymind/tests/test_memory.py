import pytest
from querymind.memory.conversation import ConversationMemory
from querymind.schemas.models import QueryRequest
from querymind.tools.query_tool import execute_nl_query
from querymind.agent.redis_memory import RedisSessionMemory

@pytest.mark.asyncio
async def test_add_and_get_turns():
    memory = ConversationMemory()
    await memory.add_turn("user", "Hello")
    await memory.add_turn("assistant", "Hi there")
    await memory.add_turn("user", "SQL?")
    assert len(await memory.get_turns()) == 3

@pytest.mark.asyncio
async def test_max_turns_enforced():
    memory = ConversationMemory(max_turns=3)
    for i in range(1, 6):
        await memory.add_turn("user" if i % 2 != 0 else "assistant", str(i))
        
    turns = await memory.get_turns()
    assert len(turns) == 3
    assert turns[0].content == "3"
    assert turns[1].content == "4"
    assert turns[2].content == "5"

@pytest.mark.asyncio
async def test_get_context_empty():
    memory = ConversationMemory()
    assert await memory.get_context() == ""

@pytest.mark.asyncio
async def test_get_context_formatted():
    memory = ConversationMemory()
    await memory.add_turn("user", "Show users")
    await memory.add_turn("assistant", "10 rows returned")
    context = await memory.get_context()
    assert "User: Show users" in context
    assert "Assistant: 10 rows returned" in context

@pytest.mark.asyncio
async def test_clear():
    memory = ConversationMemory()
    await memory.add_turn("user", "Hi")
    await memory.clear()
    assert await memory.get_turns() == []

@pytest.mark.asyncio
async def test_memory_injected_into_query():
    memory = ConversationMemory()
    
    captured_prompts = []
    
    async def mock_sampler(system_prompt: str, user_prompt: str) -> str:
        captured_prompts.append(system_prompt)
        return "SELECT 1 as num;"
        
    req1 = QueryRequest(nl_query="First question")
    await execute_nl_query(req1, mock_sampler, memory)
    
    req2 = QueryRequest(nl_query="Second question")
    await execute_nl_query(req2, mock_sampler, memory)
    
    assert len(captured_prompts) == 4
    # Second query's SQL prompt should have the first query in history
    assert "First question" in captured_prompts[2]
