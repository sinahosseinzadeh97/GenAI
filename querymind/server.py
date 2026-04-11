import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    SamplingMessage,
)

from querymind.tools.schema_tool import get_db_schema
from querymind.tools.query_tool import execute_nl_query
from querymind.schemas.models import QueryRequest
from querymind.memory.conversation import ConversationMemory
from querymind.cache.redis_cache import RedisQueryCache
from querymind.database.schema_watcher import SchemaWatcher
from querymind.config import settings
import os

# Initialize MCP Server
app = Server("querymind")
_memory = ConversationMemory()
_redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
_cache = RedisQueryCache(redis_url=_redis_url)
_watcher = SchemaWatcher()


@app.list_tools()
async def handle_list_tools() -> list[Tool]:
    """Register and expose tools to the LLM Client."""
    return [
        Tool(
            name="get_db_schema",
            description="Fetch the complete SQLite database schema as DDL statements.",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="query_database",
            description="Translate a natural language question into SQL, execute it, and return structured rows.",
            inputSchema=QueryRequest.model_json_schema()
        ),
        Tool(
            name="clear_conversation_history",
            description="Clear the conversation history and start fresh",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="get_cache_stats",
            description="Get query cache statistics",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="invalidate_cache",
            description="Clear all cached query results",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        ),
        Tool(
            name="get_schema_status",
            description="Get current schema hash and change history",
            inputSchema={
                "type": "object",
                "properties": {},
            }
        )
    ]


@app.call_tool()
async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
    """Dispatch tool execution based on requested name."""
    
    event = await _watcher.check_for_changes()
    if event is not None:
        _cache.invalidate_all()
        _watcher.acknowledge_change()

    if name == "get_db_schema":
        schema_sql = await get_db_schema()
        return [TextContent(type="text", text=schema_sql)]
    
    elif name == "query_database":
        if not arguments:
            raise ValueError("Arguments missing for query_database")
            
        req = QueryRequest(**arguments)
        
        async def mcp_sampler(system_prompt: str, user_prompt: str) -> str:
            """
            Callback that dynamically samples from the connected LLM Host via MCP.
            """
            ctx = app.request_context
            if ctx is None or ctx.session is None:
                raise RuntimeError("MCP session not available for sampling. Ensure host supports sampling.")
                
            msg = SamplingMessage(
                role="user",
                content=TextContent(type="text", text=user_prompt)
            )
            
            result = await ctx.session.create_message(
                messages=[msg],
                max_tokens=1000,
                system_prompt=system_prompt,
            )
            
            if hasattr(result.content, "text"):
                return getattr(result.content, "text")
                
            return "SELECT 'ERROR: CANNOT_ANSWER' AS error"
            
        result_obj = await execute_nl_query(req, sampler_callback=mcp_sampler, memory=_memory, cache=_cache)
        
        return [TextContent(type="text", text=result_obj.model_dump_json(indent=2))]
        
    elif name == "clear_conversation_history":
        await _memory.clear()
        return [TextContent(type="text", text="Conversation history cleared.")]
        
    elif name == "get_cache_stats":
        stats = _cache.stats()
        return [TextContent(type="text", text=stats.model_dump_json(indent=2))]
        
    elif name == "invalidate_cache":
        count = _cache.invalidate_all()
        return [TextContent(type="text", text=f"Cache cleared. {count} entries removed.")]
        
    elif name == "get_schema_status":
        status = _watcher.get_status()
        return [TextContent(type="text", text=status.model_dump_json(indent=2))]
        
    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Start the standard IO server handler natively."""
    async with stdio_server() as (read_stream, write_stream):
        await _watcher.initialize()
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )
        
    if settings.database_type == "postgresql":
        from querymind.database.pg_engine import close_pool
        await close_pool()


if __name__ == "__main__":
    # Provides Graceful Shutdown automatically through context managers in mcp stdio_server
    asyncio.run(main())
