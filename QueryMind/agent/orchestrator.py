"""
Orchestrator for the Contract Agent.
"""
import os
import json
import logging
from openai import OpenAI

from querymind.agent.memory import SessionMemory
from querymind.agent.tools import TOOLS
from querymind.rag.retrieval.search import search_chunks
from querymind.rag.generation.llm_client import generate_answer

logger = logging.getLogger(__name__)

# Global memory instance
_memory = SessionMemory()

class ContractAgent:
    """Agent orchestrator for handling RAG queries with tools."""
    
    def __init__(self):
        """Initializes the ContractAgent."""
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        
    async def run(self, session_id: str, user_message: str) -> dict:
        """
        Runs a user message through the agent.
        
        Args:
            session_id: The session ID.
            user_message: The user's input message.
            
        Returns:
            A dictionary containing the answer, tools used, and sources.
        """
        # 1. Load history from SessionMemory
        # Load history creates a copy or we can just get the list
        history = list(_memory.get_history(session_id))
        
        # 2. Append user message to history
        _memory.add_message(session_id, "user", user_message)
        messages = list(_memory.get_history(session_id))
        
        tools_used = []
        sources = []
        
        # 3. Call OpenAI chat completions with tools
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOLS
        )
        
        message = response.choices[0].message
        
        # 4. If LLM chooses a tool
        if message.tool_calls:
            # c. Append assistant message with tool calls to memory
            _memory.add_message(
                session_id, 
                role=message.role, 
                content=message.content, 
                tool_calls=[t.model_dump() for t in message.tool_calls]
            )
            
            for tool_call in message.tool_calls:
                tools_used.append(tool_call.function.name)
                args = json.loads(tool_call.function.arguments)
                tool_result_str = ""
                
                try:
                    # b. Call the matching function
                    if tool_call.function.name == "search_contracts":
                        results = await search_chunks(args["query"], args.get("top_k", 5))
                        context = [f"[Source: {r['filename']} (Page {r['page_number']})]\n{r['content']}" for r in results]
                        tool_result_str = "\n\n".join(context)
                        for r in results:
                            sources.append({"filename": r["filename"], "page_number": r["page_number"]})
                            
                    elif tool_call.function.name == "extract_field":
                        results = await search_chunks(args["query"] + f" {args['field']}", 5)
                        answer = generate_answer(f"Extract the {args['field']} based on the query: {args['query']}", results)
                        tool_result_str = answer
                        for r in results:
                            sources.append({"filename": r["filename"], "page_number": r["page_number"]})
                            
                    elif tool_call.function.name == "compare_contracts":
                        results = await search_chunks(args["query"], 10)
                        if "filenames" in args and args["filenames"]:
                            results = [r for r in results if r["filename"] in args["filenames"]]
                        answer = generate_answer(args["query"], results)
                        tool_result_str = answer
                        for r in results:
                            sources.append({"filename": r["filename"], "page_number": r["page_number"]})
                    else:
                        tool_result_str = f"Unknown tool: {tool_call.function.name}"
                except Exception as e:
                    logger.error(f"Error executing tool {tool_call.function.name}: {e}")
                    tool_result_str = f"Error: {e}"
                
                # c. Append tool result to messages
                _memory.add_message(
                    session_id, 
                    role="tool", 
                    content=tool_result_str, 
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name
                )
            
            # d. Call OpenAI again to get final answer
            messages = list(_memory.get_history(session_id))
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            final_answer = final_response.choices[0].message.content
            # 6. Append assistant response to history
            _memory.add_message(session_id, "assistant", final_answer)
            ans = final_answer
        else:
            # 5. If LLM answers directly: use that answer
            ans = message.content
            # 6. Append assistant response to history
            _memory.add_message(session_id, "assistant", ans)
            
        unique_sources = []
        seen = set()
        for s in sources:
            key = (s["filename"], s["page_number"])
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)
                
        # 7. Return expected dictionary
        return {
            "answer": ans or "",
            "tools_used": tools_used,
            "sources": unique_sources
        }
