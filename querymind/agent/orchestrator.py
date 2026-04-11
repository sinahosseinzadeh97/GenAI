"""
Orchestrator for the Contract Agent.
"""
import os
import json
import logging
import anthropic

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
        self.client = anthropic.Anthropic()
        self.model = "claude-opus-4-5"
        self.system_prompt = "You are a helpful data analyst agent. You analyze contracts and answer queries accurately."
        
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
        history = list(_memory.get_history(session_id))
        
        # 2. Append user message to history
        _memory.add_message(session_id, "user", user_message)
        messages = list(_memory.get_history(session_id))
        
        tools_used = []
        sources = []
        
        # 3. Call Anthropic with tools
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=self.system_prompt,
            messages=messages,
            tools=TOOLS
        )
        
        # 4. If LLM chooses a tool
        if response.stop_reason == "tool_use":
            # c. Append assistant message with tool calls to memory
            _memory.add_message(
                session_id, 
                role=response.role, 
                content=[block.model_dump() for block in response.content]
            )
            
            tool_results = []
            
            for block in response.content:
                if block.type == "tool_use":
                    tools_used.append(block.name)
                    args = block.input
                    tool_result_str = ""
                    
                    try:
                        # b. Call the matching function
                        if block.name == "search_contracts":
                            results = await search_chunks(args["query"], args.get("top_k", 5))
                            context = [f"[Source: {r['filename']} (Page {r['page_number']})]\n{r['content']}" for r in results]
                            tool_result_str = "\n\n".join(context)
                            for r in results:
                                sources.append({"filename": r["filename"], "page_number": r["page_number"]})
                                
                        elif block.name == "extract_field":
                            results = await search_chunks(args["query"] + f" {args['field']}", 5)
                            answer = generate_answer(f"Extract the {args['field']} based on the query: {args['query']}", results)
                            tool_result_str = answer
                            for r in results:
                                sources.append({"filename": r["filename"], "page_number": r["page_number"]})
                                
                        elif block.name == "compare_contracts":
                            results = await search_chunks(args["query"], 10)
                            if "filenames" in args and args.get("filenames"):
                                results = [r for r in results if r["filename"] in args["filenames"]]
                            answer = generate_answer(args["query"], results)
                            tool_result_str = answer
                            for r in results:
                                sources.append({"filename": r["filename"], "page_number": r["page_number"]})
                        else:
                            tool_result_str = f"Unknown tool: {block.name}"
                    except Exception as e:
                        logger.error(f"Error executing tool {block.name}: {e}")
                        tool_result_str = f"Error: {e}"
                    
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result_str
                    })
            
            # c. Append tool results to messages
            _memory.add_message(
                session_id, 
                role="user", 
                content=tool_results
            )
            
            # d. Call Anthropic again to get final answer
            messages = list(_memory.get_history(session_id))
            final_response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=self.system_prompt,
                messages=messages,
                tools=TOOLS
            )
            final_answer = next((block.text for block in final_response.content if block.type == 'text'), "")
            # 6. Append assistant response to history
            _memory.add_message(session_id, "assistant", final_answer)
            ans = final_answer
        else:
            # 5. If LLM answers directly: use that answer
            ans = next((block.text for block in response.content if block.type == 'text'), "")
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
