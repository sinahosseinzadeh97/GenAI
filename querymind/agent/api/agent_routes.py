"""
Agent API routes.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from querymind.agent.orchestrator import ContractAgent
import time
from querymind.logging_config import StructuredLogger

logger = StructuredLogger("querymind.agent")
from querymind.api.middleware.rate_limit import limiter, AGENT_LIMIT

router = APIRouter(prefix="/agent", tags=["Agent"])

class AgentChatRequest(BaseModel):
    """Request model for agent chat."""
    message: str
    session_id: str = "default"

class AgentChatResponse(BaseModel):
    """Response model for agent chat."""
    answer: str
    tools_used: list[str]
    sources: list[dict]

@router.post("/chat", response_model=AgentChatResponse)
@limiter.limit(AGENT_LIMIT)
async def chat_endpoint(request: Request, req: AgentChatRequest):
    """Endpoint for agent chat."""
    start_time = time.time()
    try:
        agent = ContractAgent()
        result = await agent.run(req.session_id, req.message)
        latency_ms = (time.time() - start_time) * 1000
        logger.log_agent(session_id=req.session_id, tools_used=result["tools_used"], latency_ms=latency_ms)
        return AgentChatResponse(
            answer=result["answer"],
            tools_used=result["tools_used"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.log_error("agent_chat_error", str(e))
        raise HTTPException(status_code=500, detail=str(e))
