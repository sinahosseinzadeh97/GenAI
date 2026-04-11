"""
Agent API routes.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from querymind.agent.orchestrator import ContractAgent

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
async def chat_endpoint(req: AgentChatRequest):
    """Endpoint for agent chat."""
    try:
        agent = ContractAgent()
        result = await agent.run(req.session_id, req.message)
        return AgentChatResponse(
            answer=result["answer"],
            tools_used=result["tools_used"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
