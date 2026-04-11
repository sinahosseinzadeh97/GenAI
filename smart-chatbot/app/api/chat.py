from fastapi import APIRouter, HTTPException, status
from typing import Optional, List, Dict
from ..models.schemas import ChatRequest, ChatResponse, ChatLog
from ..services.chat_service import chat_service

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="Send a message to the chatbot and receive a response"
)
async def chat(
    request: ChatRequest
) -> ChatResponse:
    """
    Send a chat message to the AI assistant.
    
    - **message**: The user's message
    - **session_id**: Optional session ID for conversation continuity
    - **context**: Optional list of previous messages to provide context
    - **system_prompt**: Optional custom system prompt
    - **temperature**: Response randomness (0-2)
    - **max_tokens**: Maximum response length
    """
    try:
        # پردازش پیام و دریافت پاسخ از سرویس
        result: Dict = await chat_service.process_chat(
            message=request.message,
            session_id=request.session_id,
            context=request.context,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        # بازگرداندن شی ChatResponse با استفاده از unpacking
        return ChatResponse(**result)

    except Exception as e:
        # در صورت خطا، ۵۰۰ برگردون
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat: {e}"
        )

@router.get(
    "/history/{session_id}",
    response_model=ChatLog,
    status_code=status.HTTP_200_OK,
    summary="Get chat history",
    description="Retrieve the chat history for a specific session"
)
async def get_history(
    session_id: str
) -> ChatLog:
    """
    Retrieve chat history for a session.
    
    - **session_id**: The session ID to retrieve history for
    """
    history = await chat_service.get_chat_history(session_id)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat history not found"
        )
    return history
