import uuid
from datetime import datetime
from typing import List, Dict, Optional
from ..core.openai_client import openai_client
from ..db.mongodb import get_database
from ..models.schemas import ChatLog, Message, Role

class ChatService:
    def __init__(self):
        self.collection_name = "chat_logs"
    
    async def process_chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """Process chat message and return response"""
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Prepare messages for OpenAI
        messages = await self._prepare_messages(
            message, session_id, context, system_prompt
        )
        
        # Get response from OpenAI
        response_data = await openai_client.generate_response(
            messages, temperature, max_tokens
        )
        
        # Save to database
        await self._save_chat_log(
            session_id, message, response_data["content"], messages
        )
        
        return {
            "response": response_data["content"],
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "usage": response_data["usage"]
        }
    
    async def _prepare_messages(
        self,
        message: str,
        session_id: str,
        context: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        """Prepare messages with context and history"""
        
        messages = []
        
        # Add custom system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Get chat history from database
        db = get_database()
        chat_log = await db[self.collection_name].find_one(
            {"session_id": session_id}
        )
        
        if chat_log:
            # Add previous messages for context
            for msg in chat_log.get("messages", [])[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add provided context
        if context:
            messages.extend(context)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        return messages
    
    async def _save_chat_log(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        full_context: List[Dict[str, str]]
    ):
        """Save chat log to MongoDB"""
        
        db = get_database()
        collection = db[self.collection_name]
        
        # Create new messages
        new_messages = [
            Message(role=Role.user, content=user_message),
            Message(role=Role.assistant, content=assistant_response)
        ]
        
        # Update or create chat log
        await collection.update_one(
            {"session_id": session_id},
            {
                "$push": {
                    "messages": {
                        "$each": [msg.dict() for msg in new_messages]
                    }
                },
                "$set": {
                    "updated_at": datetime.utcnow()
                },
                "$setOnInsert": {
                    "created_at": datetime.utcnow(),
                    "session_id": session_id
                }
            },
            upsert=True
        )
    
    async def get_chat_history(self, session_id: str) -> Optional[ChatLog]:
        """Retrieve chat history for a session"""
        
        db = get_database()
        chat_log = await db[self.collection_name].find_one(
            {"session_id": session_id}
        )
        
        if chat_log:
            chat_log.pop("_id", None)
            return ChatLog(**chat_log)
        
        return None

chat_service = ChatService()