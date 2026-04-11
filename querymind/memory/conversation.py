import asyncio
from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field

class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sql_generated: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ConversationMemory:
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self._turns: list[ConversationTurn] = []
        self._lock = asyncio.Lock()
        
    async def add_turn(self, role: Literal["user", "assistant"], content: str, sql: str | None = None) -> None:
        async with self._lock:
            self._turns.append(ConversationTurn(role=role, content=content, sql_generated=sql))
            if len(self._turns) > self.max_turns:
                self._turns.pop(0)
            
    async def get_context(self) -> str:
        async with self._lock:
            if not self._turns:
                return ""
            
            lines = []
            for turn in self._turns:
                if turn.role == "user":
                    lines.append(f"User: {turn.content}")
                elif turn.role == "assistant":
                    lines.append(f"Assistant: {turn.content}")
            return "\n".join(lines)
        
    async def get_turns(self) -> list[ConversationTurn]:
        async with self._lock:
            return list(self._turns)
        
    async def clear(self) -> None:
        async with self._lock:
            self._turns.clear()

    async def load_from_db(self, session_id: str, db, last_n: int = 10):
        # Prevent circular imports if used
        from querymind.database.history_repo import get_history
        recent = await get_history(session_id=session_id, limit=last_n)
        for item in reversed(recent):
            await self.add_turn(
                role="user",
                content=item["user_question"],
                sql=item["sql_generated"]
            )
            # The agent typically adds an assistant turn. 
            # We add a matching assistant turn based on the result count so history flows linearly.
            if item.get("result_row_count") is not None:
                await self.add_turn(
                    role="assistant",
                    content=f"{item['result_row_count']} rows returned",
                    sql=None
                )
