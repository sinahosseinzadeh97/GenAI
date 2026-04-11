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
        
    def add_turn(self, role: Literal["user", "assistant"], content: str, sql: str | None = None) -> None:
        self._turns.append(ConversationTurn(role=role, content=content, sql_generated=sql))
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)
            
    def get_context(self) -> str:
        if not self._turns:
            return ""
        
        lines = []
        for turn in self._turns:
            if turn.role == "user":
                lines.append(f"User: {turn.content}")
            elif turn.role == "assistant":
                lines.append(f"Assistant: {turn.content}")
        return "\n".join(lines)
        
    def get_turns(self) -> list[ConversationTurn]:
        return list(self._turns)
        
    def clear(self) -> None:
        self._turns.clear()
