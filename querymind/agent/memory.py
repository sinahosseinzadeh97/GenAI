"""
Session memory for the agent.
"""

class SessionMemory:
    """Stores conversation history per session_id."""
    def __init__(self):
        self.sessions: dict[str, list[dict]] = {}
        
    def add_message(self, session_id: str, role: str, content: str | dict | None = None, **kwargs) -> None:
        """Adds a message to the session history."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
            
        message = {"role": role}
        if content is not None:
            message["content"] = content
        message.update(kwargs)
            
        self.sessions[session_id].append(message)
        
        if len(self.sessions[session_id]) > 20:
            self.sessions[session_id] = self.sessions[session_id][-20:]
            
    def get_history(self, session_id: str) -> list[dict]:
        """Gets the history for a session."""
        return self.sessions.get(session_id, [])
        
    def clear(self, session_id: str) -> None:
        """Clears the history for a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
