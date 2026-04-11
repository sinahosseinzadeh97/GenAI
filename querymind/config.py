"""
Configuration management for QueryMind.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Pydantic settings loaded from environment variables and .env file.
    """
    db_path: str = "querymind.db"
    environment: str = "development"
    cache_ttl_seconds: int = 300
    database_type: Literal["sqlite", "postgresql"] = "sqlite"
    postgres_dsn: str | None = None
    
    querymind_api_key: str  # required
    
    # Anthropic — required for SQL, agent, insights
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-5"
    
    # OpenAI — required ONLY for embeddings
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def validate_settings(settings: Settings):
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY is required for agent and SQL generation")
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for RAG embeddings")

settings = Settings()
validate_settings(settings)
