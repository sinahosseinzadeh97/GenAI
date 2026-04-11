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
    anthropic_api_key: str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
