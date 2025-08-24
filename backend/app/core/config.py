from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
    DATABASE_URL: str = Field(..., description="SQLAlchemy URL")
    VECTORSTORE_DIR: str = "/app/storage/vectorstore"
    FILES_DIR: str = "/app/storage/files"
    USE_OPENAI: bool = False
    OPENAI_API_KEY: str | None = None
    MODEL_NAME: str = "gpt-4o-mini"
    N8N_WEBHOOK_URL: str | None = None

    class Config:
        env_file = "/app/.env"
        extra = "ignore"

settings = Settings()
Path(settings.VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.FILES_DIR).mkdir(parents=True, exist_ok=True)