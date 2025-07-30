from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    api_version: str = "v1"
    project_name: str = "Smart Chatbot API"
    
    # OpenAI Settings
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # MongoDB Settings
    mongodb_url: str
    database_name: str
    
    # CORS Settings
    cors_origins: list = ["*"]
    
    class Config:
        env_file = ".env"

settings = Settings()