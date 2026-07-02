from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "VectraCore RAG"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api"
    
    # Security: Automatically reads your old "API_KEYS" from .env
    API_KEY: str = Field(default="dev_secret_key", alias="API_KEYS")
    
    # RAG & Model Settings
    EMBEDDING_DIMENSIONS: int = 384
    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8000",  # Python server
        "http://127.0.0.1:8000"
    ]

    # extra="ignore" is the magic line that stops the crashes
    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()