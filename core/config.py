"""
core/config.py
--------------
Centralised application settings loaded from environment variables.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- LLM ---
    groq_api_key: str = Field(..., description="Groq API key")
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    llm_temperature: float = Field(default=0.85, ge=0.0, le=2.0)

    # --- Embeddings ---
    hf_token: str = Field(default="")
    hf_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # --- News ---
    news_api_key: str = Field(default="")

    # --- Router ---
    router_similarity_threshold: float = Field(default=0.18, ge=0.0, le=1.0)

    # --- Security ---
    # Stored as a plain comma-separated string in .env, NOT JSON
    # e.g.  API_KEYS=key1,key2,key3   or leave empty to disable auth
    api_keys_raw: str = Field(default="", alias="api_keys")
    allowed_origins: str = Field(default="http://localhost:8000,http://localhost:3000")
    rate_limit_per_minute: int = Field(default=60)

    # --- Storage ---
    memory_dir: str = Field(default="data/memory")

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    environment: str = Field(default="development")

    @property
    def api_keys(self) -> list[str]:
        """Parse comma-separated API keys string into a list."""
        if not self.api_keys_raw:
            return []
        return [k.strip() for k in self.api_keys_raw.split(",") if k.strip()]

    @property
    def origins_list(self) -> list[str]:
        """Parse comma-separated origins into a list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def auth_enabled(self) -> bool:
        return len(self.api_keys) > 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings: Settings = get_settings()
