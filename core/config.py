"""
core/config.py
--------------
Centralised application settings loaded from environment variables.

All secrets and tuneable parameters live here.  Modules import ``settings``
rather than calling ``os.environ.get`` directly, which gives us:

- A single place to audit what the app reads from the environment.
- Validation at startup (missing required keys raise an error immediately).
- Easy overrides in tests via environment patches.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from ``.env`` or the shell."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- LLM ---
    groq_api_key: str = Field(..., description="Groq API key for LLM inference")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model identifier",
    )
    llm_temperature: float = Field(default=0.85, ge=0.0, le=2.0)

    # --- Embeddings ---
    hf_token: str = Field(default="", description="HuggingFace token (optional for public models)")
    hf_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model path",
    )

    # --- News ---
    news_api_key: str = Field(default="", description="NewsAPI key (optional, falls back to mock)")

    # --- Router ---
    router_similarity_threshold: float = Field(
        default=0.18,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for bot matching",
    )

    # --- Security ---
    api_keys: list[str] = Field(
        default=[],
        description="Comma-separated list of valid API keys (empty = auth disabled)",
    )
    allowed_origins: list[str] = Field(
        default=["http://localhost:8000", "http://localhost:3000"],
        description="CORS allowed origins",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Max requests per client IP per minute",
    )

    # --- Storage ---
    memory_dir: str = Field(default="data/memory", description="Directory for bot memory pickles")

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    environment: str = Field(default="development", description="development | staging | production")

    @property
    def is_production(self) -> bool:
        """True when running in a production environment."""
        return self.environment.lower() == "production"

    @property
    def auth_enabled(self) -> bool:
        """True when at least one API key is configured."""
        return len(self.api_keys) > 0


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton."""
    return Settings()


settings: Settings = get_settings()
