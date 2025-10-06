from typing import Optional

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    # LLM settings
    model_name: str = Field(default="llama3.1")
    openai_api_key: str | None = "dummy"
    # defaul ollama server url
    openai_base_url: str | None = "http://localhost:11434/v1"

    # Qdrant
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    qdrant_in_memory: bool = True
    ##TODO: change this to something else
    collection_name: str = "react_rag_docs"

    # Search / ingest
    ddg_region: str = "wt-wt"
    ddg_safesearch: str = "moderate"
    polite_delay_sec: float = 0.35
    chunk_max_chars: int = 1200
    max_search_results: int = 6
    max_reddit_results: int = 6

    # RAG
    top_k: int = 6

    # Debate
    rounds: int = 2

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
logger.debug(
    "Settings loaded: model={m} qdrant_url={q}", m=settings.model_name, q=settings.qdrant_url
)
