import sys
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR: Path = Path(__file__).resolve().parent

INTENT_LABELS: tuple[str, ...] = ("general", "rag")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM provider Config
    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    openrouter_model: str = Field(default="google/gemma-2-9b-it:free")

    # Embeddings Config
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # Intent classification Config
    nli_model: str = Field(default="facebook/bart-large-mnli")
    intent_confidence_threshold: float = Field(
        default=0.55,
        description="Below this, the Coordinator falls back to the General Agent",
    )

    # Database Config
    duckdb_path: Path = Field(default=ROOT_DIR / "data" / "vectors.duckdb")

    # Rerieval Config
    retrieval_top_k: int = Field(default=4)

    # Chunking Config
    chunk_size: int = Field(default=500, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=80)

    # Logging Config
    log_level: Literal["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        "INFO"
    )
    log_dir: Path = Field(default=ROOT_DIR / "logs")

    # API Config
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)


settings = Settings()


def configure_logging() -> None:
    logger.remove()

    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
        ),
        enqueue=False,
    )

    settings.log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        settings.log_dir / "murag_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        enqueue=True,
        serialize=False,
    )

    logger.debug("Logging configured (level={})", settings.log_level)
