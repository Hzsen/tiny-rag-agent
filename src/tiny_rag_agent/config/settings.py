"""Application settings and model paths."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Typed settings loaded from environment variables."""

    mlx_model_dir: Path = Field(..., description="Root directory for MLX models.")
    qwen_model_path: Path = Field(..., description="Path to Qwen model.")
    deepseek_model_path: Path = Field(..., description="Path to DeepSeek model.")
    embedding_model_path: Path = Field(..., description="Path to embedding model.")
    data_dir: Path = Field(..., description="Data directory for local assets.")
    chroma_persist_dir: Path = Field(..., description="Chroma persistence directory.")
    log_level: str = Field(default="INFO", description="Logging level.")

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        """Load settings from environment variables and optional .env file."""

        load_dotenv(dotenv_path=env_file, override=False)
        return cls(
            mlx_model_dir=Path(_require_env("MLX_MODEL_DIR")),
            qwen_model_path=Path(_require_env("QWEN_MODEL_PATH")),
            deepseek_model_path=Path(_require_env("DEEPSEEK_MODEL_PATH")),
            embedding_model_path=Path(_require_env("EMBEDDING_MODEL_PATH")),
            data_dir=Path(_require_env("DATA_DIR")),
            chroma_persist_dir=Path(_require_env("CHROMA_PERSIST_DIR")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


def _require_env(key: str) -> str:
    """Return an environment variable or raise a clear error."""

    value = os.getenv(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value
