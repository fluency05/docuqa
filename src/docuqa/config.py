"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Config:
    """Runtime configuration for docuqa.

    Values are read from environment variables (or a local ``.env`` file) so the
    same settings can be shared between the CLI, the web UI, and any embedding
    application. ``embedder`` and ``llm`` select the backend.
    """

    index_dir: Path = field(default_factory=lambda: Path(".docuqa"))
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    # Backend selection: embedder in {openai, local, hashing}; llm in {openai, ollama, mock}.
    embedder: str = "openai"
    llm: str = "openai"
    # OpenAI backend.
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    api_key: str | None = None
    base_url: str | None = None
    # Local backend.
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    @classmethod
    def from_env(cls) -> "Config":
        """Build a :class:`Config` from environment variables / ``.env``."""
        load_dotenv()
        return cls(
            index_dir=Path(os.getenv("DOCUQA_INDEX_DIR", ".docuqa")),
            chunk_size=int(os.getenv("DOCUQA_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("DOCUQA_CHUNK_OVERLAP", "50")),
            top_k=int(os.getenv("DOCUQA_TOP_K", "4")),
            embedder=os.getenv("DOCUQA_EMBEDDER", "openai"),
            llm=os.getenv("DOCUQA_LLM", "openai"),
            llm_model=os.getenv("DOCUQA_MODEL", "gpt-4o-mini"),
            embedding_model=os.getenv("DOCUQA_EMBEDDING_MODEL", "text-embedding-3-small"),
            local_embedding_model=os.getenv(
                "DOCUQA_LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
            ),
            ollama_model=os.getenv("DOCUQA_OLLAMA_MODEL", "llama3.2"),
            ollama_base_url=os.getenv("DOCUQA_OLLAMA_BASE_URL", "http://localhost:11434"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
