"""Central configuration, loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- LLM (Gemini) -----------------------------------------------------
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    )
    gemini_base_url: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
        )
    )

    # --- Embeddings (local ONNX MiniLM) ----------------------------------
    embedding_repo: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_REPO", "Xenova/all-MiniLM-L6-v2")
    )
    embedding_onnx_file: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_ONNX_FILE", "onnx/model_quantized.onnx"
        )
    )
    embedding_tokenizer_file: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_TOKENIZER_FILE", "tokenizer.json")
    )
    embedding_max_tokens: int = field(
        default_factory=lambda: _env_int("EMBEDDING_MAX_TOKENS", 256)
    )

    # --- Knowledge base / vector store -----------------------------------
    kb_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("KB_DIR", str(PROJECT_ROOT / "data" / "knowledge_base"))
        )
    )
    chroma_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "data" / "chroma"))
        )
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "career_knowledge")
    )

    # --- Retrieval --------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))
    candidate_k: int = field(default_factory=lambda: _env_int("CANDIDATE_K", 12))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 220))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 40))

    # --- Conversation -----------------------------------------------------
    max_history_turns: int = field(
        default_factory=lambda: _env_int("MAX_HISTORY_TURNS", 6)
    )
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.6))
    max_output_tokens: int = field(
        default_factory=lambda: _env_int("MAX_OUTPUT_TOKENS", 512)
    )

    # --- Server -----------------------------------------------------------
    build_index_on_start: bool = field(
        default_factory=lambda: os.getenv("BUILD_INDEX_ON_START", "true").lower()
        in {"1", "true", "yes"}
    )
    warmup_on_start: bool = field(
        default_factory=lambda: os.getenv("WARMUP_ON_START", "true").lower()
        in {"1", "true", "yes"}
    )

    @property
    def has_llm_key(self) -> bool:
        return bool(self.gemini_api_key.strip())


settings = Settings()
