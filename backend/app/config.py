from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / "backend" / ".env")



@dataclass(frozen=True)
class Settings:
    app_name: str = "Aurag"
    data_dir: Path = ROOT_DIR / "backend" / "data"
    uploads_dir: Path = ROOT_DIR / "backend" / "data" / "uploads"
    db_path: Path = ROOT_DIR / "backend" / "data" / "aurag.sqlite3"
    whisper_model: str = os.getenv("AURAG_WHISPER_MODEL", "small.en")
    whisper_device: str = os.getenv("AURAG_WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("AURAG_WHISPER_COMPUTE_TYPE", "int8")
    embed_model: str = os.getenv("AURAG_EMBED_MODEL", "all-MiniLM-L6-v2")
    llm_model_path: str = os.getenv("AURAG_LLM_MODEL_PATH", "")
    llm_n_ctx: int = int(os.getenv("AURAG_LLM_N_CTX", "2048"))
    llm_max_tokens: int = int(os.getenv("AURAG_LLM_MAX_TOKENS", "256"))
    llm_n_gpu_layers: int = int(os.getenv("AURAG_LLM_N_GPU_LAYERS", "32"))
    max_upload_mb: int = int(os.getenv("AURAG_MAX_UPLOAD_MB", "50"))
    wer_gate: float = float(os.getenv("AURAG_WER_GATE", "0.20"))
    chunk_words: int = int(os.getenv("AURAG_CHUNK_WORDS", "120"))
    chunk_overlap: int = int(os.getenv("AURAG_CHUNK_OVERLAP", "24"))
    top_k: int = int(os.getenv("AURAG_TOP_K", "5"))
    agent_threshold: float = float(os.getenv("AURAG_AGENT_THRESHOLD", "0.35"))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.uploads_dir.mkdir(parents=True, exist_ok=True)
