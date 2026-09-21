"""Runtime configuration from environment variables (see .env.example)."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o"))
    stt_model: str = field(default_factory=lambda: os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe"))
    tts_model: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"))
    tts_voice: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_VOICE", "nova"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Presentation pacing
    slide_gap_secs: float = field(default_factory=lambda: _env_float("SLIDE_GAP_SECS", 1.2))
    stall_secs: float = field(default_factory=lambda: _env_float("STALL_SECS", 8.0))
    qna_idle_secs: float = field(default_factory=lambda: _env_float("QNA_IDLE_SECS", 25.0))
    max_idle_prompts: int = field(default_factory=lambda: int(_env_float("MAX_IDLE_PROMPTS", 2)))

    # Knowledge retrieval
    knowledge_index: Path = field(default_factory=lambda: Path(os.getenv("KNOWLEDGE_INDEX", "data/knowledge_index.json")))
    retrieval_k: int = field(default_factory=lambda: int(_env_float("RETRIEVAL_K", 3)))
    retrieval_min_score: float = field(default_factory=lambda: _env_float("RETRIEVAL_MIN_SCORE", 0.35))

    # Learning from transcripts
    transcript_dir: Path = field(default_factory=lambda: Path(os.getenv("TRANSCRIPT_DIR", "data/transcripts")))
    learned_guidance_path: Path = field(
        default_factory=lambda: Path(os.getenv("LEARNED_GUIDANCE", "data/learned/presenter_guidance.md"))
    )

    # Safety
    use_moderation_api: bool = field(default_factory=lambda: _env_bool("SAFETY_USE_MODERATION_API", False))

    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env_float("PORT", 7860)))

    def learned_guidance(self) -> str | None:
        if self.learned_guidance_path.exists():
            return self.learned_guidance_path.read_text(encoding="utf-8")
        return None
