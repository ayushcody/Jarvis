from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=False)


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(slots=True)
class Settings:
    livekit_url: str = os.getenv("LIVEKIT_URL", "").strip()
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")
    room_name: str = os.getenv("ROOM_NAME", "demo-room")
    agent_identity: str = os.getenv("AGENT_IDENTITY", "rag-agent")

    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    sarvam_language_code: str = os.getenv("SARVAM_LANGUAGE_CODE", "en-IN")
    sarvam_tts_voice: str = os.getenv("SARVAM_TTS_VOICE", "bulbul:v2")
    sarvam_stt_transport: str = os.getenv("SARVAM_STT_TRANSPORT", "ws").lower()
    sarvam_tts_transport: str = os.getenv("SARVAM_TTS_TRANSPORT", "ws").lower()

    cors_allow_origins: List[str] = field(
        default_factory=lambda: _split_csv(os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:4173"))
    )
    rag_corpus_dir: Path = Path(os.getenv("RAG_CORPUS_DIR", BASE_DIR / "rag" / "corpus_dir"))

    def validate(self) -> None:
        if not self.livekit_url:
            raise ValueError("LIVEKIT_URL is required")
        if not self.livekit_api_key or not self.livekit_api_secret:
            raise ValueError("LIVEKIT API credentials are required")
        if not self.sarvam_api_key:
            raise ValueError("SARVAM_API_KEY is required")


settings = Settings()
