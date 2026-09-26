"""
Application configuration — all values loaded from .env file.
Never hardcode secrets. Use environment variables or a secrets manager.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    app_name: str = "AI Voice Agent"
    app_version: str = "0.1.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Twilio ────────────────────────────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""          # Your Twilio number in E.164 format
    twilio_webhook_base_url: str = ""      # e.g. https://xxxx.ngrok.io

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://voice:voice@localhost:5432/voice_agent"

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    gemini_api_key: str = ""

    # ── STT & TTS — Groq ──────────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_stt_model: str = "whisper-large-v3-turbo"
    groq_tts_model: str = "canopylabs/orpheus-v1-english"
    groq_tts_voice: str = "" # Groq doesn't use ElevenLabs voice_ids, they might use voice personas but let's keep it simple

    # ── Agent defaults ────────────────────────────────────────────────────────
    agent_silence_timeout_seconds: int = 5
    agent_max_silence_retries: int = 2
    agent_initial_greeting_timeout_seconds: int = 10

    # ── Concurrency ───────────────────────────────────────────────────────────
    default_max_concurrent_calls: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
