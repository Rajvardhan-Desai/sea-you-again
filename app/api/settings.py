"""
app/api/settings.py
-------------------
Pydantic-settings config — reads from environment / .env file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/mmaras"

    # Redis (job dispatch from api -> scheduler/worker)
    redis_url: str = "redis://redis:6379/0"

    # Auth
    admin_token: str = "changeme"
    secret_key:  str = "changeme-secret-for-itsdangerous"

    # Notifications — email
    smtp_host: str  = "smtp.sendgrid.net"
    smtp_port: int  = 587
    smtp_user: str  = "apikey"
    smtp_pass: str  = ""
    smtp_from: str  = "noreply@mmaras.example.com"

    # Notifications — SMS (Twilio)
    twilio_sid:   str = ""
    twilio_token: str = ""
    twilio_from:  str = ""

    # Public base URL (used in emails/SMS for links)
    public_base_url: str = "http://localhost"

    # Inference
    checkpoint_path: str = str(Path(__file__).resolve().parent.parent.parent / "model" / "checkpoints" / "best.pt")

    # Data storage root
    data_dir: str = "/data"

    # Application
    version: str = "3.6.0"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
