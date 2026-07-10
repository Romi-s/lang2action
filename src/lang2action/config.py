"""Runtime settings, read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass

DEFAULT_MODEL = "openai:gpt-4o-mini"
DEFAULT_PERCEPTION = "sim"  # "sim" | "sgg"
DEFAULT_SGG_URL = "http://localhost:8000"


@dataclass(frozen=True)
class Settings:
    model: str
    perception_backend: str
    sgg_url: str


def load_settings() -> Settings:
    return Settings(
        model=os.getenv("LANG2ACTION_MODEL", DEFAULT_MODEL),
        perception_backend=os.getenv("LANG2ACTION_PERCEPTION", DEFAULT_PERCEPTION),
        sgg_url=os.getenv("LANG2ACTION_SGG_URL", DEFAULT_SGG_URL),
    )
