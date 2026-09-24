"""Configuration. Every setting comes from environment variables (or a .env file).

The API key is never hard-coded, printed or logged.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")  # does not override variables already set in the shell

UPLOADS_DIR = PROJECT_DIR / "uploads"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

MAX_STEPS = 8                # hard limit on tool rounds in Agent mode
DEFAULT_MAX_PAPERS = 5


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)            # repr=False: never shows up in logs/tracebacks
    model: str = "muse-spark-1.3"
    base_url: str = "https://api.meta.ai/v1"
    reasoning_effort: str = "low"               # minimal | low | medium | high | xhigh | max
    force_mock: bool = False
    semantic_scholar_key: str = field(default="", repr=False)
    openalex_key: str = field(default="", repr=False)

    @property
    def use_mock(self) -> bool:
        return self.force_mock or not self.api_key


def load_settings() -> Settings:
    return Settings(
        api_key=os.getenv("MODEL_API_KEY", "").strip(),
        model=os.getenv("MUSE_MODEL", "muse-spark-1.3").strip(),
        base_url=os.getenv("MUSE_BASE_URL", "https://api.meta.ai/v1").strip(),
        reasoning_effort=os.getenv("MUSE_REASONING_EFFORT", "low").strip(),
        force_mock=os.getenv("MINI_MUSE_MOCK", "0").strip() == "1",
        semantic_scholar_key=os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip(),
        openalex_key=os.getenv("OPENALEX_API_KEY", "").strip(),
    )
