from dataclasses import dataclass
from functools import lru_cache
from typing import Optional
import os
from dotenv import load_dotenv

# Load .env once when the settings module is imported
load_dotenv()


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: Optional[str]
    model_name: str
    langsearch_api_key: Optional[str]

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            base_url=os.getenv("URL_BASE", "https://api.openai.com/v1"),
            api_key=os.getenv("API_KEY"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            langsearch_api_key=os.getenv("LANGSEARCH_API_KEY"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings loaded from environment.
    Using LRU cache provides a simple singleton to avoid repeated env lookups.
    """
    return Settings.from_env()