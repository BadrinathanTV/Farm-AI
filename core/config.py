# core/config.py

from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Loads application settings from .env file."""
    mongo_uri: str = "mongodb://localhost:27017/"
    openai_api_key: str
    google_api_key: Optional[str] = None

    class Config:
        env_file = ".env"

# Create a single, reusable instance of the settings
settings = Settings()


