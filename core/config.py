# core/config.py

from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Loads application settings from .env file."""
    mongo_uri: Optional[str] = None
    mongo_user: Optional[str] = None
    mongo_password: Optional[str] = None
    mongo_host: str = "localhost"
    mongo_port: int = 27017
    
    openai_api_key: str
    google_api_key: Optional[str] = None
    huggingfacehub_api_token: Optional[str] = None
    
    # Twilio Configuration
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_phone_number: Optional[str] = None

    @property
    def final_mongo_uri(self) -> str:
        """Constructs safe MongoDB URI from components (preferred) or returns the provided one."""
        if self.mongo_user and self.mongo_password:
            import urllib.parse
            user = urllib.parse.quote_plus(self.mongo_user)
            password = urllib.parse.quote_plus(self.mongo_password)
            return f"mongodb+srv://{user}:{password}@{self.mongo_host}/"
            
        if self.mongo_uri:
            return self.mongo_uri
        
        return f"mongodb://{self.mongo_host}:{self.mongo_port}/"

    class Config:
        env_file = ".env"

# Create a single, reusable instance of the settings
settings = Settings()


