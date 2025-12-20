# core/models.py

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class FarmLog(BaseModel):
    """Represents a single farming activity log entry."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    activity_type: str
    details: str

class MemoryEntry(BaseModel):
    """Represents a natural language memory of the user."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    content: str
    user_id: str
    memory_type: str = "activity" # activity, preference, fact

class Crop(BaseModel):
    """Represents a crop currently being grown."""
    name: str
    sowing_date: Optional[datetime] = None
    status: str = "active"  # "active", "harvested", "removed"

class FarmerProfile(BaseModel):
    """Defines the structure for a farmer's profile, including a password."""
    user_id: str
    hashed_password: Optional[str] = None
    full_name: Optional[str] = None
    location_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    crops: List[Crop] = []
