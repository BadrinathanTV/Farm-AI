from pymongo import MongoClient
from typing import List
from .config import settings
from .models import MemoryEntry

class MemoryStore:
    """Handles all database operations for natural language memories."""
    def __init__(self, db_name: str = "farm_assistant_db"):
        self.client = MongoClient(settings.mongo_uri)
        self.db = self.client[db_name]
        self.memory_collection = self.db["memories"]
        # Ensure index on user_id and timestamp
        self.memory_collection.create_index([("user_id", 1), ("timestamp", -1)])
        print("---MEMORY STORE: Connected to MongoDB---")

    def add_memory(self, user_id: str, content: str, memory_type: str = "activity"):
        """Adds a new memory entry."""
        memory = MemoryEntry(
            user_id=user_id,
            content=content,
            memory_type=memory_type
        )
        self.memory_collection.insert_one(memory.model_dump())
        print(f"---MEMORY STORE: Saved memory for user {user_id}: {content[:30]}...---")

    def get_recent_memories(self, user_id: str, limit: int = 10) -> List[MemoryEntry]:
        """Retrieves text-based memories for a user."""
        cursor = self.memory_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
        return [MemoryEntry(**m) for m in cursor]
