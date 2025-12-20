# core/farm_log_manager.py

from pymongo import MongoClient
from typing import List
from .config import settings
from .models import FarmLog  # <-- IMPORT FROM new models.py

class FarmLogManager:
    """Handles all database operations for farm activity logs."""
    def __init__(self, db_name: str = "farm_assistant_db"):
        self.client = MongoClient(settings.final_mongo_uri)
        self.db = self.client[db_name]
        self.logs_collection = self.db["farm_logs"]
        print("---FARM LOG MANAGER: Connected to MongoDB---")

    def add_log(self, user_id: str, log: FarmLog):
        log_entry = log.model_dump()
        log_entry["user_id"] = user_id
        self.logs_collection.insert_one(log_entry)
        print(f"---FARM LOG MANAGER: Saved log for user {user_id}---")

    def get_recent_logs(self, user_id: str, limit: int = 5) -> List[FarmLog]:
        logs_cursor = self.logs_collection.find({"user_id": user_id}).sort("timestamp", -1).limit(limit)
        return [FarmLog(**log) for log in logs_cursor]

