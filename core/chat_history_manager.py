# core/chat_history_manager.py

from pymongo import MongoClient
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from typing import List, Dict
from .config import settings
import uuid

class ChatHistoryManager:
    """Handles all database operations for multiple chat sessions per user."""
    def __init__(self, db_name: str = "farm_assistant_db"):
        self.client = MongoClient(settings.mongo_uri)
        self.db = self.client[db_name]
        self.history_collection = self.db["chat_histories"]
        print("---CHAT HISTORY MANAGER: Connected to MongoDB---")

    def get_chat_sessions(self, user_id: str) -> List[Dict]:
        """Retrieves all chat sessions for a user, sorted by most recent."""
        sessions = self.history_collection.find({"user_id": user_id}).sort("timestamp", -1)
        return [{"chat_id": s["chat_id"], "title": s.get("title", "New Chat")} for s in sessions]

    def load_history(self, chat_id: str) -> List[BaseMessage]:
        """Loads the message history for a specific chat session."""
        history_data = self.history_collection.find_one({"chat_id": chat_id})
        if history_data and "messages" in history_data:
            return messages_from_dict(history_data["messages"])
        return []

    def save_history(self, user_id: str, chat_id: str, messages: List[BaseMessage]):
        """Saves or updates the message history for a specific chat session."""
        history_dict = messages_to_dict(messages)
        
        # Create a title for the chat based on the first human message
        title = "New Chat"
        if len(messages) > 1 and messages[0].type == "human":
            title = messages[0].content[:50] + "..." # Truncate for display
        
        self.history_collection.update_one(
            {"chat_id": chat_id},
            {
                "$set": {
                    "user_id": user_id,
                    "messages": history_dict,
                    "title": title,
                    "timestamp": uuid.UUID(chat_id).time
                }
            },
            upsert=True
        )
        print(f"---CHAT HISTORY MANAGER: Saved history for chat {chat_id}---")

    def delete_chat(self, chat_id: str):
        """Deletes a specific chat session."""
        self.history_collection.delete_one({"chat_id": chat_id})
        print(f"---CHAT HISTORY MANAGER: Deleted chat {chat_id}---")


