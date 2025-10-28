# core/profile_manager.py

from pymongo import MongoClient
from typing import Optional
from passlib.context import CryptContext
from .config import settings
from .models import FarmerProfile  # <-- IMPORT FROM new models.py

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class ProfileManager:
    """Handles user creation, authentication, and profile management in MongoDB."""

    def __init__(self, db_name: str = "farm_assistant_db"):
        self.client = MongoClient(settings.mongo_uri)
        self.db = self.client[db_name]
        self.profiles_collection = self.db["profiles"]
        self.profiles_collection.create_index("user_id", unique=True)
        print("---PROFILE MANAGER: Connected to MongoDB---")

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)

    def get_user(self, user_id: str) -> Optional[FarmerProfile]:
        user_data = self.profiles_collection.find_one({"user_id": user_id})
        return FarmerProfile(**user_data) if user_data else None

    def create_user(self, user_id: str, password: str) -> FarmerProfile:
        if self.get_user(user_id):
            raise ValueError("Username already exists.")
        
        hashed_password = self.get_password_hash(password)
        new_user = FarmerProfile(user_id=user_id, hashed_password=hashed_password)
        self.profiles_collection.insert_one(new_user.model_dump())
        print(f"---PROFILE MANAGER: Created new user '{user_id}'---")
        return new_user

    def authenticate_user(self, user_id: str, password: str) -> Optional[FarmerProfile]:
        user = self.get_user(user_id)
        if user and user.hashed_password and self.verify_password(password, user.hashed_password):
            return user
        return None

    def load_profile(self, user_id: str) -> FarmerProfile:
        return self.get_user(user_id) or FarmerProfile(user_id=user_id)

    def save_profile(self, profile: FarmerProfile):
        profile_dict = profile.model_dump()
        self.profiles_collection.update_one(
            {"user_id": profile.user_id},
            {"$set": profile_dict},
            upsert=True
        )
        print(f"---PROFILE MANAGER: Saved profile for user {profile.user_id}---")

