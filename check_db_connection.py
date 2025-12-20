import sys
from core.config import settings
from pymongo import MongoClient

def check_connection():
    print("--- Checking MongoDB Connection ---")
    print(f"Connection String (masked): {settings.final_mongo_uri.split('@')[-1] if '@' in settings.final_mongo_uri else '...local...'}")
    
    try:
        client = MongoClient(settings.final_mongo_uri)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("✅ Connection successful!")
        return True
    except Exception as e:
        print("❌ Connection failed!")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if check_connection():
        sys.exit(0)
    else:
        sys.exit(1)
