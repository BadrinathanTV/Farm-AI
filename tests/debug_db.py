from core.profile_manager import ProfileManager
from core.models import FarmerProfile

def debug_db():
    print("--- DEBUGGING REAL MONGODB ---")
    pm = ProfileManager()
    
    # List all profiles
    print(f"Connected to DB: {pm.db.name}")
    profiles = list(pm.profiles_collection.find())
    print(f"Found {len(profiles)} profiles.")
    
    for p in profiles:
        print(f"User: {p.get('user_id')}")
        print(f"Crops: {p.get('crops', 'NO CROPS FIELD')}")
        print(f"Crops Grown (Old): {p.get('crops_grown', 'NO OLD FIELD')}")
        print("-" * 20)

if __name__ == "__main__":
    debug_db()
