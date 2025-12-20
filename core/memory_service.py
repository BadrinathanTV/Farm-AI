# core/memory_service.py

from datetime import datetime, timezone
from core.profile_manager import ProfileManager
from core.memory_store import MemoryStore

class MemoryService:
    """
    A centralized service for providing user context to all agents.
    This is the single source of truth for what the AI knows about the farmer.
    """
    def __init__(self, profile_manager: ProfileManager, memory_store: MemoryStore):
        self.profile_manager = profile_manager
        self.memory_store = memory_store
        self._cache = {} # Simple in-memory cache for current request

    def get_context(self, user_id: str) -> dict:
        """
        Returns a dictionary containing all relevant context for a user.
        This can be passed directly to agent prompts.
        """
        # Check cache first (within same request cycle)
        if user_id in self._cache:
            return self._cache[user_id]

        profile = self.profile_manager.load_profile(user_id)
        recent_memories = self.memory_store.get_recent_memories(user_id, limit=10)

        # Format crops
        if profile.crops:
            crops_list = []
            for c in profile.crops:
                if c.status == 'active':
                    age_str = ""
                    if c.sowing_date:
                        days = (datetime.now(timezone.utc).replace(tzinfo=None) - c.sowing_date).days
                        age_str = f", {days} days old"
                    crops_list.append(f"{c.name}{age_str}")
            crops_str = ", ".join(crops_list) if crops_list else "None active"
        else:
            crops_str = "None"

        # Format memory narrative
        if recent_memories:
            # Create a chronological narrative from memories
            memory_list = []
            for mem in recent_memories:
                date_str = mem.timestamp.strftime("%Y-%m-%d")
                memory_list.append(f"- [{date_str}] {mem.content}")
            
            memory_context_str = "\n".join(memory_list)
        else:
            memory_context_str = "No recorded memories yet."

        context = {
            "farmer_name": profile.full_name or "Farmer",
            "location": profile.location_name or "Unknown",
            "latitude": profile.latitude,
            "longitude": profile.longitude,
            "active_crops": crops_str,
            "memory_narrative": memory_context_str,
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_time": datetime.now().strftime("%A, %B %d, %Y %I:%M %p"),
        }
        
        self._cache[user_id] = context
        return context

    def clear_cache(self):
        """Clear the cache after a request cycle."""
        self._cache = {}
