import os
import sys
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agents.supervisor import Supervisor
from core.profile_manager import ProfileManager
from core.config import settings

# Mock ProfileManager
class MockProfileManager:
    def load_profile(self, user_id):
        from core.models import FarmerProfile
        return FarmerProfile(user_id=user_id, full_name="Test User", location_name="Test Loc")

def test_routing():
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
    profile_manager = MockProfileManager()
    supervisor = Supervisor(llm, profile_manager)

    test_inputs = [
        "I am farming tomatoes",
        "I grow rice",
        "I planted chilli yesterday",
        "What am I farming?",
        "How old is my plant?"
    ]

    print("--- ROUTING TEST ---")
    for msg in test_inputs:
        state = {
            "user_id": "test_user", 
            "messages": [HumanMessage(content=msg)]
        }
        result = supervisor.invoke(state)
        print(f"Message: '{msg}' -> Agent: {result['next_agent']}")

if __name__ == "__main__":
    test_routing()
