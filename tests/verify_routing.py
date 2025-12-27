
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.profile_manager import ProfileManager

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agents.knowledge_support import KnowledgeSupportAgent
from core.memory_service import MemoryService
from core.farm_log_manager import FarmLogManager
from core.memory_store import MemoryStore

load_dotenv()

def verify_routing():
    print("--- Verifying Dynamic Routing ---")
    
    llm = ChatOpenAI(model="gpt-4o")
    profile_manager = ProfileManager()
    memory_store = MemoryStore()
    memory_service = MemoryService(profile_manager, memory_store)
    log_manager = FarmLogManager()
    
    agent = KnowledgeSupportAgent(llm, memory_service, log_manager)
    
    # 1. FAST PATH (Greeting)
    print("\n[TEST 1] Testing Greeting (Fast Path)...")
    msg1 = "Hello, who are you?"
    state1 = {
        "user_id": "test_routing",
        "messages": [HumanMessage(content=msg1)]
    }
    # We rely on print statements in the agent to verify the path taken
    res1 = agent.invoke(state1)
    print(f"Response: {res1['messages'][0].content}")

    # 2. SLOW PATH (RAG)
    print("\n[TEST 2] Testing RAG Question (Slow Path)...")
    msg2 = "How to grow tomatoes?"
    state2 = {
        "user_id": "test_routing",
        "messages": [HumanMessage(content=msg2)]
    }
    res2 = agent.invoke(state2)
    print(f"Response: {res2['messages'][0].content}")

if __name__ == "__main__":
    verify_routing()
