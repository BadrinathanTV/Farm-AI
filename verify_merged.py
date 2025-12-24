
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.profile_manager import ProfileManager

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from agents.supervisor import Supervisor
from agents.knowledge_support import KnowledgeSupportAgent
from core.memory_service import MemoryService
from core.farm_log_manager import FarmLogManager
from core.memory_store import MemoryStore

load_dotenv()

def verify_merged_agent():
    print("--- Verifying Merged Knowledge Support Agent ---")
    
    llm = ChatOpenAI(model="gpt-4o")
    profile_manager = ProfileManager()
    memory_store = MemoryStore()
    memory_service = MemoryService(profile_manager, memory_store)
    log_manager = FarmLogManager()
    
    agent = KnowledgeSupportAgent(llm, memory_service, log_manager)
    
    # 1. ACTION (Logging)
    print("\n[TEST 1] Testing Action Logging...")
    action_msg = "I harvested 20kg of tomatoes today."
    state1 = {
        "user_id": "test_user_merged",
        "messages": [HumanMessage(content=action_msg)],
        "detected_activity": "Harvesting" # Simulate profile agent detection
    }
    res1 = agent.invoke(state1)
    print(f"Response: {res1['messages'][0].content}")
    
    # Check if logged
    logs = log_manager.get_recent_logs("test_user_merged")
    if logs and logs[-1].activity_type == "Harvesting":
        print("PASS: Activity logged correctly.")
    else:
        print("FAIL: Activity NOT logged.")

    # 2. QUESTION (RAG)
    print("\n[TEST 2] Testing RAG Question...")
    q_msg = "How to store harvested tomatoes?"
    state2 = {
        "user_id": "test_user_merged",
        "messages": [HumanMessage(content=q_msg)]
    }
    res2 = agent.invoke(state2)
    print(f"Response: {res2['messages'][0].content}")
    
    if "refrigerat" in res2['messages'][0].content.lower() or "cool" in res2['messages'][0].content.lower() or "temperature" in res2['messages'][0].content.lower():
        print("PASS: RAG provided relevant answer.")
    else:
        print("WARNING: Answer might be generic, check content.")

if __name__ == "__main__":
    verify_merged_agent()
