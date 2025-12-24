
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.profile_manager import ProfileManager

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.supervisor import Supervisor

load_dotenv()

def test_routing():
    print("--- Testing Supervisor Routing ---")
    
    llm = ChatOpenAI(model="gpt-4o")
    profile_manager = ProfileManager() 
    supervisor = Supervisor(llm, profile_manager)
    
    # CASE 1: Question (RAG)
    msg1 = "How to prune mango trees?"
    print(f"\nUser: {msg1}")
    state1 = {
        "user_id": "test_user",
        "messages": [HumanMessage(content=msg1)]
    }
    res1 = supervisor.invoke(state1)
    print(f"Route: {res1['next_agent']}")
    
    if res1['next_agent'] == "knowledge_support":
        print("PASS: Correctly routed to knowledge_support.")
    else:
        print(f"FAIL: Expected knowledge_support, got {res1['next_agent']}")

    # CASE 2: Action (Log)
    msg2 = "I pruned the mango trees today."
    print(f"\nUser: {msg2}")
    state2 = {
        "user_id": "test_user",
        "messages": [HumanMessage(content=msg2)]
    }
    res2 = supervisor.invoke(state2)
    print(f"Route: {res2['next_agent']}")
    
    if res2['next_agent'] == "agro_advisory":
        print("PASS: Correctly routed to agro_advisory.")
    else:
        print(f"FAIL: Expected agro_advisory, got {res2['next_agent']}")

if __name__ == "__main__":
    test_routing()
