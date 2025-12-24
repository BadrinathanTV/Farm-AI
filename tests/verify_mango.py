
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from core.memory_service import MemoryService
from core.profile_manager import ProfileManager
from core.memory_store import MemoryStore

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.knowledge_support import KnowledgeSupportAgent

load_dotenv()

def verify_mango_query():
    print("--- Verifying Mango Pruning Query ---")
    
    # dependencies
    llm = ChatOpenAI(model="gpt-4o")
    profile_manager = ProfileManager() 
    memory_store = MemoryStore()
    memory_service = MemoryService(profile_manager, memory_store)
    
    agent = KnowledgeSupportAgent(llm, memory_service)
    
    question = "How to prune mango trees?"
    print(f"\nQuestion: {question}")
    
    state = {
        "user_id": "test_user",
        "messages": [HumanMessage(content=question)]
    }
    
    try:
        response = agent.invoke(state)
        content = response["messages"][0].content
        print(f"\nAgent Response:\n{content}")
        
    except Exception as e:
        print(f"\nFAIL: Exception: {e}")

if __name__ == "__main__":
    verify_mango_query()
