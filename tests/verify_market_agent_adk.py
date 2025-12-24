
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.market_intelligence import MarketIntelligenceAgent

load_dotenv()

def verify_market_agent_adk():
    print("--- Verifying Market Intelligence Agent (ADK) ---")
    
    # Initialize LLM for extraction (using OpenAI as per graph.py)
    try:
        llm = ChatOpenAI(model="gpt-4o")
    except Exception as e:
        print(f"Failed to init ChatOpenAI: {e}")
        return

    try:
        # Mock MemoryService and its dependencies
        mock_memory_service = MagicMock()
        mock_memory_service.get_context.return_value = {
            "location": "Chennai",
            "active_crops": "Tomatoes"
        }
        
        agent = MarketIntelligenceAgent(llm, mock_memory_service)
        print("Agent initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    # Test invoke
    crop = "Tomatoes"
    location = "Chennai"
    print(f"\nTesting invoke for {crop} in {location}...")
    
    state = {
        "user_id": "test_user",
        "messages": [HumanMessage(content=f"What is the price of {crop} in {location}?")]
    }
    
    try:
        response = agent.invoke(state)
        content = response["messages"][0].content
        print(f"\nAgent Response:\n{content}")
        
        # Check if the generated search query (printed in logs) contained "India"
        # We can't verify the logs programmatically easily here without capturing stdout,
        # but we will manually verify the output.
        if "DuckDuckGo" in content:
            print("\nPASS: Response contains DuckDuckGo citation.")
        else:
            print("\nWARNING: Response missing citation.")
            
    except Exception as e:
        print(f"\nFAIL: Exception during invoke: {e}")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    verify_market_agent_adk()
