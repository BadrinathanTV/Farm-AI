from google.adk.agents import Agent
from google.adk.tools import google_search

try:
    agent = Agent(
        name="market_agent",
        model="gemini-1.5-flash",
        tools=[google_search]
    )
    print("Agent instantiated successfully")
    print("Agent tools:", agent.tools)
except Exception as e:
    print(f"Failed to instantiate Agent: {e}")
