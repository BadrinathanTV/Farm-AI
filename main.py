# main.py

from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

from agents.supervisor import Supervisor
from agents.weather import WeatherAgent

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    next_agent: str

# Initialize Agents
supervisor_node = Supervisor(llm)
weather_node = WeatherAgent(llm)

# Placeholder agents
def knowledge_agent(state: AgentState):
    print("---KNOWLEDGE AGENT---")
    return {"messages": ["The Knowledge Agent is processing your request..."]}

def market_agent(state: AgentState):
    print("---MARKET AGENT---")
    return {"messages": ["The Market Agent is fetching price information..."]}

# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node.invoke)
workflow.add_node("weather", weather_node.invoke)
workflow.add_node("knowledge_support", knowledge_agent)
workflow.add_node("market_intelligence", market_agent)
workflow.set_entry_point("supervisor")

def router(state: AgentState):
    return state['next_agent'] if state['next_agent'] != "__end__" else END

workflow.add_conditional_edges(
    "supervisor",
    router,
    {
        "weather": "weather",
        "knowledge_support": "knowledge_support",
        "market_intelligence": "market_intelligence",
    }
)

workflow.add_edge("weather", END)
workflow.add_edge("knowledge_support", END)
workflow.add_edge("market_intelligence", END)
app = workflow.compile()

# Run the application
if __name__ == "__main__":
    print("---Starting new query: Weather Info---")
    weather_query = "Should I spray pesticide on my crops tomorrow?"
    inputs = {"messages": [HumanMessage(content=weather_query)]}
    final_state = app.invoke(inputs)
    
    response_message = final_state['messages'][-1]
    if isinstance(response_message, BaseMessage):
        print(f"\nFinal Response: {response_message.content}\n")
    else:
        print(f"\nFinal Response: {response_message}\n")

