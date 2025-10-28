# graph.py

from typing import TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from agents.supervisor import Supervisor
from agents.farmer_profile import FarmerProfileAgent
from agents.weather import WeatherAgent
from agents.agro_advisory import AgroAdvisoryAgent
from agents.knowledge_support import KnowledgeSupportAgent
from agents.formatter import FormatterAgent
from core.profile_manager import ProfileManager
from core.farm_log_manager import FarmLogManager
from core.config import settings

# --- INITIALIZE CORE COMPONENTS ---
llm = ChatOpenAI(model="gpt-4o", api_key=settings.openai_api_key)
profile_manager = ProfileManager()
log_manager = FarmLogManager()

# --- AGENT STATE ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    user_id: str
    next_agent: str
    detected_activity: Optional[str]

# --- AGENT NODE DEFINITIONS ---
supervisor_node = Supervisor(llm, profile_manager)
profile_agent_node = FarmerProfileAgent(llm, profile_manager, log_manager)
weather_agent_node = WeatherAgent(llm, profile_manager, log_manager)
agro_advisory_node = AgroAdvisoryAgent(llm, log_manager)
# --- FIX: Provide the log_manager to the KnowledgeSupportAgent ---
knowledge_agent_node = KnowledgeSupportAgent(llm, profile_manager, log_manager)
formatter_node = FormatterAgent(llm, profile_manager)

# --- GRAPH WIRING ---
workflow = StateGraph(AgentState)

# Add all nodes to the graph
workflow.add_node("supervisor", supervisor_node.invoke)
workflow.add_node("farmer_profile", profile_agent_node.invoke)
workflow.add_node("agro_advisory", agro_advisory_node.invoke)
workflow.add_node("weather", weather_agent_node.invoke)
workflow.add_node("knowledge_support", knowledge_agent_node.invoke)
workflow.add_node("formatter", formatter_node.invoke)
workflow.add_node("market_intelligence", lambda state: {"messages": [AIMessage(content="Market agent is under construction.")]})


# --- ROUTING LOGIC ---
def supervisor_router(state: AgentState):
    return state.get("next_agent", END)

def profile_router(state: AgentState):
    if activity := state.get("detected_activity"):
        print("---PROFILE ROUTER: Activity detected, routing to advisory agent.---")
        state["messages"][-1] = HumanMessage(content=activity)
        return "agro_advisory"
    else:
        print("---PROFILE ROUTER: No activity, routing to formatter.---")
        return "formatter"

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges("supervisor", supervisor_router, {
    "farmer_profile": "farmer_profile",
    "agro_advisory": "agro_advisory",
    "weather": "weather",
    "knowledge_support": "knowledge_support",
    "market_intelligence": "market_intelligence",
    END: END
})

workflow.add_conditional_edges("farmer_profile", profile_router)

workflow.add_edge("agro_advisory", "formatter")
workflow.add_edge("weather", "formatter")
workflow.add_edge("knowledge_support", "formatter")
workflow.add_edge("market_intelligence", "formatter")

workflow.add_edge("formatter", END)

app = workflow.compile()
print("---GRAPH COMPILED: KNOWLEDGE AGENT IS NOW DATABASE-CONNECTED---")

