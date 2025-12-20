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
from agents.market_intelligence import MarketIntelligenceAgent
from agents.plant_disease import PlantDiseaseAgent
# Formatter agent removed for latency optimization
from core.profile_manager import ProfileManager
from core.farm_log_manager import FarmLogManager
from core.memory_service import MemoryService
from core.memory_store import MemoryStore
from core.config import settings

# --- INITIALIZE CORE COMPONENTS ---
# OPTIMIZATION: Use gpt-4o-mini for faster response times
llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.openai_api_key)
profile_manager = ProfileManager()
log_manager = FarmLogManager()
memory_store = MemoryStore()
memory_service = MemoryService(profile_manager, memory_store)

# --- AGENT STATE ---
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], lambda x, y: x + y]
    user_id: str
    next_agent: str
    detected_activity: Optional[str]
    image_data: Optional[bytes]

# --- AGENT NODE DEFINITIONS ---
supervisor_node = Supervisor(llm, profile_manager)
profile_agent_node = FarmerProfileAgent(llm, profile_manager, memory_store)
weather_agent_node = WeatherAgent(llm, memory_service)
agro_advisory_node = AgroAdvisoryAgent(llm, log_manager)
knowledge_agent_node = KnowledgeSupportAgent(llm, memory_service)
market_agent_node = MarketIntelligenceAgent(llm, memory_service)
plant_disease_node = PlantDiseaseAgent(llm)

# --- GRAPH WIRING ---
workflow = StateGraph(AgentState)

# Add all nodes to the graph
workflow.add_node("supervisor", supervisor_node.invoke)
workflow.add_node("farmer_profile", profile_agent_node.invoke)
workflow.add_node("agro_advisory", agro_advisory_node.invoke)
workflow.add_node("weather", weather_agent_node.invoke)
workflow.add_node("knowledge_support", knowledge_agent_node.invoke)
workflow.add_node("market_intelligence", market_agent_node.invoke)
workflow.add_node("plant_disease", plant_disease_node.invoke)

# --- ROUTING LOGIC ---
def supervisor_router(state: AgentState):
    return state.get("next_agent", END)

def profile_router(state: AgentState):
    if state.get("detected_activity"):
        print("---PROFILE ROUTER: Activity detected, routing to advisory agent.---")
        return "agro_advisory"
    else:
        # OPTIMIZATION: Route to END instead of formatter
        print("---PROFILE ROUTER: No activity, ending turn.---")
        return END

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges("supervisor", supervisor_router, {
    "farmer_profile": "farmer_profile",
    "agro_advisory": "agro_advisory",
    "weather": "weather",
    "knowledge_support": "knowledge_support",
    "market_intelligence": "market_intelligence",
    "plant_disease": "plant_disease",
    END: END
})

workflow.add_conditional_edges("farmer_profile", profile_router)

# OPTIMIZATION: All agents route to END instead of Formatter
workflow.add_edge("agro_advisory", END)
workflow.add_edge("weather", END)
workflow.add_edge("knowledge_support", END)
workflow.add_edge("market_intelligence", END)
workflow.add_edge("plant_disease", END)

app = workflow.compile()
print("---GRAPH COMPILED: OPTIMIZED MODE (gpt-4o-mini)---")
