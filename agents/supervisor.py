# agents/supervisor.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from core.profile_manager import ProfileManager

AGENT_DESCRIPTIONS = {
    "farmer_profile": "Handles and answers questions about the user's personal profile (e.g., 'what is my name?', 'tell me about me'). Also manages the initial user onboarding.",
    "agro_advisory": "Handles conversations about specific farming activities (planting, watering, etc.), logs these activities, and provides advice.",
    "weather": "Provides personalized weather forecasts and farming advice based on the weather.",
    "market_intelligence": "For questions about market prices, government schemes, and subsidies.",
    "knowledge_support": "For all other general farming questions, best practices, pest control, etc.",
    "plant_disease": "Analyzes images or descriptions of sick plants to diagnose diseases and recommend treatments."
}

class Supervisor:
    """The central router for the multi-agent system."""

    def __init__(self, llm: BaseLanguageModel, profile_manager: ProfileManager):
        self.llm = llm
        self.profile_manager = profile_manager

        agent_options = "\n- ".join([f"{name}: {desc}" for name, desc in AGENT_DESCRIPTIONS.items()])
        prompt_template = f"""You are the supervisor of a multi-agent agricultural assistant. Your sole responsibility is to analyze the user's latest message and route it to the correct agent.

**Available Agents:**
{agent_options}

**Strict Routing Rules:**
1.  If the user's profile is incomplete, you **must** route to `farmer_profile`.
2.  If the user asks a question about themselves, their profile, or the information the AI knows about them (e.g., "tell me about me," "what's my location," "do you know my name"), you **must** route to `farmer_profile`.
3.  If the user describes a farming task they performed (e.g., "I planted rice"), route to `agro_advisory`.
4.  If the user asks about weather, route to `weather`.
5.  If the user asks about a plant disease OR uploads an image of a plant, route to `plant_disease`.
6.  For all other farming questions, route to `knowledge_support`.


Based on the rules, the user's profile status, and their last message, which agent should be called? Respond with only the agent's name.

**User Profile Status:** {{profile_status}}
**User's last message:** "{{last_message}}"
"""
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---SUPERVISOR---")
        user_id = state["user_id"]
        profile = self.profile_manager.load_profile(user_id)
        
        if not all([profile.full_name, profile.location_name]):
            print("Supervisor: Profile incomplete. Forcing route to farmer_profile.")
            return {"next_agent": "farmer_profile"}

        last_message: BaseMessage = state['messages'][-1]
        
        response = self.chain.invoke({
            "profile_status": "Profile is complete.",
            "last_message": last_message.content
        })
        next_agent_name = response.content.strip()
        
        print(f"Supervisor decided next agent is: {next_agent_name}")
        return {"next_agent": next_agent_name}

