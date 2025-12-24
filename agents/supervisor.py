# agents/supervisor.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from core.profile_manager import ProfileManager

AGENT_DESCRIPTIONS = {
    "farmer_profile": "Handles the user's profile, planting new crops, and profile questions.",
    "weather": "Provides personalized weather forecasts and farming advice.",
    "market_intelligence": "For questions about market prices, government schemes, and subsidies.",
    "knowledge_support": "The MAIN agent for: 1. Logging COMPLETED actions ('I watered', 'I pruned'), 2. Answering ANY farming questions ('How to prune?', 'Best pesticide?').",
    "plant_disease": "Analyzes images or descriptions of sick plants to diagnose diseases."
}

class Supervisor:
    """The central router for the multi-agent system."""

    def __init__(self, llm: BaseLanguageModel, profile_manager: ProfileManager):
        self.llm = llm
        self.profile_manager = profile_manager

        agent_options = "\n- ".join([f"{name}: {desc}" for name, desc in AGENT_DESCRIPTIONS.items()])
        prompt_template = f"""You are the supervisor of a multi-agent agricultural assistant. Route the user's message to the correct agent.

**Available Agents:**
{agent_options}

**Strict Routing Rules:**
1.  **Profile Check:** If profile is incomplete, route to `farmer_profile` ONLY if needed. Otherwise, route to specific agent.
3.  **Unified Support:**
    - If the user asks **"HOW to..."**, **"Guide for..."**, or asks advice -> Route to `knowledge_support` (It searches manuals).
    - If the user asks about **Government Schemes**, **Subsidies**, or **Policies** -> Route to `knowledge_support` (It searches RAG/Documents).
    - If the user says **"I HAVE DONE"** or describes a completed action ("I pruned trees", "Watering done") -> Route to `knowledge_support` (It logs the action).
4.  **Market Intelligence:** 
    - Queries about **PRICES**, **RATES**, or **MARKET COSTS** ONLY.
    - **CRITICAL:** If user says "find in [Location]" or "price of it" or "find it [location]", route to `market_intelligence`.
5.  **Farmer Profile:** Use ONLY for planting NEW crops ("I planted tomatoes") or asking about self ("What am I growing?").
6.  **Weather:** For weather questions.
7.  **Disease:** For sick plants/images.

Based on the rules, which agent should be called? Respond with only the agent's name.

**User Profile Status:** {{profile_status}}
**User's last message:** "{{last_message}}"
"""
        self.prompt = ChatPromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---SUPERVISOR---")
        # RULE 0: If there is an image, it MUST go to plant_disease. THIS IS TOP PRIORITY.
        if state.get("image_data"):
             print("Supervisor: Image detected. Forcing route to plant_disease.")
             return {"next_agent": "plant_disease"}

        user_id = state["user_id"]
        profile = self.profile_manager.load_profile(user_id)
        
        last_message: BaseMessage = state['messages'][-1]
        
        # Check if we should warn about profile but let the LLM decide
        profile_status_msg = "Profile is complete."
        if not all([profile.full_name, profile.location_name]):
             profile_status_msg = "Profile is INCOMPLETE (missing name or location)."

        response = self.chain.invoke({
            "profile_status": profile_status_msg,
            "last_message": last_message.content
        })
        next_agent_name = response.content.strip()
        
        print(f"Supervisor decided next agent is: {next_agent_name}")
        return {"next_agent": next_agent_name}

