# agents/farmer_profile.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from core.profile_manager import ProfileManager
from core.farm_log_manager import FarmLogManager  # <-- IMPORT THE LOG MANAGER
from core.models import FarmerProfile
from tools.geocoding_api import get_coordinates_for_location

class AgentResponse(BaseModel):
    """Defines the structured output we expect from the LLM."""
    response_message: str = Field(description="The conversational message to be sent to the user.")
    extracted_name: Optional[str] = Field(description="The full name extracted from the user's message, if present.")
    extracted_location: Optional[str] = Field(description="The location extracted from the user's message, if present.")
    detected_activity: Optional[str] = Field(description="A description of any farming activity mentioned by the user, if present.")

class FarmerProfileAgent:
    """A smart agent to manage user onboarding and answer questions about the user's profile and activities."""

    def __init__(self, llm: BaseLanguageModel, profile_manager: ProfileManager, log_manager: FarmLogManager): # <-- INJECT LOG MANAGER
        self.llm = llm
        self.profile_manager = profile_manager
        self.log_manager = log_manager  # <-- STORE LOG MANAGER
        self.parser = JsonOutputParser(pydantic_object=AgentResponse)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a friendly and precise Farmer Profile agent.
Your goal is to have a natural conversation to manage the user's profile and answer their questions about it using ONLY the data provided.

**System Instructions:**
- Your response **must** be a JSON object following this format: {format_instructions}
- **Crucially, you must not invent, assume, or hallucinate any information.** Your answers must be grounded in the context below.

**CONTEXT - ALL KNOWN INFORMATION ABOUT THE USER:**
1.  **Static Profile (JSON):**
    {profile_data}
2.  **Recent Farm Activities (from their log):**
    {recent_activities}

**User's Latest Message:**
"{last_message}"

**Conversation History:**
{chat_history}

**Decision Logic:**
1.  **If the user asks about themselves** (e.g., "tell me about me," "what am I growing?"): Your `response_message` must synthesize a complete summary from BOTH the static profile and the recent activities.
2.  **If the profile is incomplete:** Analyze the user's message for missing info (`full_name`, `location_name`). Populate extracted fields and ask for the *next* missing piece of information.
3.  **If the profile is complete and the user is not asking a question:** Your `response_message` can be a polite confirmation.

Respond with ONLY the JSON object.
""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, state: dict) -> dict:
        print("---FARMER PROFILE AGENT (GROUNDED)---")
        user_id = state["user_id"]
        profile = self.profile_manager.load_profile(user_id)
        last_message = state["messages"][-1]
        
        # Create history string (excluding the last message which is passed separately)
        chat_history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][:-1]])

        # --- FETCH RECENT ACTIVITIES TO GROUND THE LLM ---
        recent_logs = self.log_manager.get_recent_logs(user_id, limit=5)
        if recent_logs:
            activities_str = "\n".join([f"- {log.details} ({(datetime.utcnow() - log.timestamp).days} days ago)" for log in recent_logs])
        else:
            activities_str = "No recent activities logged."

        response_data = self.chain.invoke({
            "profile_data": profile.model_dump_json(),
            "recent_activities": activities_str,  # <-- PASS THE MEMORY TO THE AGENT'S BRAIN
            "last_message": last_message.content,
            "chat_history": chat_history
        })

        final_message = response_data.get("response_message", "I'm sorry, something went wrong. Could you try again?")
        new_data = {}

        if name := response_data.get("extracted_name"):
            if profile.full_name is None: new_data['full_name'] = name
        if loc := response_data.get("extracted_location"):
            if profile.location_name is None: new_data['location_name'] = loc
        
        if new_data:
            if new_loc := new_data.get("location_name"):
                try:
                    coords = get_coordinates_for_location.invoke({"location_query": new_loc})
                    if coords and "error" not in coords:
                        new_data['latitude'] = coords.get('latitude')
                        new_data['longitude'] = coords.get('longitude')
                except Exception as e:
                    print(f"Geocoding tool failed: {e}")
            
            updated_profile = profile.model_copy(update=new_data)
            self.profile_manager.save_profile(updated_profile)
        
        activity = response_data.get("detected_activity")

        return {
            "messages": [AIMessage(content=final_message)],
            "detected_activity": activity
        }

