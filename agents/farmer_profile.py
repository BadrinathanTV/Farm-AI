# agents/farmer_profile.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime, timezone

from core.profile_manager import ProfileManager
from core.memory_store import MemoryStore
from core.models import FarmerProfile, Crop
from tools.geocoding_api import get_coordinates_for_location

class CropUpdate(BaseModel):
    """Represents an action to update the farmer's crop list."""
    action: str = Field(description="The action to perform: 'add', 'remove', or 'harvest'.")
    crop_name: str = Field(description="The name of the crop.")
    sowing_date_str: Optional[str] = Field(description="The sowing date in YYYY-MM-DD format, or None if unknown. Calculate this if the user provides relative time (e.g., '3 weeks ago').")

class AgentResponse(BaseModel):
    """Defines the structured output we expect from the LLM."""
    response_message: str = Field(description="The conversational message to be sent to the user.")
    extracted_name: Optional[str] = Field(description="The full name extracted from the user's message, if present.")
    extracted_location: Optional[str] = Field(description="The location extracted from the user's message, if present.")
    crop_updates: Optional[List[CropUpdate]] = Field(description="List of crop updates (add/remove/harvest) mentioned by the user.")
    memorable_facts: Optional[List[str]] = Field(description="List of important facts or activities mentioned by the user to be remembered (e.g., 'I planted tomatoes', 'I harvested rice'). Extract these as natural language sentences.")
    detected_activity: Optional[str] = Field(description="A description of any farming activity mentioned by the user, if present.")

class FarmerProfileAgent:
    """A smart agent to manage user onboarding and answer questions about the user's profile and activities."""

    def __init__(self, llm: BaseLanguageModel, profile_manager: ProfileManager, memory_store: MemoryStore): 
        self.llm = llm
        self.profile_manager = profile_manager
        self.memory_store = memory_store  
        self.parser = JsonOutputParser(pydantic_object=AgentResponse)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a friendly and precise Farmer Profile agent.
Your goal is to have a natural conversation to manage the user's profile and answer their questions about it using ONLY the data provided.

**System Instructions:**
- Your response **must** be a JSON object following this format: {format_instructions}
- **Crucially, you must not invent, assume, or hallucinate any information.** Your answers must be grounded in the context below.
- **Response Style:** 
    - Be helpful, proactive, and natural. 
    - **DO NOT** simply list data properties like "I have logged X". 
    - **Instead, SYNTHESIZE the memory content into a conversational narrative.** 
    - Example: Instead of "I have logged you planted tomatoes on 2025-01-01", say "You planted tomatoes about 3 weeks ago. How are they coming along? Do you need any tips on irrigation?"

**CONTEXT - ALL KNOWN INFORMATION ABOUT THE USER:**
1.  **Static Profile:**
    {profile_data}
2.  **Memory / Context (Recent):**
    {memory_narrative}
3.  **Current Date:** {current_date}

**User's Latest Message:**
"{last_message}"

**Conversation History:**
{chat_history}

**Decision Logic:**
1.  **If the user asks about existence or past activities ("what am I farming?", "activities?"):** 
    - Use the **Memory / Context** to tell a story about their recent work.
    - Combine related events (e.g. "You planted tomatoes last week and harvested rice yesterday").
    - Be proactive: Add a helpful follow-up question related to their most recent activity.

2.  **Profile Updates (Dynamic Memory):**
    - Analyze the message for updates to the crop list.
    - **ADD/UPDATE:** If user says "I planted tomatoes 3 weeks ago" or "I am farming rice":
        - Action: "add"
        - Calculate `sowing_date_str` (YYYY-MM-DD) based on the *Current Date* and their relative time description (e.g., "3 weeks ago").
    - **HARVEST:** If user says "I harvested the rice":
        - Action: "harvest"
    - **REMOVE:** If user says "Remove tomatoes from list":
        - Action: "remove"

        - Action: "remove"
    
    - **MEMORABLE FACTS:**
        - Extract any significant activities or facts as natural language strings in `memorable_facts`.
        - E.g., User: "I harvested 50kg of rice yesterday." -> memorable_facts: ["User harvested 50kg of rice on [Date of yesterday]"]

3.  **General:** If the user provided an update, confirm it in `response_message`.

Respond with ONLY the JSON object.
""",
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, state: dict) -> dict:
        print("---FARMER PROFILE AGENT (DYNAMIC)---")
        user_id = state["user_id"]
        profile = self.profile_manager.load_profile(user_id)
        last_message = state["messages"][-1]
        
        chat_history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][:-1]])
        recent_memories = self.memory_store.get_recent_memories(user_id, limit=10)
        
        memory_list = []
        for mem in recent_memories:
            date_str = mem.timestamp.strftime("%Y-%m-%d")
            memory_list.append(f"- [{date_str}] {mem.content}")
        memory_narrative = "\n".join(memory_list) if memory_list else "No recorded memories yet."

        try:
            print("---FARMER PROFILE: Invoking LLM chain...---")
            response_data = self.chain.invoke({
                "profile_data": profile.model_dump_json(),
                "memory_narrative": memory_narrative,  
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "last_message": last_message.content,
                "chat_history": chat_history
            })
            print(f"---FARMER PROFILE: LLM response received---")
        except Exception as e:
            print(f"---FARMER PROFILE: LLM ERROR: {type(e).__name__} - {e}---")
            return {"messages": [AIMessage(content=f"I encountered an error processing your request. Please try again.")]}

        final_message = response_data.get("response_message", "I'm sorry, something went wrong.")
        new_data = {}

        if name := response_data.get("extracted_name"):
            if profile.full_name is None: new_data['full_name'] = name
        if loc := response_data.get("extracted_location"):
            if profile.location_name is None: new_data['location_name'] = loc
            try:
                coords = get_coordinates_for_location.invoke({"location_query": loc})
                if coords and "error" not in coords:
                    new_data['latitude'] = coords.get('latitude')
                    new_data['longitude'] = coords.get('longitude')
            except Exception as e:
                print(f"Geocoding tool failed: {e}")

        # --- DYNAMIC CROP MEMORY LOGIC ---
        if updates := response_data.get("crop_updates"):
            current_crops = {c.name.lower(): c for c in profile.crops}
            
            for update in updates:
                name = update['crop_name'].lower().strip()
                action = update['action']
                sowing_date = None
                if update.get('sowing_date_str'):
                    try:
                        sowing_date = datetime.strptime(update['sowing_date_str'], "%Y-%m-%d")
                    except ValueError:
                        pass # Keep None if parsing fails

                if action == "add":
                    # Update existing or create new
                    if name in current_crops:
                        print(f"Updating existing crop: {name}")
                        current_crops[name].status = "active" # Reactivate if it was harvested
                        if sowing_date:
                            current_crops[name].sowing_date = sowing_date
                    else:
                        print(f"Adding new crop: {name}")
                        current_crops[name] = Crop(name=update['crop_name'], sowing_date=sowing_date, status="active")
                
                elif action == "harvest":
                    if name in current_crops:
                        current_crops[name].status = "harvested"
                
                elif action == "remove":
                    if name in current_crops:
                        del current_crops[name]

            new_data['crops'] = list(current_crops.values())
        
        if new_data:
            updated_profile = profile.model_copy(update=new_data)
            self.profile_manager.save_profile(updated_profile)
        
        # --- SAVE MEMORIES ---
        if facts := response_data.get("memorable_facts"):
            for fact in facts:
                self.memory_store.add_memory(user_id, fact)

        activity = response_data.get("detected_activity")

        return {
            "messages": [AIMessage(content=final_message)],
            "detected_activity": activity
        }

