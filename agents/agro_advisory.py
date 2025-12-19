# agents/agro_advisory.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from core.farm_log_manager import FarmLogManager
from core.models import FarmLog

class AgroAdvisoryAgent:
    """
    A time-intelligent agent that understands and logs farming activities with accurate timestamps.
    It works in the background and does not generate a direct response to the user.
    """
    def __init__(self, llm: BaseLanguageModel, log_manager: FarmLogManager):
        self.llm = llm
        self.log_manager = log_manager
        
        # This Pydantic model now includes a timestamp for the LLM to populate.
        class ActivityLog(BaseModel):
            activity_type: str = Field(description="The category of farming activity, e.g., 'Planting', 'Irrigation'.")
            details: str = Field(description="A concise summary of the specific activity performed.")
            # The LLM will calculate the date based on the user's message and the current date.
            timestamp_str: str = Field(description="The date of the activity in YYYY-MM-DD format. Calculate this based on the current date if the user provides a relative time like 'today' or 'yesterday'.")

        self.parser = JsonOutputParser(pydantic_object=ActivityLog)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a data extraction specialist with perfect understanding of time.
Your job is to analyze a user's description of a farming activity and extract its details, including the correct date.
RUN FASTERRR , WE NEED LOW LATECY !!
**Current Date:** {current_date}

**User's description of activity:** "{message}"

**Your Task:**
1.  Analyze the user's message to find the activity and when it happened.
2.  If the user says "today," use the current date.
3.  If the user says "yesterday," calculate yesterday's date.
4.  You MUST return the date in "YYYY-MM-DD" format in the `timestamp_str` field.

{format_instructions}
"""
        )
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, state: dict) -> dict:
        """Main entry point for this 'silent' agent to log activities with correct timestamps."""
        print("---AGRO ADVISORY AGENT (TIME-AWARE LOGGING)---")
        user_id = state["user_id"]
        activity_text = state.get("detected_activity")
        
        if activity_text:
            try:
                activity_data = self.chain.invoke({
                    "message": activity_text,
                    "current_date": datetime.now().strftime("%Y-%m-%d")
                })
                
                # Convert the LLM's date string into a real datetime object
                log_timestamp = datetime.strptime(activity_data["timestamp_str"], "%Y-%m-%d")

                farm_log = FarmLog(
                    activity_type=activity_data["activity_type"],
                    details=activity_data["details"],
                    timestamp=log_timestamp
                )
                self.log_manager.add_log(user_id, farm_log)
            except Exception as e:
                print(f"---AGRO ADVISORY FAILED to parse or log activity: {e}---")

        # Return an empty dictionary to pass the state through without changing messages.
        return {}

