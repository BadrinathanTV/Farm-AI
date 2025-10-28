# agents/weather.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from tools.weather_api import get_weather_forecast
from core.profile_manager import ProfileManager
from core.farm_log_manager import FarmLogManager
from datetime import datetime

class WeatherAgent:
    """A hyper-personalized and precise weather advisor."""

    def __init__(self, llm: BaseLanguageModel, profile_manager: ProfileManager, log_manager: FarmLogManager):
        self.llm = llm
        self.profile_manager = profile_manager
        self.log_manager = log_manager
        # --- THE FINAL, PRECISION PROMPT ---
        self.prompt = ChatPromptTemplate.from_template(
            """You are a world-class agronomist AI providing hyper-precise weather advice.

**System Time:** {current_time}

**Context:**
- Farmer's Name: {farmer_name}
- Location: {location}
- Farmer's Recent Activities: {recent_activities}
- Raw 7-Day Weather Forecast: {weather_data}
- User's Specific Question: "{question}"

**Your Strict Instructions:**
1.  **Analyze the user's question to determine the exact time frame they are asking about (e.g., "today," "tomorrow," "this week").**
2.  **Your response MUST be limited to that specific time frame.** If they ask for "today," only provide today's forecast and advice. Do not give a 7-day summary unless they ask for it.
3.  **Ground your advice in the farmer's recent activities.** If they planted paddy, the advice must be for their paddy field within the requested time frame.
4.  Your tone should be that of a knowledgeable and reliable expert.

Based on these instructions, provide a concise and actionable response.
"""
        )
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---WEATHER AGENT (PRECISION MODE)---")
        user_id = state["user_id"]
        profile = self.profile_manager.load_profile(user_id)

        if not profile.latitude or not profile.longitude:
            return {"messages": [AIMessage(content="I can't give a forecast without your location.")]}

        recent_logs = self.log_manager.get_recent_logs(user_id, limit=3)
        activities_str = "\n".join([f"- {log.details}" for log in recent_logs]) if recent_logs else "No recent activities."

        weather_data = get_weather_forecast.invoke({
            "latitude": profile.latitude,
            "longitude": profile.longitude
        })

        response = self.chain.invoke({
            "current_time": datetime.now().strftime("%A, %B %d, %Y %I:%M %p"),
            "weather_data": weather_data,
            "location": profile.location_name,
            "farmer_name": profile.full_name or "Farmer",
            "recent_activities": activities_str,
            "question": state["messages"][-1].content
        })

        # Return the raw, expert output for the Formatter to polish
        return {"messages": [AIMessage(content=response.content)]}

