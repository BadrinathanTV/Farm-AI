# agents/weather.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from tools.weather_api import get_weather_forecast
from core.memory_service import MemoryService

class WeatherAgent:
    """A hyper-personalized and precise weather advisor."""

    def __init__(self, llm: BaseLanguageModel, memory_service: MemoryService):
        self.llm = llm
        self.memory = memory_service
        self.prompt = ChatPromptTemplate.from_template(
            """You are a world-class agronomist AI providing hyper-precise weather advice.

**System Time:** {current_time}

**Context:**
- Farmer's Name: {farmer_name}
- Location: {location}
- **Active Crops:** {active_crops}
- Recent Activities: {recent_activities}
- Raw 7-Day Weather Forecast: {weather_data}
- User's Specific Question: "{question}"

**Your Strict Instructions:**
1.  **Analyze the user's question to determine the exact time frame they are asking about (e.g., "today," "tomorrow," "this week").**
2.  **Your response MUST be limited to that specific time frame.** If they ask for "today," only provide today's forecast and advice.
3.  **Ground your advice in the farmer's ACTIVE CROPS.** e.g., if growing tomatoes, warn about blight in high humidity.
4.  Your tone should be that of a knowledgeable and reliable expert.

Based on these instructions, provide a concise and actionable response.
"""
        )
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---WEATHER AGENT (MEMORY SERVICE)---")
        user_id = state["user_id"]
        ctx = self.memory.get_context(user_id)

        if not ctx.get("latitude") or not ctx.get("longitude"):
            return {"messages": [AIMessage(content="I can't give a forecast without your location.")]}

        weather_data = get_weather_forecast.invoke({
            "latitude": ctx["latitude"],
            "longitude": ctx["longitude"]
        })

        response = self.chain.invoke({
            "current_time": ctx["current_time"],
            "weather_data": weather_data,
            "location": ctx["location"],
            "farmer_name": ctx["farmer_name"],
            "active_crops": ctx["active_crops"],
            "recent_activities": ctx["memory_narrative"],
            "question": state["messages"][-1].content
        })

        return {"messages": [AIMessage(content=response.content)]}
