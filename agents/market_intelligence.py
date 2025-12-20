# agents/market_intelligence.py

import uuid
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

# Google ADK imports
from google.adk import Agent, Runner
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
from google.adk.flows.llm_flows.contents import types as content_types
from google.adk.models.lite_llm import LiteLlm

from core.memory_service import MemoryService

load_dotenv()

class MarketIntelligenceAgent:
    """
    Agent responsible for providing market insights and crop prices using Google ADK and Google Search.
    Uses MemoryService to know what crops the user is farming.
    """
    def __init__(self, llm: BaseLanguageModel, memory_service: MemoryService):
        self.llm = llm
        self.memory = memory_service
        
        # Configure ADK Agent
        self.adk_agent = Agent(
            name="market_agent",
            model=LiteLlm("openai/gpt-4o-mini"),
            tools=[google_search],
            instruction="""You are a local market expert. Your goal is to find the most recent and specific market prices for crops in the specified location.
            - Prioritize "mandi" rates and daily market reports.
            - If exact local data is missing, find the nearest major market data.
            - Always mention the date of the price information.
            - Be concise and direct."""
        )
        self.session_service = InMemorySessionService()
        self.runner = Runner(
            agent=self.adk_agent, 
            session_service=self.session_service, 
            app_name="farm_ai"
        )

        class MarketQuery(BaseModel):
            crop: str = Field(description="The crop the user is asking about. If user says 'my crops', use the active crops from context.")
            location: str = Field(description="The location for the market data.")

        self.parser = JsonOutputParser(pydantic_object=MarketQuery)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a market analyst. Extract the crop and location from the user's query.
            If no location is specified, use the user's known location.
            If user says "my crops", use the active crops from context.
            
            **User's Context:**
            - Location: {location}
            - Active Crops: {active_crops}
            
            User Query: {message}
            
            {format_instructions}
            """
        )
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, state: dict) -> dict:
        print("---MARKET INTELLIGENCE AGENT (MEMORY SERVICE)---")
        user_id = state["user_id"]
        ctx = self.memory.get_context(user_id)
        
        messages = state["messages"]
        last_message = messages[-1].content
        
        try:
            # Extract intent with user context
            query_data = self.chain.invoke({
                "message": last_message,
                "location": ctx["location"],
                "active_crops": ctx["active_crops"],
                "format_instructions": self.parser.get_format_instructions()
            })
            crop = query_data["crop"]
            location = query_data["location"]
            
            # Construct query for ADK Agent
            search_query = f"latest market price of {crop} in {location} today mandi rates"
            
            session_id = str(uuid.uuid4())
            adk_user_id = "farm_ai_user"
            
            self.session_service.create_session_sync(
                session_id=session_id, 
                user_id=adk_user_id,
                app_name="farm_ai"
            )
            
            adk_message = content_types.Content(
                role='user', 
                parts=[content_types.Part(text=search_query)]
            )
            
            response_text = ""
            events = self.runner.run(
                user_id=adk_user_id,
                session_id=session_id,
                new_message=adk_message
            )
            
            for event in events:
                if hasattr(event, 'content') and event.content:
                    content = event.content
                    if hasattr(content, 'role') and content.role == 'model':
                        if hasattr(content, 'parts'):
                            for part in content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text

            if not response_text:
                response_text = "I searched but couldn't find specific market data."

            final_response = (
                f"**Market Report for {crop} in {location}:**\n\n"
                f"{response_text}\n\n"
                "*(Source: Google Search via ADK)*"
            )
            
            return {"messages": [AIMessage(content=final_response)]}
            
        except Exception as e:
            print(f"Error in Market Agent: {e}")
            return {"messages": [AIMessage(content=f"I couldn't fetch the market data right now. Error: {e}")]}
