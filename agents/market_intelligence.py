import os
import uuid
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

# Google ADK imports
from google.adk import Agent, Runner
from google.adk.tools import google_search
from google.adk.sessions import InMemorySessionService
from google.adk.flows.llm_flows.contents import types as content_types

load_dotenv()

class MarketIntelligenceAgent:
    """
    Agent responsible for providing market insights and crop prices using Google ADK and Google Search.
    """
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        
        # Configure ADK Agent
        # We use a unique session ID per agent instance or handle it in invoke
        self.adk_agent = Agent(
            name="market_agent",
            model="gemini-2.0-flash-lite",
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
            crop: str = Field(description="The crop the user is asking about.")
            location: str = Field(description="The location for the market data.")

        self.parser = JsonOutputParser(pydantic_object=MarketQuery)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a market analyst. Extract the crop and location from the user's query.
            If no location is specified, default to 'Global'.
            
            User Query: {message}
            
            Conversation History:
            {chat_history}
            
            {format_instructions}
            """
        )
        self.chain = self.prompt | self.llm | self.parser

    def invoke(self, state: dict) -> dict:
        print("---MARKET INTELLIGENCE AGENT (ADK)---")
        messages = state["messages"]
        last_message = messages[-1].content
        
        # Create history string (excluding the last message which is passed separately)
        chat_history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][:-1]])
        
        try:
            # Extract intent
            query_data = self.chain.invoke({
                "message": last_message,
                "chat_history": chat_history,
                "format_instructions": self.parser.get_format_instructions()
            })
            crop = query_data["crop"]
            location = query_data["location"]
            
            # Construct query for ADK Agent
            search_query = f"latest market price of {crop} in {location} today mandi rates"
            
            # Update ADK Agent instructions to be more specific
            # Note: We are re-initializing the agent here to update instructions, or we could have set it in __init__
            # For now, let's just rely on the query being better. 
            # But actually, we can pass instructions to the Agent constructor.
            # Let's update the __init__ to include instructions.
            
            # Create a new session for each query to avoid context pollution or manage session properly
            # Using a random session ID for now as we treat each query independently here
            session_id = str(uuid.uuid4())
            user_id = "farm_ai_user" # Could be passed from state if available
            
            self.session_service.create_session_sync(
                session_id=session_id, 
                user_id=user_id,
                app_name="farm_ai"
            )
            
            # Create Content object
            adk_message = content_types.Content(
                role='user', 
                parts=[content_types.Part(text=search_query)]
            )
            
            # Run ADK Agent
            response_text = ""
            events = self.runner.run(
                user_id=user_id,
                session_id=session_id,
                new_message=adk_message
            )
            
            for event in events:
                # print(f"DEBUG: Event type: {type(event)}")
                # print(f"DEBUG: Event attrs: {dir(event)}")
                
                # Check if event has content
                if hasattr(event, 'content') and event.content:
                    content = event.content
                    # Check if content has role 'model'
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
