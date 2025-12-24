# agents/knowledge_support.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from core.memory_service import MemoryService
from core.rag_service import RAGService
from core.farm_log_manager import FarmLogManager
from core.models import FarmLog
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class ExpandedQueries(BaseModel):
    queries: List[str] = Field(description="List of 3 alternative search queries to find relevant farming info.")

class ActivityLog(BaseModel):
    """Schema for extracting farming activities."""
    activity_type: str = Field(description="The category of farming activity, e.g., 'Planting', 'Irrigation'.")
    details: str = Field(description="A concise summary of the specific activity performed.")
    timestamp_str: str = Field(description="The date of the activity in YYYY-MM-DD format. Calculate based on current date.")
    advice: str = Field(description="A friendly, brief, and helpful farming tip or warning related to this specific activity.")

class KnowledgeSupportAgent:
    """A unified agent for answering questions (RAG) and logging activities."""
    
    def __init__(self, llm: BaseLanguageModel, memory_service: MemoryService, log_manager: FarmLogManager):
        self.llm = llm
        self.memory = memory_service
        self.log_manager = log_manager
        self.rag = RAGService()
        
        # Activity Extraction Chain
        self.log_parser = JsonOutputParser(pydantic_object=ActivityLog)
        activity_prompt = ChatPromptTemplate.from_template(
            """You are a data extraction specialist. Analyze the user's message.
            
            Current Date: {current_date}
            User Message: "{message}"
            
            If the user describes a COMPLETED farming activity (e.g., "I pruned logs", "Watered today"):
            1. Extract details.
            2. specific date (YYYY-MM-DD).
            3. Generate helpful advice.
            
            {format_instructions}
            """
        )
        self.activity_chain = activity_prompt | self.llm | self.log_parser
        
        self.activity_chain = activity_prompt | self.llm | self.log_parser
        
        # Query Router (Optimization)
        class QueryRouter(BaseModel):
            intent: str = Field(description="The intent of the user. Options: 'GREETING', 'GENERAL_CHAT', 'FARMING_QUERY'.")
        
        self.router_parser = JsonOutputParser(pydantic_object=QueryRouter)
        router_prompt = ChatPromptTemplate.from_template(
            """Classify the user instructions.
            User Input: "{question}"
            
            Options:
            - GREETING: Hello, Hi, Thanks, Bye.
            - GENERAL_CHAT: How are you?, What is your name?, simple small talk.
            - FARMING_QUERY: Any question about crops, soil, chemicals, diseases, government schemes, prices, "how to", "why".
            
            {format_instructions}
            """
        )
        self.router_chain = router_prompt | self.llm | self.router_parser
        
        self.router_chain = router_prompt | self.llm | self.router_parser
        
        # Query Expansion removed for latency optimization.
        # We will use the raw user query + Hybrid Search.
        self.prompt = ChatPromptTemplate.from_template(
            """You are 'Farm-AI', a knowledgeable and encouraging farming companion (Agri-Friend).
Your goal is to provide expert advice with a warm, personal touch.

**CONTEXT:**
- **Farmer's Name:** {farmer_name}
- **Active Crops:** {active_crops}
- **Current Date:** {current_date}
- **Recent Activities:** {recent_activities}

**OFFICIAL KNOWLEDGE BASE (RAG):**
{retrieved_context}

**Conversation History:**
{chat_history}

**USER'S MESSAGE:** "{question}"

**INSTRUCTIONS:**
1. **Be Warm & Personable:** Don't just answer; connect. Use phrases like "I'm glad you asked" or "That's a great question."
2. **Contextualize:** If the user asks about crops they are growing (context above), mention their specific situation. (e.g., "Since you're growing tomatoes in Chennai, you should watch out for...")
3. **No Robot-Speak:** Avoid saying "According to the context provided" or "Based on the documents." Just give the advice naturally as an expert.
4. **Actionable & Encouraging:** End with encouragement or a practical next step.

Respond naturally:
"""
        )
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---KNOWLEDGE SUPPORT AGENT (UNIFIED: RAG + LOGGING)---")
        user_id = state["user_id"]
        ctx = self.memory.get_context(user_id)
        
        last_message = state["messages"][-1]
        user_query = last_message.content
        history_str = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][:-1]])
        detected_activity = state.get("detected_activity")
        
        # --- PATH 1: ACTIVITY LOGGING (Action) ---
        # Trigger if Profile Agent detected activity OR if the user message strongly implies action
        if detected_activity or "I " in user_query: # Simple heuristic + explicit signal
            # Try to extract activity
            try:
                print(f"--- KNOWLEDGE: Attempting to extract activity from: {user_query} ---")
                activity_data = self.activity_chain.invoke({
                    "message": user_query,
                    "current_date": ctx["current_date"],
                    "format_instructions": self.log_parser.get_format_instructions()
                })
                
                # If extraction succeeded (fields present)
                if activity_data.get('activity_type'):
                    print(f"--- KNOWLEDGE: Logging Activity: {activity_data['activity_type']} ---")
                    log_timestamp = datetime.strptime(activity_data["timestamp_str"], "%Y-%m-%d")
                    farm_log = FarmLog(
                        activity_type=activity_data["activity_type"],
                        details=activity_data["details"],
                        timestamp=log_timestamp
                    )
                    self.log_manager.add_log(user_id, farm_log)
                    return {"messages": [AIMessage(content=activity_data['advice'])]}
            except Exception as e:
                print(f"--- KNOWLEDGE: Not a valid activity or error extraction: {e} ---")
                # Fallthrough to RAG if it wasn't a loggable action
        
        # --- PATH 2: KNOWLEDGE RAG vs GENERAL CHAT (Dynamic Routing) ---
        print(f"--- KNOWLEDGE: Routing query '{user_query}'... ---")
        
        try:
            # OPTIMIZATION: Check intent before triggering expensive RAG
            route_result = self.router_chain.invoke({
                "question": user_query,
                "format_instructions": self.router_parser.get_format_instructions()
            })
            intent = route_result.get("intent", "FARMING_QUERY")
            print(f"--- KNOWLEDGE: Identified Intent: {intent} ---")
            
            if intent in ["GREETING", "GENERAL_CHAT"]:
                # Fast Path: No RAG
                print("--- KNOWLEDGE: Taking FAST PATH (No RAG) ---")
                fast_prompt = ChatPromptTemplate.from_template(
                    """You are 'Farm-AI', a warm and friendly farming companion.
                    
                    **CONTEXT - RECENT CONVERSATION:**
                    {chat_history}
                    
                    User said: "{question}"
                    
                    Respond comfortably and enthusiastically. 
                    - If they refer to previous topics (like "he", "it", "that"), use the CONTEXT to understand who or what they mean.
                    - **CRITICAL: DO NOT GREET** unless the user explicitly said "Hello" or "Hi" in this message.
                    - If they are just continuing the chat (e.g., "he is sad", "ok", "tell me more"), jump straight to the response.
                    - Keep it conversational and brief.
                    """
                )
                fast_chain = fast_prompt | self.llm
                response = fast_chain.invoke({
                    "question": user_query,
                    "chat_history": history_str
                })
                return {"messages": [AIMessage(content=response.content)]}
                
        except Exception as e:
            print(f"--- KNOWLEDGE: Router failed ({e}), defaulting to RAG ---")

        # --- SLOW PATH: RAG ---
        print("--- KNOWLEDGE: Proceeding to RAG (Farming Query) ---")
        
        # --- SLOW PATH: RAG ---
        print("--- KNOWLEDGE: Proceeding to RAG (Farming Query) ---")
        
        # 1. Direct Search (Optimized)
        queries = [user_query]
        print(f"--- RAG: Using Query: {queries} ---")

        # 2. Hybrid Search for each query
        retrieved_docs = []
        for q in queries:
            docs = self.rag.hybrid_search(q, k=2)
            retrieved_docs.extend(docs)
            
        # Deduplicate by content
        unique_docs = {}
        for d in retrieved_docs:
            unique_docs[d.page_content] = d
        
        final_docs = list(unique_docs.values())[:4] # Top 4 unique
        context_str = "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')}]\n{d.page_content}" for d in final_docs])
        
        if not context_str:
            context_str = "No specific official documents found."

        # LOGGING SOURCES FOR USER VERIFICATION
        print("--- RAG CITATIONS ---")
        for i, d in enumerate(final_docs):
            source = d.metadata.get('source', 'Unknown')
            page = d.metadata.get('page', 'N/A')
            print(f"[{i+1}] Source: {source} | Page: {page}")
        print("---------------------")

        response = self.chain.invoke({
            "farmer_name": ctx["farmer_name"],
            "active_crops": ctx["active_crops"],
            "current_date": ctx["current_date"],
            "recent_activities": ctx.get("memory_narrative", "No recent activities."),
            "retrieved_context": context_str,
            "chat_history": history_str,
            "question": user_query
        })

        return {"messages": [AIMessage(content=response.content)]}
