# agents/knowledge_support.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from core.memory_service import MemoryService

class KnowledgeSupportAgent:
    """A fully context-aware agent with access to the user's profile and farm logs via MemoryService."""
    
    def __init__(self, llm: BaseLanguageModel, memory_service: MemoryService):
        self.llm = llm
        self.memory = memory_service
        self.prompt = ChatPromptTemplate.from_template(
            """You are a friendly and knowledgeable AI farming assistant.

**CONTEXT:**
- **Farmer's Name:** {farmer_name}
- **Active Crops:** {active_crops}
- **Current Date:** {current_date}
- **Recent Activities:** {recent_activities}

**Conversation History:**
{chat_history}

**USER'S MESSAGE:** "{question}"

**INSTRUCTIONS:**
1. If the user says "hello" or a greeting, respond warmly and offer to help with farming questions.
2. If the user asks about their farm, crops, or activities, use the context above.
3. For general farming questions (pests, techniques, best practices), provide helpful expert advice.
4. Be concise, friendly, and practical.

Respond naturally:
"""
        )
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---KNOWLEDGE SUPPORT AGENT (MEMORY SERVICE)---")
        user_id = state["user_id"]
        ctx = self.memory.get_context(user_id)

        history_str = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][:-1]])
        last_message = state["messages"][-1]
        
        response = self.chain.invoke({
            "farmer_name": ctx["farmer_name"],
            "active_crops": ctx["active_crops"],
            "current_date": ctx["current_date"],
            "recent_activities": ctx["recent_activities"],
            "chat_history": history_str,
            "question": last_message.content
        })

        return {"messages": [AIMessage(content=response.content)]}
