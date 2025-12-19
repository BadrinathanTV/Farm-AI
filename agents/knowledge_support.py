# agents/knowledge_support.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from core.profile_manager import ProfileManager
from core.farm_log_manager import FarmLogManager
from datetime import datetime

class KnowledgeSupportAgent:
    """A fully context-aware agent with access to the user's profile and farm logs."""
    def __init__(self, llm: BaseLanguageModel, profile_manager: ProfileManager, log_manager: FarmLogManager):
        self.llm = llm
        self.profile_manager = profile_manager
        self.log_manager = log_manager
        # --- THE FINAL, ANALYTICAL PROMPT ---
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful and highly analytical AI assistant.
Your primary function is to answer the user's question based ONLY on the provided context.
RUN FASTERRR , WE NEED LOW LATECY !!
**CONTEXT:**
1.  **Farmer's Name:** {farmer_name}
2.  **Current Date:** {current_date}
3.  **Recent Activities Log:**
    {recent_activities}
4.  **Full Conversation History:**
    {chat_history}

**USER'S LATEST QUESTION:** "{question}"

**STRICT INSTRUCTIONS:**
-   **If the user's question is about the conversation itself** (e.g., "what did I ask you?", "what have we talked about?"), you **MUST** summarize ONLY the user's previous questions from the 'HUMAN' parts of the chat history.
-   For all other questions, provide a direct and helpful answer using the provided context.
-   Do not invent any information or answer questions outside of the provided context.

Based on these instructions, provide a direct and accurate answer.
"""
        )
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---KNOWLEDGE SUPPORT AGENT (ANALYTICAL MODE)---")
        user_id = state["user_id"]
        profile = self.profile_manager.load_profile(user_id)
        
        recent_logs = self.log_manager.get_recent_logs(user_id, limit=10)
        if recent_logs:
            activities_str = "\n".join([f"- On {log.timestamp.strftime('%Y-%m-%d')}, they performed: {log.details}" for log in recent_logs])
        else:
            activities_str = "No recent activities logged."

        history_str = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][:-1]])
        last_message = state["messages"][-1]
        
        response = self.chain.invoke({
            "farmer_name": profile.full_name or "Farmer",
            "current_date": datetime.now().strftime("%A, %B %d, %Y"),
            "recent_activities": activities_str,
            "chat_history": history_str,
            "question": last_message.content
        })

        return {"messages": [AIMessage(content=response.content)]}

