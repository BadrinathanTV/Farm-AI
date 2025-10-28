# agents/formatter.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from core.profile_manager import ProfileManager

class FormatterAgent:
    """The final 'voice' of the AI, ensuring conversations are natural and fluid."""
    def __init__(self, llm: BaseLanguageModel, profile_manager: ProfileManager):
        self.llm = llm
        self.profile_manager = profile_manager
        self.prompt = ChatPromptTemplate.from_template(
            """You are the friendly, final voice of a Farm AI assistant.
Your user is a farmer named {farmer_name}.

**Your Task:**
Your job is to take the raw output from an internal expert agent and rephrase it into a natural, helpful, and **contextually appropriate** final answer.

**Full Conversation History (for context):**
{chat_history}

**Raw Output from Internal Expert:**
"{raw_output}"

**Strict Rules for Your Response:**
1.  **Analyze the conversation history.** Your response must flow naturally from the last message.
2.  **DO NOT use repetitive greetings.** If you have already greeted the user, get straight to the point in a friendly way.
3.  **Maintain a consistent persona:** You are a single, helpful assistant. Your response should feel like a continuation of the conversation.
4.  Address the farmer by name, but naturally, not every single time.

Rephrase the raw output into a polished final response.
"""
        )
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---FORMATTER AGENT (CONTEXT-AWARE)---")
        user_id = state["user_id"]
        profile = self.profile_manager.load_profile(user_id)
        
        raw_output_message = state["messages"][-1]
        
        # Create a clean, readable version of the chat history
        history_str = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][:-1]])

        response = self.chain.invoke({
            "farmer_name": profile.full_name or "Farmer",
            "chat_history": history_str,
            "raw_output": raw_output_message.content
        })
        
        # Replace the raw message with the polished one
        final_messages = state["messages"][:-1] + [response]

        return {"messages": final_messages}

