# agents/knowledge_support.py

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from core.memory_service import MemoryService
from core.rag_service import RAGService
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class ExpandedQueries(BaseModel):
    queries: List[str] = Field(description="List of 3 alternative search queries to find relevant farming info.")

class KnowledgeSupportAgent:
    """A fully context-aware agent with access to the user's profile and farm logs via MemoryService."""
    
    def __init__(self, llm: BaseLanguageModel, memory_service: MemoryService):
        self.llm = llm
        self.memory = memory_service
        self.rag = RAGService()
        
        self.query_expander = (
            ChatPromptTemplate.from_template(
                """You are a helpful farming assistant.
                Generate 3 different search queries to find relevant information for the user's question.
                Focus on:
                1. Synonyms (e.g., "dying" -> "wilt", "necrosis")
                2. Technical terms (e.g., "pests" -> "aphids", "thrips")
                3. The core intent.
                
                User Question: {question}
                
                {format_instructions}
                """
            )
            | self.llm
            | JsonOutputParser(pydantic_object=ExpandedQueries)
        )
        self.prompt = ChatPromptTemplate.from_template(
            """You are a friendly and knowledgeable AI farming assistant.

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
1. If the user says "hello" or a greeting, respond warmly and offer to help with farming questions.
2. If the user asks about their farm, crops, or activities, use the context above.
3. For general farming questions (pests, techniques, best practices), provide helpful expert advice.
4. Be concise, friendly, and practical.

Respond naturally:
"""
        )
        self.chain = self.prompt | self.llm

    def invoke(self, state: dict) -> dict:
        print("---KNOWLEDGE SUPPORT AGENT (RAG ENABLED)---")
        user_id = state["user_id"]
        ctx = self.memory.get_context(user_id)

        history_str = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in state["messages"][:-1]])
        last_message = state["messages"][-1]
        user_query = last_message.content
        
        # 1. Query Expansion
        try:
            expanded = self.query_expander.invoke({
                "question": user_query,
                "format_instructions": JsonOutputParser(pydantic_object=ExpandedQueries).get_format_instructions()
            })
            queries = expanded.get("queries", [user_query])
            print(f"--- RAG: Expanded Queries: {queries} ---")
        except Exception as e:
            print(f"--- RAG: Expansion failed ({e}), using original query. ---")
            queries = [user_query]

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
