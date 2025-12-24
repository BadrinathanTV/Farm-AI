# agents/market_intelligence.py

import uuid
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from core.mcp_wrapper import MCPWrapper

from langchain_core.messages import get_buffer_string
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
        
        # Configure MCP Wrapper for DuckDuckGo (SSE/HTTP)
        self.mcp = MCPWrapper(server_url="http://localhost:8000/mcp")
        
        # Summary Chain to format the search results
        self.summary_prompt = ChatPromptTemplate.from_template(
            """You are a helpful market assistant.
            The user asked: "{query}"
            Current Date: {current_date}
            
            Here are the search results from DuckDuckGo:
            {search_results}
            
            Please summarize these results into a clear, helpful answer for the farmer.
            - **CRITICAL:** Only provide prices that are EXACTLY stated in the text for {current_date} or very recent (last 2-3 days).
            - **ANTI-HALLUCINATION:** If the text does NOT explicitly state the price for this crop in this location, say: "I couldn't find the exact current daily rate." Do NOT guess or use data from other cities.
            - Focus on finding the price if available.
            - Cite the source links if useful.
            """
        )
        self.summary_chain = self.summary_prompt | self.llm

        class MarketQuery(BaseModel):
            crop: str = Field(description="The crop the user is asking about. If user says 'my crops', use the active crops from context.")
            location: str = Field(description="The location for the market data.")
            search_query: str = Field(description="The BEST search query. Must include Location + Country + 'mandi rates' or 'market price'.")

        self.parser = JsonOutputParser(pydantic_object=MarketQuery)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a smart market analyst. 
            1. **Analyze Location (CRITICAL):** 
               - **Rule A:** Did the user mention a place in the current message? (e.g., "in Gurugram") -> **USE THAT.**
               - **Rule B:** Did they mention a place in the *Recent History* just now? -> **USE THAT.**
               - **Rule C:** If flows A & B fail, ONLY THEN use Profile Location: {location}.
            
            2. **Infer Major Market (Mandi):** 
               - Infer the largest wholesale market (Mandi) or APMC for the detected location.
               - E.g. If location is "Madurai", inferred market is "Mattuthavani".
               - If the location is a small town, use the nearest District Mandi.
            
            3. **Construct Search Query:**
               - Format: `[Inferred Market Name] [Location] [Crop] price today daily market rate`
               - **NO SITE RESTRICTION:** Do NOT limit to specific sites.
               - **APPEND NEGATIVE FILTERS:** You MUST append this string to remove noise:
                 `-site:wikipedia.org -site:quora.com -site:facebook.com -site:youtube.com -site:instagram.com -news`
               
            4. **Goal:** Find ANY reliable source (government or private) that has the data.
            
            **User's Context:**
            - Location: {location}
            ...
            **Recent History:**
            {chat_history}
            
            User Query: {message}

            **Crop Logic:**
            1. User Mentioned? -> Use that.
            2. Missing/Implicit ("it", "price")? -> YOU MUST FETCH IT FROM *RECENT HISTORY*.
            3. *Example:* If history says "User: Tomato price?", then "User: in Bengaluru?", the crop is "Tomato".
            
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
            today_str = datetime.now().strftime("%d %B %Y")
            
            # Use chat history to resolve "it" or implicit crop references
            # messages[:-1] skips the current user query which is already in 'message'
            history_str = get_buffer_string(messages[:-1])
            
            query_data = self.chain.invoke({
                "message": last_message,
                "chat_history": history_str,
                "location": ctx["location"],
                "active_crops": ctx["active_crops"],
                "current_date": today_str,
                "format_instructions": self.parser.get_format_instructions()
            })
            crop = query_data["crop"]
            location = query_data["location"]
            search_query = query_data["search_query"]
            
            # Execute Search via MCP
            print(f"--- MARKETS: Searching for '{search_query}' ---")
            search_results_str = self.mcp.execute_tool("web_search", {"query": search_query})
            
            # --- SCRAPING LOGIC ---
            top_content = ""
            try:
                # Extract all links using regex
                import re
                # Pattern to find all links in the formatted string "Link: (url)"
                links = re.findall(r"Link: (https?://\S+)", search_results_str)
                
                # Try scraping up to 3 unique links
                unique_links = []
                for link in links:
                    if link not in unique_links:
                        unique_links.append(link)
                
                max_scrapes = 3
                links_to_try = unique_links[:max_scrapes]
                
                scraping_results = []
                
                print(f"--- MARKETS: Found {len(unique_links)} links. Trying top {len(links_to_try)}... ---")
                
                for i, link in enumerate(links_to_try):
                    print(f"--- MARKETS: Fetching result {i+1}/{len(links_to_try)}: '{link}' ---")
                    try:
                        content = self.mcp.execute_tool("fetch_page", {"url": link})
                        
                        # Check if content indicates an error (the tool returns error strings on exception)
                        if content and "Failed to fetch page" not in content:
                            scraping_results.append(f"\n\n--- Content from {link} ---\n{content}\n")
                            # If we have a good result, we might not need many more, but let's get up to 2 for robustness
                            if len(scraping_results) >= 2:
                                break
                        else:
                            print(f"--- MARKETS: Failed to fetch {link} (Tool returned error) ---")
                            
                    except Exception as e:
                        print(f"--- MARKETS: Error fetching {link}: {e} ---")
                
                if scraping_results:
                    top_content = "".join(scraping_results)
                else:
                    print("--- MARKETS: All scraping attempts failed or no links found ---")

            except Exception as e:
                print(f"--- MARKETS SCRAPE ERROR: {e} ---")
            
            # Summarize with LLM
            final_response_msg = self.summary_chain.invoke({
                "query": search_query,
                "search_results": search_results_str + top_content,
                "current_date": today_str
            })
            
            final_response = final_response_msg.content + "\n\n*(Source: DuckDuckGo via MCP)*"
            
            return {"messages": [AIMessage(content=final_response)]}
            
        except Exception as e:
            print(f"Error in Market Agent: {e}")
            return {"messages": [AIMessage(content=f"I couldn't fetch the market data right now. Error: {e}")]}

