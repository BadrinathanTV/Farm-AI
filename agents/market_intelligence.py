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
        self.mcp = MCPWrapper(server_url="http://localhost:8000/sse")
        
        # Summary Chain to format the search results
        self.summary_prompt = ChatPromptTemplate.from_template(
            """You are a helpful market assistant.
            The user asked: "{query}"
            Current Date: {current_date}
            
            Here are the search results from DuckDuckGo:
            {search_results}
            
            Please summarize these results into a clear, helpful answer for the farmer.
            - **CRITICAL:** Only provide prices that are from {current_date} or very recent (last 2-3 days).
            - If the data is old (e.g., from last year or months ago), EXPLICITLY state that "Current prices are not available, but here is older data...".
            - Focus on finding the price if available.
            - Cite the source links if useful.
            """
        )
        self.summary_chain = self.summary_prompt | self.llm

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
            
            # Construct query for search with dynamic date (Keyword optimized)
            today_str = datetime.now().strftime("%d %B %Y")
            search_query = f"{crop} price {location} {today_str} mandi rates"
            
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

