import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

async def run():
    url = "http://localhost:8000/sse"
    print(f"--- Connecting to DuckDuckGo MCP Server at {url} ---")
    
    try:
        async with sse_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # List available tools
                tools = await session.list_tools()
                print("\n--- Available Tools ---")
                for tool in tools.tools:
                    print(f"- {tool.name}: {tool.description}")

                # Execute a search
                print("\n--- Testing 'web_search' ---")
                query = "latest market price of tomatoes in Chennai on 20 December 2025 mandi rates"
                print(f"Query: {query}")
                
                # Call the tool
                result = await session.call_tool("web_search", arguments={"query": query})
                
                # Print the result text
                print("\n--- Search Results ---")
                if result.content:
                    print(result.content[0].text)
                else:
                    print("No results found.")
                    
    except Exception as e:
        print(f"\n\033[91mConnection Failed:\033[0m {e}")
        print("Ensure the server is running: 'uv run tools/mcp_server_ddg.py'")

if __name__ == "__main__":
    asyncio.run(run())
