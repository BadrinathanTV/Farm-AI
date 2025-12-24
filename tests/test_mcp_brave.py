import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Get API Key from environment
api_key = os.environ.get("BRAVE_API_KEY")

async def run():
    if not api_key:
        print("\n\033[91mError: BRAVE_API_KEY not found in environment.\033[0m")
        print("Please add it to your .env file or export it:")
        print("  export BRAVE_API_KEY=your_key_here")
        return

    # Define the server parameters
    # We use npx to run the Brave Search MCP server directly
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-brave-search"],
        env={"BRAVE_API_KEY": api_key}
    )

    print("--- Connecting to Brave Search MCP Server ---")
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()

                # List available tools
                tools = await session.list_tools()
                print("\n--- Available Tools ---")
                for tool in tools.tools:
                    print(f"- {tool.name}: {tool.description}")

                # Execute a search
                print("\n--- Testing 'brave_web_search' ---")
                query = "current tomato price in Chennai Koyambedu market"
                print(f"Query: {query}")
                
                # Call the tool
                result = await session.call_tool("brave_web_search", arguments={"q": query})
                
                # Print the result text
                print("\n--- Search Results ---")
                if result.content:
                    print(result.content[0].text)
                else:
                    print("No results found.")
                    
    except Exception as e:
        print(f"\n\033[91mConnection Failed:\033[0m {e}")
        print("Ensure you have 'npx' installed and the API key is valid.")

if __name__ == "__main__":
    asyncio.run(run())
