import asyncio
import nest_asyncio
from typing import Any, Dict, Optional
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# Apply nest_asyncio to allow nested event loops (crucial for Streamlit)
nest_asyncio.apply()

class MCPWrapper:
    """
    A synchronous wrapper for MCP Servers via Streamable HTTP.
    Manages the lifecycle of the connection for each call.
    Uses nest_asyncio to support execution within existing event loops.
    """
    def __init__(self, server_url: str):
        self.server_url = server_url
        print(f"--- MCP WRAPPER: Initialized for {server_url} ---")

    async def _execute_async(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        print(f"--- MCP WRAPPER: Connecting to {self.server_url} for tool '{tool_name}' ---")
        try:
            # Connect to the Streamable HTTP endpoint
            # Yields: (read, write, get_session_id_callback)
            async with streamable_http_client(self.server_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Log tools for debugging
                    # tools = await session.list_tools()
                    # print(f"--- MCP WRAPPER: Tools available: {[t.name for t in tools.tools]} ---")
                    
                    print(f"--- MCP WRAPPER: Calling tool '{tool_name}' with args {arguments} ---")
                    result = await session.call_tool(tool_name, arguments=arguments)
                    
                    if result.content:
                        content_text = result.content[0].text
                        print(f"--- MCP WRAPPER: Success. Result length: {len(content_text)} chars ---")
                        return content_text
                    
                    print("--- MCP WRAPPER: No content returned ---")
                    return "No results returned from tool."
        except Exception as e:
            print(f"--- MCP WRAPPER INTERNAL ERROR: {type(e).__name__}: {e} ---")
            if hasattr(e, 'exceptions'):
                for i, sub_exc in enumerate(e.exceptions):
                    print(f"--- Sub-exception {i}: {type(sub_exc).__name__}: {sub_exc} ---")
            elif hasattr(e, '__cause__') and e.__cause__:
                 print(f"--- Cause: {type(e.__cause__).__name__}: {e.__cause__} ---")
            raise e

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Executes an MCP tool synchronously via SSE.
        Handles asyncio event loops gracefully.
        """
        try:
            # Check if there is a running loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We are in a running loop, use the loop to run the coroutine
                # nest_asyncio makes this possible if asyncio.run was called,
                # but if we are deeply nested, we might needed ensure_future?
                # Actually, nest_asyncio allows asyncio.run() to be called even if a loop is running.
                return asyncio.run(self._execute_async(tool_name, arguments))
            else:
                return asyncio.run(self._execute_async(tool_name, arguments))

        except Exception as e:
            error_msg = f"MCP Tool Execution Failed: {str(e)}"
            print(f"--- MCP WRAPPER ERROR: {error_msg} ---")
            return error_msg
