#!/bin/bash

# Start the MCP Server in the background
echo "Starting DuckDuckGo MCP Server on port 8000..."
# Using python directly as dependencies will be installed in the environment
python tools/mcp_server_ddg.py > mcp_server.log 2>&1 &
MCP_PID=$!

# Wait for a few seconds to ensure MCP server starts
sleep 3

# Check if MCP server is running
if ! kill -0 $MCP_PID > /dev/null 2>&1; then
    echo "MCP Server failed to start. Check mcp_server.log for details."
    cat mcp_server.log
    exit 1
fi

echo "MCP Server started with PID $MCP_PID"

# Start Streamlit App
echo "Starting Streamlit App..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# When Streamlit exits, kill the MCP server
kill $MCP_PID
