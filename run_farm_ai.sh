#!/bin/bash
# Startup script for Farm-AI (App + MCP Server)

BLACK='\033[0;30m'
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}--- Starting Farm-AI Ecosystem ---${NC}"

# Function to clean up background processes on exit
cleanup() {
    echo -e "\n${RED}Shutting down Farm-AI...${NC}"
    if [ ! -z "$MCP_PID" ]; then
        echo "Stopping MCP Server (PID: $MCP_PID)..."
        kill $MCP_PID
    fi
    exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# 1. Start MCP Server in the background
echo -e "${GREEN}[1/2] Starting DuckDuckGo MCP Server (Port 8000)...${NC}"
# Use uv run to ensure dependencies are loaded
uv run tools/mcp_server_ddg.py > mcp_server.log 2>&1 &
MCP_PID=$!
echo "MCP Server running with PID: $MCP_PID"

# Wait a moment for server to initialize
sleep 3

# 2. Start Streamlit App
echo -e "${GREEN}[2/2] Launching Streamlit App...${NC}"
echo -e "${BLUE}Logs are saved to 'mcp_server.log'${NC}"
uv run streamlit run app.py

cleanup
