from fastmcp import FastMCP
import asyncio
import threading
import time
import requests

mcp = FastMCP("Test")

@mcp.tool()
def hello() -> str:
    return "world"

def run_server():
    try:
        mcp.run(transport="streamable-http", port=8001)
    except SystemExit:
        pass

if __name__ == "__main__":
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(3)
    
    print("Checking endpoints...")
    try:
        resp = requests.get("http://localhost:8001/sse")
        print(f"GET /sse: {resp.status_code}")
    except:
        print("GET /sse failed")

    try:
        resp = requests.post("http://localhost:8001/messages")
        print(f"POST /messages: {resp.status_code}")
    except:
        print("POST /messages failed")
        
    try:
        resp = requests.get("http://localhost:8001/")
        print(f"GET /: {resp.status_code}")
    except:
        print("GET / failed")
