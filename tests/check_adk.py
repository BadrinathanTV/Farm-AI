try:
    import google.adk
    print("Imported google.adk")
    print(dir(google.adk))
except ImportError as e:
    print(f"Failed to import google.adk: {e}")

try:
    from google.adk import agents
    print("Imported google.adk.agents")
except ImportError as e:
    print(f"Failed to import google.adk.agents: {e}")

try:
    from google.adk import tools
    print("Imported google.adk.tools")
except ImportError as e:
    print(f"Failed to import google.adk.tools: {e}")
